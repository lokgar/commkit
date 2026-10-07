"""Blind frequency-domain block equalizer engine (FDAF) backing blind.py's
block_cma / block_rde.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...backend import to_device
from ...logger import logger
from ..result import EqualizerResult, _log_equalizer_exit
from ._engine import (
    _block_loop,
    _block_state,
    _fdaf_forward,
    _fdaf_gradient_update,
    _prepare_block,
)


def _block_fdaf_blind(
    kind: str,
    samples: Any,
    *,
    num_taps: int,
    sps: int,
    step_size: float,
    block_size: int,
    r2: float,
    radii_np: np.ndarray | None,
    initial_taps: Any,
    state: Any,
    pad_mode: str,
    pilot_ref: Any,
    pilot_mask: np.ndarray | None,
    pilot_gain_db: float,
    c_ps: Any,
    cuda_graph: bool,
    name: str,
) -> EqualizerResult:
    """Shared overlap-save FDAF engine for ``block_cma``/``block_rde``.

    Blind, phase-directed adaptation with no CPR - the per-block error is the
    Godard (``kind='cma'``) or nearest-ring (``kind='rde'``) gradient, with
    pilot positions overridden by the LMS residual.  ``r2`` is the Godard radius
    (CMA) and ``radii_np`` the unique ring radii (RDE).
    """
    use_pilots = pilot_ref is not None and pilot_mask is not None
    equalizer = f"block_{kind}"
    run = _prepare_block(
        samples,
        equalizer=equalizer,
        sps=sps,
        num_taps=num_taps,
        block_size=block_size,
        initial_taps=initial_taps,
        state=state,
        pad_mode=pad_mode,
        name=name,
        cpu_hint=f"{kind}()",
        pilot_mask=pilot_mask if use_pilots else None,
        pilot_gain_db=pilot_gain_db,
    )
    xp, C, n_sym, sps = run.xp, run.C, run.n_sym, run.sps
    logger.info(
        "%s: C=%s, num_taps=%s, sps=%s, block_size=%s, fftsize=%s, mu=%s, "
        "n_sym=%s, pilot_aided=%s",
        name,
        C,
        run.num_taps,
        sps,
        run.block_size,
        run.fftsize,
        step_size,
        n_sym,
        use_pilots,
    )

    if use_pilots:
        pref = xp.asarray(
            np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
        )
        if pref.ndim == 1:
            pref = xp.tile(pref[None, :], (C, 1))
        if c_ps is not None:
            pref = (pref * xp.complex64(c_ps)).astype(xp.complex64)
        pmask_dev = xp.asarray(np.asarray(pilot_mask).astype(bool))
    if kind == "rde":
        radii = xp.asarray(np.asarray(radii_np, dtype=np.float64))

    e_scatter = xp.zeros((C, run.fftsize), dtype=xp.complex64)
    y_all = xp.empty((C, n_sym), dtype=xp.complex64)
    e_all = xp.empty((C, n_sym), dtype=xp.complex64)
    y_ws = xp.empty((C, run.block_size), dtype=xp.complex64)
    e_ws = xp.empty((C, run.block_size), dtype=xp.complex64)

    def run_block(B: int, b_start: int) -> None:
        """Forward FDAF, blind error, in-place gradient update.

        Results go to the persistent ``y_ws``/``e_ws`` buffers so the block is
        CUDA-graph capturable; pilot positions use the LMS residual.
        """
        y_block, X_fd = _fdaf_forward(run.h, run.x_win, run.fftsize, sps, B, xp)
        abs2 = xp.real(y_block * xp.conj(y_block))  # (C, B) strict-real |y|^2
        if kind == "cma":
            e = y_block * (xp.float32(r2) - abs2)
        else:  # rde - nearest ring radius per symbol
            abs_y = xp.sqrt(abs2)
            rd = radii[
                xp.argmin(xp.abs(abs_y[:, :, None] - radii[None, None, :]), axis=-1)
            ]
            e = y_block * (rd.astype(xp.float32) ** 2 - abs2)
        if use_pilots:
            pm = pmask_dev[b_start : b_start + B][None, :]
            e = xp.where(pm, pref[:, b_start : b_start + B] - y_block, e)
        y_ws[:, :B] = y_block
        e_ws[:, :B] = e
        _fdaf_gradient_update(
            run.h, X_fd, e, e_scatter, sps, B, run.num_taps, step_size, xp
        )

    def store(b_start: int, b_end: int, B: int) -> None:
        y_all[:, b_start:b_end] = y_ws[:, :B]
        e_all[:, b_start:b_end] = e_ws[:, :B]

    snap: list[np.ndarray] = []

    def at_resume() -> None:
        snap.append(to_device(run.h, "cpu").copy())

    _block_loop(
        run,
        run_block=run_block,
        store=store,
        capturable=lambda B, b_start: B == run.block_size,
        # Pilot indexing depends on b_start, which a replayed graph freezes.
        use_graph=(
            cuda_graph
            and xp is not np
            and not use_pilots
            and n_sym // run.block_size >= 2  # >= 1 warmup + 1 captured block
        ),
        name=name,
        at_resume=at_resume,
    )

    h = run.h
    if not bool(xp.isfinite(h).all()):
        raise RuntimeError(
            f"{name} diverged (step_size={step_size}, block_size={block_size}). "
            f"step_size is on the same scale as {kind}(); because the weights are "
            f"frozen across the block the stability ceiling is ~{block_size}x lower. "
            f"Reduce step_size (e.g. {step_size / 2:.2e}, then keep halving)."
        )
    if run.was_1d:
        y_out, e_out, W_out = y_all[0], e_all[0], h[0, 0]
    else:
        y_out, e_out, W_out = y_all, e_all, h
    result = EqualizerResult(
        y_hat=y_out,
        weights=W_out,
        error=e_out,
        weights_history=None,
        num_train_symbols=0,
        input_norm_factor=run.eq_norm,
        state=_block_state(
            run, equalizer=equalizer, cpr=None, weights=snap[0], carrier=None
        ),
    )
    return _log_equalizer_exit(result, name=name, check_convergence=True)
