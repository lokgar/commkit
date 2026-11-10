"""Blind frequency-domain block equalizers (private): ``block_cma`` and ``block_rde``.

The frequency-domain block versions of ``cma`` and ``rde`` and the FDAF loop
behind them, :func:`_block_fdaf_blind`; re-exported by
:mod:`commkit.equalization`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...backend import ArrayType, to_device
from ...core._signal_adapter import adapt_signal, require_integer_sps
from ...core.signal import Signal
from ...logger import logger
from .._common import _godard_radius, _rde_ring_radii, _rde_rings
from .._sequential._blind import _check_pilots
from ..pilots import EqualizerPilots
from ..result import (
    EqualizerResult,
    EqualizerState,
    _attach_equalized_signal,
    _log_equalizer_exit,
)
from ._engine import (
    _block_loop,
    _block_state,
    _fdaf_forward,
    _fdaf_gradient_update,
    _prepare_block,
)

__all__ = ["block_cma", "block_rde"]

_RDE_KERNEL: dict[str, Any] = {}


def _cupy_rde_error() -> Any:
    """Fused CuPy kernel of the RDE error ``y · (r_k² - |y|²)``.

    The ring of each symbol is counted against the squared midpoints of the
    ``(2, K)`` table of ``_rde_rings`` and the error formed, in one launch
    instead of a search and three ufunc passes.
    """
    if "k" not in _RDE_KERNEL:
        import cupy as cp

        _RDE_KERNEL["k"] = cp.ElementwiseKernel(
            "complex64 y, float32 a2, raw float32 rings, int32 K",
            "complex64 e",
            """
            int k = 0;
            for (int j = 0; j < K - 1; ++j) k += a2 > rings[K + j];
            e = y * (rings[k] - a2);
            """,
            "commkit_rde_error",
        )
    return _RDE_KERNEL["k"]


def _block_fdaf_blind(
    kind: str,
    samples: Any,
    *,
    num_taps: int,
    sps: int,
    step_size: float | None,
    block_size: int,
    r2: float,
    radii_np: np.ndarray | None,
    initial_taps: Any,
    state: Any,
    pad_mode: str,
    pilot_ref: Any,
    pilot_mask: np.ndarray | None,
    pilot_gain_db: float,
    cuda_graph: bool,
    name: str,
) -> EqualizerResult:
    """Shared overlap-save FDAF engine for ``block_cma``/``block_rde``.

    Blind, phase-directed adaptation with no CPR - the per-block error is the
    Godard (``kind='cma'``) or nearest-ring (``kind='rde'``) gradient, with
    pilot positions overridden by the LMS residual.  ``r2`` is the Godard radius
    (CMA) and ``radii_np`` the unique ring radii (RDE), decided by their
    squared midpoints (``_rde_rings``).  ``step_size=None``
    is the default of both, ``min(1e-3, 0.5 / block_size) / sps``.
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
    mu = min(1e-3, 0.5 / run.block_size) / sps if step_size is None else step_size
    logger.info(
        "%s: C=%s, num_taps=%s, sps=%s, block_size=%s, fftsize=%s, mu=%s, "
        "n_sym=%s, pilot_aided=%s",
        name,
        C,
        run.num_taps,
        sps,
        run.block_size,
        run.fftsize,
        mu,
        n_sym,
        use_pilots,
    )

    if use_pilots:
        pref = xp.asarray(
            np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
        )
        if pref.ndim == 1:
            pref = xp.tile(pref[None, :], (C, 1))
        pmask_dev = xp.asarray(np.asarray(pilot_mask).astype(bool))
    if kind == "rde":
        assert radii_np is not None
        rings = xp.asarray(_rde_rings(radii_np))
        ring2, bounds2 = rings[0], rings[1, :-1]
        rde_error = _cupy_rde_error() if xp is not np else None

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
            if rde_error is not None:
                e = rde_error(y_block, abs2, rings, np.int32(rings.shape[1]))
            else:
                e = y_block * (ring2[np.searchsorted(bounds2, abs2)] - abs2)
        if use_pilots:
            pm = pmask_dev[b_start : b_start + B][None, :]
            e = xp.where(pm, pref[:, b_start : b_start + B] - y_block, e)
        y_ws[:, :B] = y_block
        e_ws[:, :B] = e
        _fdaf_gradient_update(run.h, X_fd, e, e_scatter, sps, B, run.num_taps, mu, xp)

    def store(b_start: int, b_end: int, B: int) -> None:
        """Store the block's outputs and errors."""
        run.put(y_all, y_ws, B)
        run.put(e_all, e_ws, B)

    snap: list[np.ndarray] = []

    def at_resume() -> None:
        """Snapshot the weights where a later call resumes."""
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
            f"{name} diverged (step_size={mu}, block_size={block_size}). "
            f"step_size is on the same scale as {kind}(); because the weights are "
            f"frozen across the block the stability ceiling is ~{block_size}x lower. "
            f"Reduce step_size (e.g. {mu / 2:.2e}, then keep halving)."
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
    return _log_equalizer_exit(result, name=name)


# -----------------------------------------------------------------------------
# BLOCK BLIND EQUALIZERS (Signal-aware)
# -----------------------------------------------------------------------------


def _block_blind(
    kind: str,
    samples: ArrayType | Signal,
    *,
    num_taps: int,
    sps: int | None,
    step_size: float | None,
    block_size: int,
    constellation: Any,
    initial_taps: ArrayType | None,
    pilots: EqualizerPilots | None,
    pad_mode: str,
    state: EqualizerState | None,
    cuda_graph: bool,
) -> EqualizerResult:
    """The body of :func:`block_cma` and :func:`block_rde` (``kind``)."""
    name = f"block_{kind}()"
    signal_adapter = adapt_signal(samples, function_name=name)
    samples = signal_adapter.array
    sps = require_integer_sps(signal_adapter.resolve_fact("sps", sps), name)
    constellation = signal_adapter.resolve_choice("constellation", constellation)
    pilot_ref, pilot_mask, pilot_gain_db = _check_pilots(
        signal_adapter.array, sps, pilots, name, state
    )
    if kind == "cma":
        r2, radii_np = _godard_radius(constellation), None
    else:
        r2, radii_np = 1.0, _rde_ring_radii(constellation)
    label = f"Block-{kind.upper()}"
    result = _block_fdaf_blind(
        kind,
        samples,
        num_taps=num_taps,
        sps=sps,
        step_size=step_size,
        block_size=block_size,
        r2=r2,
        radii_np=radii_np,
        initial_taps=initial_taps,
        state=state,
        pad_mode=pad_mode,
        pilot_ref=pilot_ref,
        pilot_mask=pilot_mask,
        pilot_gain_db=pilot_gain_db,
        cuda_graph=cuda_graph,
        name=label if pilot_ref is None else f"{label}(PA)",
    )
    return _attach_equalized_signal(result, signal_adapter.signal, state)


def block_cma(
    samples: ArrayType | Signal,
    *,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float | None = None,
    block_size: int = 256,
    constellation: Any = None,
    initial_taps: ArrayType | None = None,
    pilots: EqualizerPilots | None = None,
    pad_mode: str = "zeros",
    state: EqualizerState | None = None,
    cuda_graph: bool = True,
) -> EqualizerResult:
    """Equalize blindly with the frequency-domain constant-modulus algorithm.

    The phase-blind sibling of :func:`block_lms`: the same overlap-save
    frequency-domain adaptive filter (FDAF), driven by the Godard
    constant-modulus error instead of training symbols or decisions.  For
    blind acquisition at the high-throughput operating point (slow or static
    channels, GPU input); :func:`cma` adapts every symbol instead, for the
    fastest channel dynamics, and is usually faster on the CPU.

    Like :func:`cma`, it recovers the channel up to a phase ambiguity: run a
    carrier-phase recovery stage afterwards (there is no ``cpr``, see the
    Notes of :func:`cma`).  ``pilots`` switch the pilot positions to the LMS
    residual error, which resolves the ambiguity there while the data
    positions stay blind.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` supplies
        ``sps`` and ``constellation``.
    num_taps : int, default 21
        Taps per FIR filter, in samples.
    sps : int, optional
        Samples per symbol, an integer.  Taken from the Signal; required for
        array input.  A value that disagrees with the Signal raises.
    step_size : float, optional
        Step size on the scale of :func:`cma`.  The weights are frozen
        across each block, so the stability ceiling is about ``block_size``
        times lower than per symbol; reduce it if the run raises divergence.
        Defaults to ``min(1e-3, 0.5 / block_size) / sps``, as for
        :func:`block_rde`.
    block_size : int, default 256
        Output symbols per weight update.
    constellation : Constellation, optional
        Sets the Godard radius, as for :func:`cma`.  Defaults to the
        Signal's; without one the target is the unit circle.
    initial_taps : array_like, optional
        Initial tap weights, as for :func:`cma`; only for a cold start.
    pilots : EqualizerPilots, optional
        Known pilots, as for :func:`cma`.
    pad_mode : {'zeros', 'edge', 'reflect'}, default 'zeros'
        Left padding of a cold start; see :func:`lms`.
    state : EqualizerState, optional
        Continue from ``result.state`` of a previous call with the same
        configuration; see :func:`lms`.
    cuda_graph : bool, default True
        On the GPU, capture the per-block body once into a CUDA graph and
        replay it (one launch per block, a large gain at small
        ``block_size``).  Ignored on the CPU and for the pilot-aided path.

    Returns
    -------
    EqualizerResult
        ``y_hat``, ``weights``, ``error`` and ``state`` on the input's
        device, and for Signal input ``signal``, the 1-SPS output Signal.
    """
    return _block_blind(
        "cma",
        samples,
        num_taps=num_taps,
        sps=sps,
        step_size=step_size,
        block_size=block_size,
        constellation=constellation,
        initial_taps=initial_taps,
        pilots=pilots,
        pad_mode=pad_mode,
        state=state,
        cuda_graph=cuda_graph,
    )


def block_rde(
    samples: ArrayType | Signal,
    *,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float | None = None,
    block_size: int = 256,
    constellation: Any = None,
    initial_taps: ArrayType | None = None,
    pilots: EqualizerPilots | None = None,
    pad_mode: str = "zeros",
    state: EqualizerState | None = None,
    cuda_graph: bool = True,
) -> EqualizerResult:
    """Equalize blindly with the frequency-domain radius-directed algorithm.

    Related to :func:`block_cma` as :func:`rde` is to :func:`cma`: each
    symbol is driven towards its nearest constellation ring instead of one
    Godard circle, which converges blindly on multi-ring QAM (16-, 64-QAM)
    where CMA's single radius stalls.  The same overlap-save engine and
    operating point as :func:`block_cma`.

    Blind up to a phase ambiguity (run carrier-phase recovery afterwards);
    ``pilots`` resolve it at the pilot positions.  From a
    cold start RDE converges several times slower than CMA: pass ``state=``
    from a :func:`block_cma` call over the first part of the record to start
    from CMA-converged taps.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` supplies
        ``sps`` and ``constellation``.
    num_taps : int, default 21
        Taps per FIR filter, in samples.
    sps : int, optional
        Samples per symbol, an integer.  Taken from the Signal; required for
        array input.  A value that disagrees with the Signal raises.
    step_size : float, optional
        Step size on the scale of :func:`rde`, with the about ``block_size``
        times lower stability ceiling of frozen-block adaptation.  Defaults
        to ``min(1e-3, 0.5 / block_size) / sps``: the per-symbol step, capped
        at half the ceiling (``step_size * block_size * sps`` below about 1,
        on input normalized to unit mean power) for large blocks.
    block_size : int, default 256
        Output symbols per weight update.
    constellation : Constellation, optional
        Sets the ring radii, as for :func:`rde`.  Defaults to the Signal's;
        without one the single ring is the unit circle.
    initial_taps : array_like, optional
        Initial tap weights, as for :func:`rde`; only for a cold start.
    pilots : EqualizerPilots, optional
        Known pilots, as for :func:`rde`.
    pad_mode : {'zeros', 'edge', 'reflect'}, default 'zeros'
        Left padding of a cold start; see :func:`lms`.
    state : EqualizerState, optional
        Continue from ``result.state`` of a previous ``block_rde`` or
        ``block_cma`` call with the same configuration; see :func:`lms`.
    cuda_graph : bool, default True
        On the GPU, capture the per-block body once into a CUDA graph and
        replay it.  Ignored on the CPU and for the pilot-aided path.

    Returns
    -------
    EqualizerResult
        ``y_hat``, ``weights``, ``error`` and ``state`` on the input's
        device, and for Signal input ``signal``, the 1-SPS output Signal.
    """
    return _block_blind(
        "rde",
        samples,
        num_taps=num_taps,
        sps=sps,
        step_size=step_size,
        block_size=block_size,
        constellation=constellation,
        initial_taps=initial_taps,
        pilots=pilots,
        pad_mode=pad_mode,
        state=state,
        cuda_graph=cuda_graph,
    )


# -----------------------------------------------------------------------------
# PILOT REFERENCE BUILDER (array-only)
