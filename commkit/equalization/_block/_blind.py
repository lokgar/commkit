"""Blind frequency-domain block equalizer engine (FDAF) backing blind.py's
block_cma / block_rde.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from ...backend import dispatch, to_device
from ...helpers import as_2d
from ...logger import logger
from .._common import (
    _build_padded_samples,
    _init_butterfly_weights_numpy,
    _normalize_inputs,
    _validate_sps,
    _validate_w_init,
)
from ..result import EqualizerResult, _log_equalizer_exit

# -----------------------------------------------------------------------------
# BLIND FREQUENCY-DOMAIN EQUALIZERS  (block_cma / block_rde)
# -----------------------------------------------------------------------------
#
# block_cma / block_rde are the blind, phase-directed siblings of block_lms:
# the same overlap-save frequency-domain adaptive filter (FDAF), but with the
# Godard / ring-radius error instead of the trained/DD slicer.  They share the
# two FDAF primitives below (forward butterfly + frequency-domain gradient);
# block_lms keeps its own CPR/cycle-slip/CUDA-graph-optimised body.
#
# All three use the same additive update ``h += mu * conj(E_fd) * X_fd`` with
# the gradient correlation in the frequency domain, so the blind error is cast
# into the trained convention's sign: CMA ``E = y*(R2 - |y|^2)`` and RDE
# ``E = y*(R_d^2 - |y|^2)`` (the negated Godard/ring gradient), and pilot
# positions use the LMS residual ``E = pilot_ref - y`` directly.


def _fdaf_forward(h, x_win, fftsize, sps, B, xp):
    """One overlap-save FDAF forward block.

    Returns ``(y_block (C, B), X_fd (C, F))`` - the decimated output symbols and
    the input spectrum (reused by the gradient).  The butterfly contraction is a
    complex128-accumulated broadcast-reduce (CLAUDE.md filter-dot-product rule),
    matching ``block_lms``'s capture-safe form.
    """
    X_fd = xp.fft.fft(x_win, axis=-1)  # (C, F)
    H_fd = xp.fft.fft(h, n=fftsize, axis=-1)  # (C, C, F)
    Y_fd = (
        (xp.conj(H_fd).astype(xp.complex128) * X_fd.astype(xp.complex128)[None])
        .sum(axis=1)
        .astype(xp.complex64)
    )  # (C, F)
    y_time = xp.fft.ifft(Y_fd, axis=-1)
    y_block = y_time[:, : B * sps : sps].astype(xp.complex64)  # (C, B)
    return y_block, X_fd


def _fdaf_gradient_update(h, X_fd, e_block, e_scatter, sps, B, num_taps, mu, xp):
    """Scatter ``e_block`` to sample positions, form the frequency-domain
    gradient ``dH[i,j] = conj(E_fd[i]) * X_fd[j]``, and apply the in-place
    update ``h += mu * IFFT(dH)[..., :num_taps]``."""
    e_scatter.fill(0)
    e_scatter[:, : B * sps : sps] = e_block
    E_fd = xp.fft.fft(e_scatter, axis=-1)  # (C, F)
    dH_fd = xp.conj(E_fd)[:, None, :] * X_fd[None, :, :]  # (C, C, F)
    dh = xp.fft.ifft(dH_fd, axis=-1)[:, :, :num_taps]  # (C, C, T)
    h += xp.float32(mu) * dh


def _block_fdaf_blind(
    kind,
    samples,
    *,
    num_taps,
    sps,
    step_size,
    block_size,
    r2,
    radii_np,
    w_init,
    input_norm_factor,
    samples_prefix,
    pad_mode,
    pilot_ref,
    pilot_mask,
    pilot_gain_db,
    c_ps,
    cuda_graph,
    debug_plot,
    plot_smoothing,
    name,
):
    """Shared overlap-save FDAF engine for ``block_cma``/``block_rde``.

    Blind, phase-directed adaptation with no CPR - the per-block error is the
    Godard (``kind='cma'``) or nearest-ring (``kind='rde'``) gradient, with
    pilot positions overridden by the LMS residual.  ``r2`` is the Godard radius
    (CMA) and ``radii_np`` the unique ring radii (RDE).
    """
    num_taps = int(num_taps)
    block_size = int(block_size)
    _validate_sps(sps, num_taps)
    sps = int(sps)

    samples, xp, _ = dispatch(samples)
    if xp is np:
        logger.warning(
            "%s is running on CPU (NumPy). For CPU workloads "
            "%s(..., backend='numba') is typically faster. Move samples to "
            "GPU (CuPy) to benefit from block-FFT acceleration.",
            name,
            kind,
        )

    samples, was_1d = as_2d(samples, name="samples")
    C = samples.shape[0]
    N = samples.shape[1]
    n_sym = N // sps

    use_pilots = pilot_ref is not None and pilot_mask is not None
    # Deboost pilot positions before normalisation so boosted pilots don't
    # inflate the RMS and bias the Godard target (matches cma/rde).
    if use_pilots and pilot_gain_db != 0.0:
        amp = xp.float32(10.0 ** (pilot_gain_db / 20.0))
        smask = xp.asarray(np.repeat(np.asarray(pilot_mask).astype(bool), sps))
        samples = samples.copy()
        samples[..., smask] /= amp
    samples, _, eq_norm = _normalize_inputs(
        samples, None, sps, input_norm_factor=input_norm_factor
    )

    # -- OLS block size + padding (matches block_lms) -----------------------
    _ols_min = block_size * sps + num_taps - 1
    fftsize = 1 << (_ols_min - 1).bit_length()
    logger.info(
        "%s: C=%s, num_taps=%s, sps=%s, block_size=%s, fftsize=%s, mu=%s, "
        "n_sym=%s, pilot_aided=%s",
        name,
        C,
        num_taps,
        sps,
        block_size,
        fftsize,
        step_size,
        n_sym,
        use_pilots,
    )
    c_tap = num_taps // 2
    pad_total = max(0, n_sym * sps - N + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left
    if samples_prefix is not None or xp is np or pad_mode != "zeros":
        _cpu = to_device(samples, "cpu").astype(np.complex64)
        x_padded = xp.asarray(
            _build_padded_samples(
                _cpu, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
            )
        )
    else:
        _f32 = (
            samples if samples.dtype == xp.complex64 else samples.astype(xp.complex64)
        )
        _l = xp.zeros((C, pad_left), dtype=xp.complex64)
        _r = (
            xp.zeros((C, pad_right), dtype=xp.complex64)
            if pad_right > 0
            else xp.empty((C, 0), dtype=xp.complex64)
        )
        x_padded = xp.concatenate([_l, _f32, _r], axis=1)
    N_padded = x_padded.shape[1]

    # -- Weight initialisation ----------------------------------------------
    if w_init is not None:
        w_arr = _validate_w_init(
            np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64),
            C,
            num_taps,
        )
        h = xp.asarray(w_arr.copy())
    else:
        h = xp.asarray(_init_butterfly_weights_numpy(C, num_taps))  # (C, C, T)

    # -- Pilot / radii device arrays ----------------------------------------
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

    # -- Scratch + output buffers -------------------------------------------
    # Pre-allocated and reused every block: avoids per-block heap pressure and
    # (on GPU) gives the CUDA-graph capture stable read/write pointers.
    x_win = xp.zeros((C, fftsize), dtype=xp.complex64)
    e_scatter = xp.zeros((C, fftsize), dtype=xp.complex64)
    y_all = xp.empty((C, n_sym), dtype=xp.complex64)
    e_all = xp.empty((C, n_sym), dtype=xp.complex64)
    # Fixed-width (block_size) output workspaces.  The capturable body always
    # writes its per-symbol outputs here; the eager driver copies the valid
    # [:, :B] slice into the full-length arrays at the right offset.  Routing
    # through fixed buffers (rather than writing y_all[:, b_start:b_end]
    # directly) is what lets the body be captured once and replayed: only the
    # input fill and output copy vary per block.
    y_ws = xp.empty((C, block_size), dtype=xp.complex64)
    e_ws = xp.empty((C, block_size), dtype=xp.complex64)
    n_blocks = (n_sym + block_size - 1) // block_size

    def _run_block(B, b_start):
        """Compute one block: forward FDAF, blind error, gradient + in-place
        weight update.  Reads the window from ``x_win`` (filled by the caller),
        updates ``h`` in place, and writes outputs into ``y_ws``/``e_ws`` (cols
        ``[:, :B]``).  Issues no host sync, so on GPU it is capturable.  The
        ``b_start``-varying pilot slice is only touched on the (non-captured)
        pilot-aided path."""
        nonlocal h
        y_block, X_fd = _fdaf_forward(h, x_win, fftsize, sps, B, xp)

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
        _fdaf_gradient_update(h, X_fd, e, e_scatter, sps, B, num_taps, step_size, xp)

    # CUDA-graph invariant: _fill_x_win / _store_outputs run eagerly *between*
    # replays and MUST NOT allocate from the device memory pool (a fresh
    # allocation could hand out the blocks the captured graph's freed
    # intermediates still reference).  Both touch only pre-allocated buffers.
    def _fill_x_win(b_start):
        """Eager (per-block, varying offset) load of the input window."""
        x_start = b_start * sps
        x_win.fill(0)
        avail = min(fftsize, N_padded - x_start)
        if avail > 0:
            x_win[:, :avail] = x_padded[:, x_start : x_start + avail]

    def _store_outputs(b_start, b_end, B):
        """Eager copy of the fixed workspaces into the result arrays."""
        y_all[:, b_start:b_end] = y_ws[:, :B]
        e_all[:, b_start:b_end] = e_ws[:, :B]

    # -- CUDA-graph eligibility ----------------------------------------------
    # The block body is host-sync-free, so on GPU it can be captured once and
    # replayed per block, collapsing the per-block kernel launches into one.
    # Only full blocks (B == block_size) share a fixed shape and control flow,
    # so only those are captured; the final partial block runs eagerly.  The
    # pilot-aided path reads b_start-varying pilot slices inside the body and is
    # therefore not capturable - it always runs the eager loop.
    n_full = n_sym // block_size
    _use_graph = (
        cuda_graph
        and xp is not np
        and not use_pilots
        and n_full >= 2  # need ≥1 warmup block + ≥1 captured block to pay off
    )
    try:
        import cupy as _cp_graph

        _graph_stream = _cp_graph.cuda.Stream(non_blocking=True) if _use_graph else None
    except Exception:
        _use_graph = False
        _graph_stream = None

    # Run the whole loop on one stream so the shared state (h) stays ordered
    # across the eager<->replay boundary.  The non-blocking graph stream does not
    # implicitly serialize with the default stream that produced x_padded/h/the
    # scratch buffers, so join it to that setup work before the first block to
    # avoid capturing still-in-flight garbage.
    _loop_stream_ctx: Any
    if _use_graph:
        assert _graph_stream is not None
        _setup_done = _cp_graph.cuda.Event()
        _setup_done.record()
        _graph_stream.wait_event(_setup_done)
        _loop_stream_ctx = _graph_stream
    else:
        _loop_stream_ctx = contextlib.nullcontext()

    # -- Block loop ----------------------------------------------------------
    _graph = None  # captured CUDA graph, built lazily on the 2nd full block
    _graph_warmed = False  # True once one full block has primed the mem pool
    with _loop_stream_ctx:
        for b in range(n_blocks):
            b_start = b * block_size
            b_end = min(b_start + block_size, n_sym)
            B = b_end - b_start
            _capturable = _use_graph and block_size == B

            _fill_x_win(b_start)

            if not _capturable:
                _run_block(B, b_start)  # eager (partial block, or graph off)
            elif _graph is not None:
                _graph.launch()  # replay (current stream == _graph_stream)
            elif not _graph_warmed:
                _run_block(B, b_start)  # eager warmup - primes the memory pool
                _graph_warmed = True
            else:
                # Capture the body once; pool allocations reuse the blocks the
                # warmup freed, so no cudaMalloc occurs during capture.  On any
                # failure, fall back to the eager loop for the rest of the run.
                assert _graph_stream is not None
                try:
                    _graph_stream.begin_capture()
                    _run_block(B, b_start)
                    _graph = _graph_stream.end_capture()
                    _graph.launch()
                except Exception as exc:  # pragma: no cover - hw/version dependent
                    with contextlib.suppress(Exception):
                        _graph_stream.end_capture()
                    _graph = None
                    _use_graph = False
                    logger.warning(
                        "%s CUDA-graph capture failed (%s); "
                        "continuing with the eager block loop.",
                        name,
                        exc,
                    )
                    _run_block(B, b_start)  # ensure this block runs once

            _store_outputs(b_start, b_end, B)

    if _graph_stream is not None:
        _graph_stream.synchronize()

    # Single D->H sync to surface divergence after the whole loop.
    if not bool(xp.isfinite(h).all()):
        raise RuntimeError(
            f"{name} diverged (step_size={step_size}, block_size={block_size}). "
            f"step_size is on the same scale as {kind}(); because the weights are "
            f"frozen across the block the stability ceiling is ~{block_size}x lower. "
            f"Reduce step_size (e.g. {step_size / 2:.2e}, then keep halving)."
        )

    if was_1d:
        y_out, e_out, W_out = y_all[0], e_all[0], h[0, 0]
    else:
        y_out, e_out, W_out = y_all, e_all, h
    result = EqualizerResult(
        y_hat=y_out,
        weights=W_out,
        error=e_out,
        weights_history=None,
        num_train_symbols=0,
        input_norm_factor=eq_norm,
    )
    return _log_equalizer_exit(
        result,
        name=name,
        debug_plot=debug_plot,
        check_convergence=True,
        plot_smoothing=plot_smoothing,
    )
