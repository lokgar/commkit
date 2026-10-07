"""Frequency-domain (FDAF) block equalizer engine shared by block_lms/cma/rde.

One run is: :func:`_prepare_block` (normalization, padding, weights on the
input's device), a ``run_block(B, b_start)`` callback per block driven by
:func:`_block_loop` (eager, or replayed from a captured CUDA graph), and the
caller's result assembly.  :func:`_fdaf_forward` and
:func:`_fdaf_gradient_update` are the filter and the LMS-type update every
block uses.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._array import as_2d
from ...backend import ArrayType, dispatch, to_device
from ...logger import logger
from .._common import (
    _build_padded_samples,
    _init_butterfly_weights_numpy,
    _normalize_inputs,
    _validate_sps,
    _validate_w_init,
)
from ..result import EqualizerState, _check_state


@dataclass
class _Block:
    """Device buffers and geometry of one block-equalizer run."""

    xp: Any
    was_1d: bool
    C: int
    n_sym: int
    sps: int
    num_taps: int
    block_size: int
    fftsize: int
    x_padded: ArrayType  # (C, N_pad) complex64, normalized and padded
    lead: int  # samples of x_padded before symbol 0's nominal position
    real_end: int  # x_padded[:, :real_end] is data (or left padding)
    resume: int  # block boundary where the continuation state is taken
    training: ArrayType | None  # (C, K) normalized training on the device
    eq_norm: Any
    h: ArrayType  # (C, C, T) complex64 weights, updated in place
    x_win: ArrayType  # (C, F) input window of the current block

    def fill_window(self, b_start: int, B: int) -> None:
        """Load the ``B·sps + T - 1`` input samples of block ``b_start`` into
        ``x_win``, zeros after them.

        Only these samples reach the block's outputs and gradient; loading
        later ones too would change the FFT's rounding with data the block
        does not use (and break exact continuation at a block boundary).
        """
        x_start = b_start * self.sps
        self.x_win.fill(0)
        available = min(
            B * self.sps + self.num_taps - 1, self.x_padded.shape[1] - x_start
        )
        if available > 0:
            self.x_win[:, :available] = self.x_padded[:, x_start : x_start + available]


def _prepare_block(
    samples: ArrayType,
    *,
    equalizer: str,
    sps: int,
    num_taps: int,
    block_size: int,
    initial_taps: ArrayType | None,
    state: EqualizerState | None,
    pad_mode: str,
    name: str,
    cpu_hint: str,
    cpr: Any = None,
    training_symbols: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
) -> _Block:
    """Normalize, pad and allocate on the input's device.

    A cold start pads around the center tap; a ``state`` prepends its
    pending input and reuses its weights and normalization.  ``pilot_mask``
    with a non-zero ``pilot_gain_db`` de-boosts the pilot samples (on a
    copy) before the normalization.
    """
    num_taps = int(num_taps)
    block_size = int(block_size)
    _validate_sps(sps, num_taps)
    sps = int(sps)

    samples, xp, _ = dispatch(samples)
    if xp is np:
        logger.warning(
            "%s is running on CPU (NumPy). For CPU workloads %s is typically "
            "faster. Move samples to GPU (CuPy) to benefit from block-FFT "
            "acceleration.",
            name,
            cpu_hint,
        )
    samples, was_1d = as_2d(samples, name="samples")
    C, N = samples.shape
    _check_state(
        state,
        equalizer=equalizer,
        num_taps=num_taps,
        sps=sps,
        num_ch=C,
        block_size=block_size,
        cpr=cpr,
        initial_taps=initial_taps,
    )
    if state is None:
        n_sym = N // sps
        offset = 0
    else:
        offset = state.pending.shape[-1] - state.lead
        n_sym = (offset + N) // sps

    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)
        # Known symbols are small: they follow the samples' device.
        training_symbols = to_device(training_symbols, "cpu" if xp is np else "gpu")
        if training_symbols.ndim == 1:
            training_symbols = training_symbols[np.newaxis, :]

    if pilot_mask is not None and pilot_gain_db != 0.0:
        amp = xp.float32(10.0 ** (pilot_gain_db / 20.0))
        smask_np = np.repeat(np.asarray(pilot_mask).astype(bool), sps)
        if state is not None:  # the pending samples were de-boosted already
            new = np.zeros(offset + N, dtype=bool)
            new[: smask_np.size] = smask_np[: offset + N]
            smask_np = new[offset:]
        smask = xp.asarray(smask_np)
        samples = samples.copy()
        samples[..., smask] /= amp

    samples, training_symbols, eq_norm = _normalize_inputs(
        samples,
        training_symbols,
        sps,
        input_norm_factor=None if state is None else state.input_norm_factor,
    )

    # Overlap-save FFT size: the next power of 2 >= block_size*sps + T - 1.
    ols_min = block_size * sps + num_taps - 1
    fftsize = 1 << (ols_min - 1).bit_length()

    if state is None:
        c_tap = num_taps // 2
        pad_total = max(0, n_sym * sps - N + num_taps - 1)
        lead = min(c_tap, pad_total)
        pad_right = pad_total - lead
        real_end = lead + N
        if xp is np or pad_mode != "zeros":
            samples_cpu = to_device(samples, "cpu").astype(np.complex64)
            x_padded = xp.asarray(
                _build_padded_samples(
                    samples_cpu, lead, pad_right, None, pad_mode, None, sps
                )
            )
        else:
            # Zero-pad on the device: no host round trip for GPU input.
            f32 = (
                samples
                if samples.dtype == xp.complex64
                else samples.astype(xp.complex64)
            )
            left = xp.zeros((C, lead), dtype=xp.complex64)
            right = (
                xp.zeros((C, pad_right), dtype=xp.complex64)
                if pad_right > 0
                else xp.empty((C, 0), dtype=xp.complex64)
            )
            x_padded = xp.concatenate([left, f32, right], axis=1)
    else:
        lead = state.lead
        real_end = state.pending.shape[-1] + N
        pad_right = max(0, n_sym * sps + num_taps - 1 - real_end)
        x_padded = xp.concatenate(
            [
                xp.asarray(state.pending),
                samples.astype(xp.complex64, copy=False),
                xp.zeros((C, pad_right), dtype=xp.complex64),
            ],
            axis=1,
        )

    # Last block boundary before the first symbol whose window
    # x[k*sps : k*sps + T] reaches past the data.
    n_done = min(n_sym, max(0, (real_end - num_taps) // sps + 1))
    resume = (n_done // block_size) * block_size

    if state is not None:
        h = xp.asarray(state.weights.copy())
    elif initial_taps is not None:
        w_arr = np.ascontiguousarray(to_device(initial_taps, "cpu"), dtype=np.complex64)
        h = xp.asarray(_validate_w_init(w_arr, C, num_taps).copy())
    else:
        h = xp.asarray(_init_butterfly_weights_numpy(C, num_taps))  # (C, C, T)

    return _Block(
        xp=xp,
        was_1d=was_1d,
        C=C,
        n_sym=n_sym,
        sps=sps,
        num_taps=num_taps,
        block_size=block_size,
        fftsize=fftsize,
        x_padded=x_padded,
        lead=lead,
        real_end=real_end,
        resume=resume,
        training=training_symbols,
        eq_norm=eq_norm,
        h=h,
        x_win=xp.zeros((C, fftsize), dtype=xp.complex64),
    )


def _block_state(
    run: _Block, *, equalizer: str, cpr: Any, weights: np.ndarray, carrier: Any
) -> EqualizerState:
    """The continuation state at ``run.resume`` (host copies)."""
    return EqualizerState(
        equalizer=equalizer,
        num_taps=run.num_taps,
        sps=run.sps,
        block_size=run.block_size,
        cpr=cpr,
        weights=weights,
        input_norm_factor=run.eq_norm,
        pending=to_device(
            run.x_padded[:, run.resume * run.sps : run.real_end], "cpu"
        ).copy(),
        lead=run.lead,
        overlap=run.n_sym - run.resume,
        carrier=carrier,
    )


def _fdaf_forward(
    h: ArrayType, x_win: ArrayType, fftsize: int, sps: int, B: int, xp: Any
) -> tuple[ArrayType, ArrayType]:
    """Butterfly filter of one block in the frequency domain.

    ``y[c] = Σ_c' conj(h[c, c']) ⋆ x[c']`` (cross-correlation), accumulated in
    complex128 and decimated to the ``B`` symbols of the block.  Returns the
    block output ``(C, B)`` and the input spectrum ``X_fd`` reused by the
    update.
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


def _fdaf_gradient_update(
    h: ArrayType,
    X_fd: ArrayType,
    e_block: ArrayType,
    e_scatter: ArrayType,
    sps: int,
    B: int,
    num_taps: int,
    mu: float,
    xp: Any,
) -> None:
    """In-place block update ``h += mu · Σ_n conj(e[n]) x_n`` via the FFT.

    The errors are scattered onto the sample grid (every ``sps``-th sample),
    correlated with the block input and truncated to the ``num_taps`` taps.
    """
    e_scatter.fill(0)
    e_scatter[:, : B * sps : sps] = e_block
    E_fd = xp.fft.fft(e_scatter, axis=-1)  # (C, F)
    dH_fd = xp.conj(E_fd)[:, None, :] * X_fd[None, :, :]  # (C, C, F)
    dh = xp.fft.ifft(dH_fd, axis=-1)[:, :, :num_taps]  # (C, C, T)
    h += xp.float32(mu) * dh


def _block_loop(
    run: _Block,
    *,
    run_block: Callable[[int, int], None],
    store: Callable[[int, int, int], None],
    capturable: Callable[[int, int], bool],
    use_graph: bool,
    name: str,
    at_resume: Callable[[], None] | None = None,
) -> None:
    """Drive ``run_block(B, b_start)`` over all blocks of ``run``.

    With ``use_graph`` (CuPy only), the first full ``capturable`` block runs
    eagerly to prime the memory pool, the second is captured into a CUDA
    graph and every later capturable block replays it: one launch per block
    instead of dozens.  A failed capture falls back to the eager loop.
    ``at_resume()`` runs once, before block ``run.resume`` (or after the last
    block), to snapshot the continuation state.
    """
    graph_stream = None
    if use_graph:
        try:
            import cupy as cp

            graph_stream = cp.cuda.Stream(non_blocking=True)
        except Exception:
            use_graph = False

    stream_ctx: Any
    if use_graph:
        assert graph_stream is not None
        # Order the setup work on the default stream before the loop stream.
        setup_done = cp.cuda.Event()
        setup_done.record()
        graph_stream.wait_event(setup_done)
        stream_ctx = graph_stream
    else:
        stream_ctx = contextlib.nullcontext()

    graph = None  # captured CUDA graph, built lazily on the 2nd full block
    warmed = False  # True once one full block has primed the memory pool
    n_blocks = (run.n_sym + run.block_size - 1) // run.block_size
    with stream_ctx:
        for b in range(n_blocks):
            b_start = b * run.block_size
            b_end = min(b_start + run.block_size, run.n_sym)
            B = b_end - b_start  # symbols this block (may be short for the last)
            if at_resume is not None and b_start == run.resume:
                at_resume()
            graph_ok = use_graph and capturable(B, b_start)
            run.fill_window(b_start, B)
            if not graph_ok:
                run_block(B, b_start)  # eager
            elif graph is not None:
                graph.launch()  # replay (current stream == graph_stream)
            elif not warmed:
                run_block(B, b_start)  # eager warmup - primes the memory pool
                warmed = True
            else:
                assert graph_stream is not None
                try:
                    graph_stream.begin_capture()
                    run_block(B, b_start)
                    graph = graph_stream.end_capture()
                    graph.launch()
                except Exception as exc:  # pragma: no cover - hw/version dependent
                    with contextlib.suppress(Exception):
                        graph_stream.end_capture()
                    graph = None
                    use_graph = False
                    logger.warning(
                        "%s CUDA-graph capture failed (%s); continuing with the "
                        "eager block loop.",
                        name,
                        exc,
                    )
                    run_block(B, b_start)  # ensure this block runs once
            store(b_start, b_end, B)
        if at_resume is not None and run.resume >= run.n_sym:
            at_resume()
    if graph_stream is not None:
        graph_stream.synchronize()
