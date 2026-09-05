"""Pilot-tone-based polarization demultiplexing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast, overload

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import prepare_signal_input
from ..core.signal import Signal
from ..filtering import fir_filter, fir_taps
from ..logger import logger

# -----------------------------------------------------------------------------
# TIME-VARYING MATRIX APPLY (Signal-aware)
# -----------------------------------------------------------------------------
# Generic interpolate-and-apply core shared with the dynamic demux below.


@overload
def apply_interpolated_matrix(
    samples: ArrayType, matrix_grid: ArrayType, grid_positions: ArrayType
) -> ArrayType: ...


@overload
def apply_interpolated_matrix(
    samples: Signal, matrix_grid: ArrayType, grid_positions: ArrayType
) -> Signal: ...


def apply_interpolated_matrix(
    samples: ArrayType | Signal,
    matrix_grid: ArrayType,
    grid_positions: ArrayType,
) -> ArrayType | Signal:
    r"""Apply a time-varying matrix ``M(n)`` to ``samples``, interpolated from a grid.

    Output sample ``n`` is ``M(n) · samples[:, n]``, where ``M(n)`` is the exact
    linear blend ``M[g] + (M[g+1]-M[g])·t`` inside grid cell ``g`` (``t`` ramps
    ``0->1`` across the cell).  This is the apply half of the dynamic pilot demux,
    but it is a generic operator: given any per-grid-point matrix stack (an inverse
    ``W(n)``, a polar unitary ``Qᴴ(n)``, a butterfly seed, ...) it interpolates and
    applies it as a fixed forward pass.

    Within each uniform cell the apply collapses to two batched ``(K, C)`` GEMMs -
    ``M[g]·X`` and ``dM·(X·t)`` - instead of materialising a per-sample
    ``(L, K, C)`` matrix and an einsum mat-vec, so the cost is a handful of batched
    kernels rather than an ``O(N)`` Python/launch-bound loop.  The ``O(N)`` data
    path runs in ``complex64`` (a well-conditioned mix with no long accumulation),
    so a ``matrix_grid`` built by a double-precision inverse/SVD keeps its
    precision while the bulk traffic moves at half the bandwidth.

    Parameters
    ----------
    samples : (C, N) array, or Signal
        Input channels.  A :class:`Signal` returns a new :class:`Signal`.
    matrix_grid : (G, K, C) array
        Per-grid-point matrices mapping the ``C`` inputs to ``K`` outputs.
    grid_positions : (G,) array
        Sample indices at which ``matrix_grid`` was evaluated.  The interior must
        be uniformly spaced (the last point may be pinned to ``N-1``).

    Returns
    -------
    (K, N) array
        ``M(n) · samples[:, n]``, same dtype as ``samples``.
    """
    context = prepare_signal_input(samples, function_name="apply_interpolated_matrix()")
    samples = context.array

    samples, xp, _ = dispatch(samples)
    C, N = samples.shape
    M = xp.asarray(matrix_grid, dtype=xp.complex64)  # (G, K, C)
    gp = xp.asarray(grid_positions, dtype=xp.float64)  # (G,)
    G, K = int(M.shape[0]), int(M.shape[1])
    xc = samples.astype(xp.complex64, copy=False)
    step = int(round(float(gp[1] - gp[0]))) if G > 1 else N
    out = xp.empty((K, N), dtype=xp.complex64)

    nblk = (N - 1) // step if step > 0 else 0  # full uniform cells over [0, nblk·step)
    bulk = nblk * step
    if nblk > 0:
        Xb = xc[:, :bulk].reshape(C, nblk, step).transpose(1, 0, 2)  # (nblk, C, step)
        M0 = M[:nblk]  # (nblk, K, C)
        dM = M[1 : nblk + 1] - M0
        t = (xp.arange(step, dtype=xp.float32) / step).astype(xp.complex64)
        # Diagonal scaling on the sample axis commutes through the left matmul
        # (dM @ (Xb·diag(t)) == (dM @ Xb)·diag(t)), so stacking [M0; dM] turns the
        # two batched GEMMs into one and moves the ramp multiply from the
        # input-sized (nblk, C, step) to the output-sized (nblk, K, step) array.
        Y = xp.concatenate([M0, dM], axis=1) @ Xb  # (nblk, 2K, step)
        y = Y[:, K:, :]
        y *= t[None, None, :]
        y += Y[:, :K, :]
        out[:, :bulk] = y.transpose(1, 0, 2).reshape(K, bulk)
    if bulk < N:  # tail (< step; spans the pinned last cell) - per-sample blend
        nn = xp.arange(bulk, N, dtype=xp.float64)
        lo = xp.clip(xp.searchsorted(gp, nn, side="right") - 1, 0, G - 2)
        frac = ((nn - gp[lo]) / (gp[lo + 1] - gp[lo])).astype(xp.complex64)
        M_full = M[lo] + (M[lo + 1] - M[lo]) * frac[:, None, None]  # (L, K, C)
        out[:, bulk:] = xp.einsum("lkc,cl->kl", M_full, xc[:, bulk:])
    return context.return_value(out.astype(samples.dtype, copy=False))


# -----------------------------------------------------------------------------
# TONE-BASED POLARIZATION DEMULTIPLEXING (Signal-aware)
# -----------------------------------------------------------------------------

_EXTRACT_CHUNK = 1 << 20  # samples per block in the chunked tone-phasor GEMM


def _tone_phasor_matrix(xw: ArrayType, freqs, sampling_rate: float) -> ArrayType:
    r"""Tone-phasor matrix ``T[i, j] = (1/N) Σ_n xw[i, n]·exp(-j2π f_j n/fs)``.

    The whole-record accumulation is precision-sensitive (it feeds a matrix
    inverse), but running the O(N) GEMM in complex128 would put the entire data
    path on the slow FP64 units.  Instead the record is processed in
    ``_EXTRACT_CHUNK`` blocks: each block's ``(C, chunk) @ (chunk, K)`` product
    runs in the signal's working precision (complex64 for complex64 input) and
    the small ``(C, K)`` per-block partials are accumulated in complex128 - the
    round-off no longer grows with N beyond a block, at complex64 bandwidth.
    The phase ramp is always formed and wrapped in float64 (a float32 ramp
    loses the integer turn count over long records) and only the *wrapped*
    phase drops to float32 for the ``exp``.
    """
    xw, xp, _ = dispatch(xw)
    C, N = xw.shape
    fc = xp.asarray([float(f) for f in freqs], dtype=xp.float64).reshape(-1, 1)
    K = int(fc.shape[0])
    real_dtype = xp.float64 if xw.dtype == xp.complex128 else xp.float32
    two_pi = 2.0 * np.pi
    T = xp.zeros((C, K), dtype=xp.complex128)
    for start in range(0, N, _EXTRACT_CHUNK):
        stop = min(start + _EXTRACT_CHUNK, N)
        nn = xp.arange(start, stop, dtype=xp.float64)
        ph = fc * (two_pi / sampling_rate) * nn[None, :]  # (K, chunk) float64
        ph -= xp.round(ph / two_pi) * two_pi
        basis = xp.exp(-1j * ph.astype(real_dtype))  # (K, chunk) working dtype
        T += xw[:, start:stop] @ basis.T  # (C, K) partial, accumulated in c128
    return T / N


def _refine_tone_frequencies(
    xw: ArrayType,
    T: ArrayType,
    freqs,
    sampling_rate: float,
    search_band: float,
) -> list[float]:
    """Sub-bin refine each tone on the receive channel where it is strongest.

    Two stages, replacing the per-tone ``find_bias_tone`` calls (each of which
    ran its own full-record, power-of-two zero-padded FFT):

    1. **Coarse**: one batched FFT (working precision) of only the *unique*
       strongest channels + the device-side log-parabolic fit - good to a
       fraction of a bin, but ill-conditioned for a tone sitting exactly on a
       bin (its neighbours are sinc nulls).
    2. **Fine**: two-segment phase slope - the exact tone phasor of each
       half-record at the coarse frequency (chunked GEMMs); the phase advance
       between the segment centroids gives the residual offset
       ``δ = ∠(z_b·conj(z_a))·fs/(2π·Δc)``, unambiguous for ``|δ| < 1`` bin
       (which stage 1 guarantees) and accurate to well below either parabolic
       fit.  The demux mix-down is Hz-sensitive - a tone-frequency error
       leaves a slow residual rotation on the demuxed output - hence this
       stage.
    """
    from ..frequency import _refine_tones_from_spectrum

    xw, xp, _ = dispatch(xw)
    N = int(xw.shape[-1])
    K = len(list(freqs))
    best_ch = to_device(xp.argmax(xp.abs(T), axis=0), "cpu")  # (K,) host ints
    uniq, inv = np.unique(np.asarray(best_ch), return_inverse=True)
    xr = xw[xp.asarray(uniq)]  # (U, N) unique strongest channels only
    X = xp.fft.fft(xr, axis=-1)
    f_coarse = _refine_tones_from_spectrum(
        X, sampling_rate, freqs, search_band, rows=inv
    )  # (K,) float64 host

    # Two-segment phase slope.  z_b is computed with a local time origin, so
    # its known carrier phase 2π·f̂·N2/fs is undone first - evaluated in
    # *turns* on host float64 (the fractional part keeps ~1e-9 rad precision
    # for any realistic record length).
    N2 = N // 2
    Za = _tone_phasor_matrix(xr[:, :N2], f_coarse, sampling_rate)  # (U, K)
    Zb = _tone_phasor_matrix(xr[:, N2:], f_coarse, sampling_rate)  # (U, K)
    rows_dev = xp.asarray(inv)
    cols = xp.arange(K)
    turns = f_coarse * N2 / sampling_rate
    corr = xp.asarray(np.exp(-2j * np.pi * (turns - np.round(turns))))  # (K,)
    dphi = to_device(
        xp.angle(Zb[rows_dev, cols] * corr * xp.conj(Za[rows_dev, cols])), "cpu"
    )
    dc = N2 + 0.5 * (N % 2)  # exact centroid spacing of the two segments
    delta_f = dphi * sampling_rate / (2.0 * np.pi * dc)
    return [float(f) for f in f_coarse + delta_f]


def _jones_at_grid_points(
    xw: ArrayType,
    h: np.ndarray,
    freqs,
    grid_np: np.ndarray,
    sampling_rate: float,
) -> ArrayType:
    r"""Tone-tracked Jones estimate evaluated **only** at the grid points.

    The mix-down + centred-FIR tracker output at sample ``g`` factors as

        T[c, k, g] = e^{-j2π f_k g/fs} · Σ_q h̃_k[q] · r_c[g - lead + q],
        h̃_k[q]    = h[L-1-q] · e^{-j2π f_k (q - lead)/fs},   lead = ⌈(L-1)/2⌉,

    i.e. a bandpass filter at ``+f_k`` - so instead of materialising the
    (C, K, N) mixed array and FFT-filtering every sample only to keep one in
    ``grid_step``, this gathers the (chunk, L) sample windows around the grid
    points and hits all K modulated tap vectors with one GEMM per chunk.  No
    full-record temporaries are created.  Zero-padding at the record edges
    reproduces ``mode='same'`` semantics exactly, so the ``num_taps//2``
    edge-guard contract is unchanged.

    Parameters
    ----------
    xw : (C, N) array
        Received samples in working precision (complex64/complex128).
    h : (L,) np.ndarray
        Tracking low-pass FIR taps.
    freqs : sequence of float
        The K tracked tone frequencies in Hz.
    grid_np : (G,) np.ndarray of int
        Sample indices of the inversion grid (host).
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    (G, C, K) array
        Jones stack in complex128, ready for the batched Gram inverse.
    """
    xw, xp, _ = dispatch(xw)
    C, N = xw.shape
    L = int(len(h))
    G = int(len(grid_np))
    lead = (L - 1) - (L - 1) // 2
    real_dtype = xp.float64 if xw.dtype == xp.complex128 else xp.float32
    two_pi = 2.0 * np.pi
    f_dev = xp.asarray([float(f) for f in freqs], dtype=xp.float64)  # (K,)
    K = int(f_dev.shape[0])

    # Modulated (bandpass) taps h̃ (L, K) and per-grid-point de-rotation (G, K):
    # phases formed and wrapped in float64, exp in working precision.
    q = xp.arange(L, dtype=xp.float64)
    ph_m = (two_pi / sampling_rate) * (q[:, None] - lead) * f_dev[None, :]  # (L, K)
    ph_m -= xp.round(ph_m / two_pi) * two_pi
    h_rev = xp.asarray(np.ascontiguousarray(h[::-1]), dtype=real_dtype)  # h[L-1-q]
    Ht = h_rev[:, None] * xp.exp(-1j * ph_m.astype(real_dtype))  # (L, K)

    gp = xp.asarray(grid_np, dtype=xp.float64)
    ph_g = (two_pi / sampling_rate) * gp[:, None] * f_dev[None, :]  # (G, K)
    ph_g -= xp.round(ph_g / two_pi) * two_pi
    rot = xp.exp(-1j * ph_g.astype(real_dtype))  # (G, K)

    # Interior windows lie fully inside [0, N); the few edge windows get a
    # masked (zero-padded) gather.  Chunk the gather so the (chunk, L) window
    # and index arrays stay ~128 MB regardless of G·L.
    starts_np = grid_np.astype(np.int64) - lead  # (G,) host, sorted
    i0 = int(np.searchsorted(starts_np, 0, side="left"))
    i1 = int(np.searchsorted(starts_np, N - L, side="right"))
    idx_dtype = xp.int64 if (N + L) >= (1 << 31) else xp.int32
    itemsize = xw.dtype.itemsize
    chunk = max(1, int((128 << 20) // (L * (itemsize + 4))))
    off = xp.arange(L, dtype=idx_dtype)
    Tg_work = xp.empty((C, G, K), dtype=xw.dtype)

    def _fill(a: int, b: int, masked: bool) -> None:
        for s0 in range(a, b, chunk):
            s1 = min(s0 + chunk, b)
            st = xp.asarray(starts_np[s0:s1], dtype=idx_dtype)
            idx = st[:, None] + off[None, :]  # (chunk, L)
            if masked:
                valid = (idx >= 0) & (idx < N)
                idx = xp.clip(idx, 0, N - 1)
            for c in range(C):
                wnd = xw[c][idx]  # (chunk, L) gather
                if masked:
                    wnd *= valid
                Tg_work[c, s0:s1] = wnd @ Ht  # (chunk, K)

    _fill(0, i0, True)  # leading edge (zero-padded)
    _fill(i0, i1, False)  # interior - plain gather
    _fill(i1, G, True)  # trailing edge

    Tg_work *= rot[None, :, :]
    return xp.moveaxis(Tg_work, 0, 1).astype(xp.complex128)  # (G, C, K)


def demultiplex_polarization_tones_static(
    samples: ArrayType | Signal,
    sampling_rate: float | None = None,
    tone_frequencies: Sequence[float] | None = None,
    *,
    refine_tones: bool = True,
    search_band: float | None = None,
    normalize: bool = True,
    return_matrix: bool = False,
) -> ArrayType | Signal | tuple[ArrayType | Signal, ArrayType]:
    r"""
    One-shot polarization demux from distinct per-stream CW pilot tones.

    Undoes a **frequency-flat** polarization / spatial mixing by inverting the
    channel's Jones matrix, which is read directly off pilot tones placed at a
    *distinct* frequency on each transmitted stream (see ``add_pilot_tone`` with
    a per-channel ``frequency`` list).

    **Principle.**  With tone ``f_j`` carried *only* by transmitted stream ``j``
    and a frequency-flat mixing ``r = J s`` (e.g.
    ``apply_polarization_mixing``), the complex amplitude of tone ``f_j`` in
    received channel ``i`` is ``T[i, j] = J[i, j] · α_j`` (``α_j`` = the TX tone
    amplitude of stream ``j``).  Hence the measured tone-phasor matrix factors as
    ``T = J · diag(α)``, and ``W = pinv(T)`` unmixes::

        W r = diag(1/α) · J^{-1} J s = diag(1/α) · s,

    recovering each stream up to a trivial per-stream complex scale ``1/α_j``
    (removed by ``normalize`` and/or downstream CPR).  Because each tone uniquely
    labels its stream, output row ``j`` always corresponds to ``tone_frequencies[j]``
    - there is **no polarization-permutation ambiguity** (unlike blind CMA; cf.
    ``resolve_polarization_permutation``).

    **Speed.**  Tones added with ``add_pilot_tone`` are grid-quantized
    (buffer-periodic), so a single DFT bin is the exact, mutually-orthogonal,
    maximum-likelihood estimator of each tone phasor.  The whole operation is two
    small GEMMs (extract ``T`` via a ``(C,N)·(N,K)`` product, apply ``W`` via a
    ``(K,C)·(C,N)`` product) plus one ``KxK`` inverse - no iteration, no
    convergence, far below the cost of a full FFT.

    **Scope.**  Frequency-flat **and time-invariant** SOP only - the whole-record
    DFT bin estimates a *single* Jones matrix.  If the state of polarization
    drifts appreciably over the capture (long records and/or low baud rate, so
    the wall-clock duration exceeds the SOP coherence time), the averaged tone
    phasor is biased and attenuated and the one-shot inverse leaves residual,
    time-growing crosstalk; use ``demultiplex_polarization_tones_dynamic``
    instead.  For PMD / DGD use the returned unmixer (``return_matrix=True``) to
    seed a butterfly equalizer (``cma`` / ``block_lms``).

    Parameters
    ----------
    samples : array_like or Signal
        Received MIMO samples. Shape ``(C, N)`` - time on the last axis.  A
        :class:`Signal` returns a new demultiplexed :class:`Signal`.
    sampling_rate : float, optional
        Sampling rate f_s in Hz.  Required for array input; ignored for
        :class:`Signal` input, which always uses the signal's own
        ``sampling_rate``.
    tone_frequencies : sequence of float
        The ``K`` distinct per-stream tone frequencies in Hz (as added at the
        TX, in transmitted-stream order).  Require ``K <= C``.  Output row ``j``
        corresponds to ``tone_frequencies[j]``.
    refine_tones : bool, default True
        If ``True``, sub-bin-refine each tone centre with ``find_bias_tone``
        (on the receive channel where that tone is strongest) before extraction,
        absorbing a residual carrier frequency offset that has dragged the tone
        off its nominal bin.  If ``False``, extract exactly at
        ``tone_frequencies``.
    search_band : float, optional
        Half-width in Hz of the per-tone peak search when ``refine_tones=True``.
        Defaults to ``4 * f_s / N`` (a few FFT bins).  Widen it if the carrier
        offset can exceed that, but keep it inside the tone-to-data guard so the
        data band never wins the argmax.
    normalize : bool, default True
        If ``True``, rescale each demuxed row so its mean power equals the mean
        per-channel input power, removing the arbitrary ``1/α_j`` per-stream
        scale and preserving the library power invariant.
    return_matrix : bool, default False
        If ``True``, also return the ``(K, C)`` unmixing matrix ``W``.

    Returns
    -------
    demuxed : array_like
        Demultiplexed streams. Shape ``(K, N)`` - one row per recovered
        *stream* (``K`` = number of tones), **not** per receive channel: the
        ``C`` input channels are mapped down to the ``K`` transmitted streams.
        In the usual square dual-pol case ``K == C == 2`` the two coincide.
        Same complex dtype and backend as the input.  Row ``j`` is the stream
        that carried ``tone_frequencies[j]``.
    W : array_like, optional
        Returned only if ``return_matrix=True``: the ``(K, C)`` unmixing matrix
        (``complex128``), i.e. the estimated inverse Jones matrix up to per-stream
        scaling.  Suitable as a seed for a butterfly equalizer.

    Raises
    ------
    ValueError
        If ``samples`` is not 2-D, ``tone_frequencies`` is empty, ``K > C``, or
        any tone frequency lies outside ``(-f_s/2, f_s/2)``.

    See Also
    --------
    add_pilot_tone : Add the per-stream tones at the transmitter.
    demultiplex_polarization_tones_dynamic : Time-varying (drifting-SOP) demux.
    """
    context = prepare_signal_input(
        samples, function_name="demultiplex_polarization_tones_static()"
    )
    samples = context.array
    sampling_rate = context.required("sampling_rate", sampling_rate)
    if tone_frequencies is None:
        raise ValueError(
            "demultiplex_polarization_tones_static() requires tone_frequencies."
        )

    samples, xp, _ = dispatch(samples)
    if samples.ndim != 2:
        raise ValueError(
            "demultiplex_polarization_tones_static requires a 2-D (C, N) MIMO "
            f"input; got ndim={samples.ndim}."
        )
    C, N = samples.shape

    f_tones = [float(f) for f in tone_frequencies]
    K = len(f_tones)
    if K == 0:
        raise ValueError("tone_frequencies must contain at least one frequency.")
    if K > C:
        raise ValueError(
            f"got K={K} tones but only C={C} receive channels; need K <= C to "
            "unmix (one tone per transmitted stream)."
        )
    nyq = sampling_rate / 2.0
    for f in f_tones:
        if not (-nyq < f < nyq):
            raise ValueError(
                f"tone_frequencies entry {f} must lie in (-fs/2, fs/2) = "
                f"(±{nyq:.3g}) Hz."
            )

    # The KxK inverse is precision-sensitive (CLAUDE.md) and stays in
    # complex128, but every O(N) pass runs in the signal's working precision:
    # the tone phasors are accumulated block-wise with complex128 partials
    # (_tone_phasor_matrix), and the unmix is a well-conditioned per-sample
    # mat-vec with no long accumulation, so complex64 halves the bulk traffic
    # without touching the precision that matters.
    xw = (
        samples
        if samples.dtype == xp.complex128
        else samples.astype(xp.complex64, copy=False)
    )

    df = sampling_rate / N
    T = _tone_phasor_matrix(xw, f_tones, sampling_rate)  # (C, K) complex128

    if refine_tones:
        if search_band is None:
            search_band = 4.0 * df
        # Sub-bin refine each tone on the channel where it is strongest (one
        # batched FFT of the unique channels; single host transfer), re-extract.
        f_used = _refine_tone_frequencies(xw, T, f_tones, sampling_rate, search_band)
        T = _tone_phasor_matrix(xw, f_used, sampling_rate)
    else:
        f_used = f_tones

    # Unmix.  pinv covers the over-determined C > K case and equals the inverse
    # when square; kept in complex128 (inversion is precision-sensitive).
    W = xp.linalg.pinv(T)  # (K, C)
    demuxed = W.astype(xw.dtype) @ xw  # (K, N) working precision

    if normalize:
        # Single-pass BLAS reduction - no full-record |x|² temporary.
        p_in = xp.vdot(samples, samples).real / samples.size
        p_out = xp.mean(xp.abs(demuxed) ** 2, axis=-1, keepdims=True)  # (K, 1)
        scale = xp.sqrt(p_in / xp.where(p_out > 0, p_out, 1.0))
        demuxed = demuxed * scale

    demuxed = demuxed.astype(samples.dtype, copy=False)

    logger.info(
        "demultiplex_polarization_tones: f_tones=%s Hz, refine=%s [C=%d, K=%d, N=%d]",
        [f"{f:.3g}" for f in f_used],
        refine_tones,
        C,
        K,
        N,
    )

    if return_matrix:
        wrapped = context.return_value(demuxed)
        return wrapped, W
    return context.return_value(demuxed)


def demultiplex_polarization_tones_dynamic(
    samples: ArrayType | Signal,
    sampling_rate: float | None = None,
    tone_frequencies: Sequence[float] | None = None,
    *,
    track_bandwidth: float,
    num_taps: int | None = None,
    grid_step: int | None = None,
    refine_tones: bool = True,
    search_band: float | None = None,
    normalize: bool = True,
    trim_edges: bool = False,
    return_matrix: bool = False,
    apply: bool = True,
) -> ArrayType | tuple[Any, ...]:
    r"""
    Time-varying polarization demux from distinct per-stream CW pilot tones.

    Drifting-SOP counterpart of ``demultiplex_polarization_tones_static``.  Where
    the static routine reads a *single* Jones matrix from a whole-record DFT bin,
    this one **tracks** a slowly rotating frequency-flat mixing ``r(n) = J(n) s(n)``
    by following each pilot tone continuously in time.

    **Principle.**  Tone ``f_j`` is a CW carried *only* by transmitted stream
    ``j`` (amplitude ``α_j``).  Mixing receive channel ``i`` down by ``f_j`` and
    low-pass filtering isolates that tone's slowly-varying phasor::

        z_ij(n) = LPF{ r_i(n) · exp(-j2π f_j n / f_s) } ≈ J_ij(n) · α_j,

    because every other tone ``f_k`` (k ≠ j) lands at ``f_k - f_j`` and the data
    band is pushed away from DC, so the LPF rejects them.  Running this for all
    ``K`` tones (one mix-down per *distinct* tone frequency) yields a continuous
    estimate of the whole Jones matrix ``T(n) = J(n) diag(α)``, shape ``(C, K, N)``.
    Inverting ``T`` on a decimated time grid and interpolating back to full rate
    gives a per-sample unmixer ``W(n) = pinv(T(n))`` with::

        W(n) r(n) = diag(1/α) · s(n),

    recovering each stream up to the same trivial per-stream scale ``1/α_j`` as
    the static routine.  As with the static version each tone uniquely labels its
    stream, so there is **no polarization-permutation ambiguity**: output row
    ``j`` corresponds to ``tone_frequencies[j]``.

    **Tracking-bandwidth trade-off.**  ``track_bandwidth`` (the LPF cut-off) is
    the single design knob and is bounded on both sides:

    * It must be **≥ the SOP rotation rate**, or ``W(n)`` lags the true ``J(n)``
      and residual crosstalk remains (lag bias).
    * It must be **≤ the guard** to the nearest other tone and to the data band,
      or those leak into ``z_ij`` and corrupt the estimate.  The hard ceiling is
      roughly half the smallest tone spacing.

    If the SOP rotates faster than the available tone spacing permits to track,
    the tones are simply spaced too closely for that drift - a real feasibility
    limit; a warning is logged when ``2·track_bandwidth`` (plus the FIR
    transition) encroaches on the nearest tone spacing.

    Parameters
    ----------
    samples : array_like or Signal
        Received MIMO samples. Shape ``(C, N)`` - time on the last axis.  A
        :class:`Signal` returns a new demultiplexed :class:`Signal` (when
        ``apply=True``).
    sampling_rate : float, optional
        Sampling rate f_s in Hz.  Required for array input; ignored for
        :class:`Signal` input, which always uses the signal's own
        ``sampling_rate``.
    tone_frequencies : sequence of float
        The ``K`` distinct per-stream tone frequencies in Hz (as added at the
        TX, in transmitted-stream order).  Require ``K <= C``.  Output row ``j``
        corresponds to ``tone_frequencies[j]``.
    track_bandwidth : float
        One-sided LPF cut-off in Hz - the polarization-tracking bandwidth.  Set
        it a few times above the expected SOP rotation rate but well below the
        smallest tone spacing (see the trade-off above).
    num_taps : int, optional
        Length of the tracking low-pass FIR.  Defaults to ``~3.3·f_s/track_bandwidth``
        (the Hamming transition width that resolves ``track_bandwidth``), forced
        odd and clipped below ``N``.  Increase for sharper neighbour-tone
        rejection at the cost of longer edge transients.
    grid_step : int, optional
        Decimation (in samples) of the grid on which ``T(n)`` is inverted.
        Defaults to ``max(1, floor(f_s / (4·track_bandwidth)))`` - i.e. oversample
        the tracked process ~4x.  ``W`` is linearly interpolated between grid
        points, so a finer grid costs more inverses but tracks marginally better.
    refine_tones : bool, default True
        If ``True``, sub-bin-refine each tone centre with ``find_bias_tone`` (on
        the receive channel where it is strongest) before mixing down, absorbing
        a residual carrier frequency offset.
    search_band : float, optional
        Half-width in Hz of the per-tone peak search when ``refine_tones=True``.
        Defaults to ``4 · f_s / N``.
    normalize : bool, default True
        If ``True``, rescale each demuxed row so its mean power equals the mean
        per-channel input power (removes the ``1/α_j`` scale; preserves the
        library power invariant).  When ``trim_edges=True`` the power is measured
        over the retained interior only.
    trim_edges : bool, default False
        The tracking FIR is applied with centred (``'same'``) convolution, so the
        Jones estimate - and hence ``W(n)`` - is unreliable within ``num_taps//2``
        samples of each record end (the convolution averages in zero-padding
        there).  The **data is never filtered**, so timing is unaffected, but
        those edge samples carry residual crosstalk.  If ``True``, drop them:
        ``demuxed`` is returned as the reliable interior ``(K, N - 2·g)`` with
        ``g = num_taps//2``, together with a ``valid`` slice giving the retained
        sample range in **original** coordinates (so full-length references align
        as ``ref[..., valid]``).
    return_matrix : bool, default False
        If ``True``, also return the decimated unmixer stack ``W_grid`` and the
        sample positions ``grid_positions`` it was evaluated at (suitable for
        seeding a time-varying butterfly equalizer).  ``W_grid`` / ``grid_positions``
        always span the **full** record, even when ``trim_edges=True``.
    apply : bool, default True
        If ``True`` (default), interpolate ``W(n)`` to full rate and apply it,
        returning the demuxed signal as documented below.  If ``False``,
        **matrix-only mode**: skip the ``O(N)`` interpolate-and-apply entirely and
        return just ``(W_grid, grid_positions)`` (``return_matrix`` is implied).
        Use this when only the unmixer stack is needed - e.g. to make a
        PDL/unitarity decision and then apply a *different* factor (a polar unitary
        ``Qᴴ(n)``) without paying for a demux that would be discarded.  ``normalize``
        and ``trim_edges`` act on the applied signal, so they have **no effect**
        when ``apply=False``.

    Returns
    -------
    demuxed : array_like
        Demultiplexed streams. Same complex dtype and backend as the input; row
        ``j`` carried ``tone_frequencies[j]``.  Shape ``(K, N)``, or
        ``(K, N - 2·(num_taps//2))`` when ``trim_edges=True``.  **Omitted** when
        ``apply=False`` (the return is then ``(W_grid, grid_positions)``).
    valid : slice, optional
        Returned only if ``trim_edges=True``: the ``slice(g, N - g)`` of original
        sample indices retained in ``demuxed`` (``g = num_taps//2``).  Always
        precedes ``W_grid`` in the output tuple.
    W_grid : array_like, optional
        Returned only if ``return_matrix=True``: the ``(G, K, C)`` stack of
        per-grid-point unmixing matrices (``complex128``).
    grid_positions : array_like, optional
        Returned only if ``return_matrix=True``: the ``(G,)`` sample indices
        (``float64``) at which ``W_grid`` was evaluated.

    Raises
    ------
    ValueError
        If ``samples`` is not 2-D, ``tone_frequencies`` is empty, ``K > C``,
        ``track_bandwidth`` is not positive, or any tone frequency lies outside
        ``(-f_s/2, f_s/2)``.

    See Also
    --------
    demultiplex_polarization_tones_static : One-shot static-SOP demux (faster).
    add_pilot_tone : Add the per-stream tones at the transmitter.
    """
    context = prepare_signal_input(
        samples, function_name="demultiplex_polarization_tones_dynamic()"
    )
    samples = context.array
    sampling_rate = context.required("sampling_rate", sampling_rate)
    if tone_frequencies is None:
        raise ValueError(
            "demultiplex_polarization_tones_dynamic() requires tone_frequencies."
        )

    samples, xp, _ = dispatch(samples)
    if samples.ndim != 2:
        raise ValueError(
            "demultiplex_polarization_tones_dynamic requires a 2-D (C, N) MIMO "
            f"input; got ndim={samples.ndim}."
        )
    C, N = samples.shape

    f_tones = [float(f) for f in tone_frequencies]
    K = len(f_tones)
    if K == 0:
        raise ValueError("tone_frequencies must contain at least one frequency.")
    if K > C:
        raise ValueError(
            f"got K={K} tones but only C={C} receive channels; need K <= C to "
            "unmix (one tone per transmitted stream)."
        )
    if not (track_bandwidth > 0):
        raise ValueError(f"track_bandwidth must be positive; got {track_bandwidth}.")
    nyq = sampling_rate / 2.0
    for f in f_tones:
        if not (-nyq < f < nyq):
            raise ValueError(
                f"tone_frequencies entry {f} must lie in (-fs/2, fs/2) = "
                f"(±{nyq:.3g}) Hz."
            )

    df = sampling_rate / N
    # Working precision for all O(N) traffic (complex64 unless the caller
    # supplied complex128); the batched inverse below stays complex128.
    xw = (
        samples
        if samples.dtype == xp.complex128
        else samples.astype(xp.complex64, copy=False)
    )

    # --- Tracking low-pass design + feasibility check ------------------------
    if num_taps is None:
        # Hamming transition width ≈ 3.3·fs/num_taps; size it to resolve the
        # requested tracking bandwidth.  Force odd; keep it shorter than N.
        num_taps = int(round(3.3 * sampling_rate / track_bandwidth))
        num_taps += 1 - (num_taps % 2)  # nearest odd >= value
        num_taps = max(num_taps, 3)
    num_taps = min(int(num_taps), (N // 2) * 2 - 1)
    h = fir_taps(sampling_rate, num_taps, track_bandwidth, btype="low")

    # Edge guard: 'same' convolution corrupts num_taps//2 samples at each end.
    # num_taps is clipped < N above, so the retained interior is always non-empty.
    guard = num_taps // 2 if trim_edges else 0

    if K > 1:
        sorted_f = sorted(f_tones)
        d_min = min(b - a for a, b in zip(sorted_f, sorted_f[1:]))
        transition = 3.3 * sampling_rate / num_taps
        if 2.0 * track_bandwidth + transition >= d_min:
            logger.warning(
                "demultiplex_polarization_tones_dynamic: tracking bandwidth "
                "(%.3g Hz, FIR transition %.3g Hz) approaches the nearest tone "
                "spacing %.3g Hz - neighbouring tones may leak into the Jones "
                "estimate. Reduce track_bandwidth or widen the tone spacing.",
                track_bandwidth,
                transition,
                d_min,
            )

    # --- Optional sub-bin tone refinement (static one-shot bin picks, per
    #     tone, the receive channel where it is strongest) --------------------
    if refine_tones:
        if search_band is None:
            search_band = 4.0 * df
        T0 = _tone_phasor_matrix(xw, f_tones, sampling_rate)  # (C, K)
        f_used = _refine_tone_frequencies(xw, T0, f_tones, sampling_rate, search_band)
    else:
        f_used = f_tones

    # --- Decimated inversion grid --------------------------------------------
    if grid_step is None:
        grid_step = max(1, int(sampling_rate / (4.0 * track_bandwidth)))
    grid_step = int(min(max(grid_step, 1), N))
    grid_np = np.arange(0, N, grid_step, dtype=np.int64)
    if int(grid_np[-1]) != N - 1:
        grid_np = np.concatenate([grid_np, [N - 1]])  # pin the last sample
    G = int(grid_np.shape[0])
    grid_positions = xp.asarray(grid_np, dtype=xp.float64)

    # --- Track the Jones matrix at the grid points ---------------------------
    # T[i, j, g] = LPF{ r_i(n) · exp(-j2π f_j n/fs) }(g) ≈ J_ij(g)·α_j, with a
    # centred ('same') linear-phase FIR, so the LPF group delay is compensated
    # and the estimate aligns in time with the input.  Only the G grid points
    # feed the batched inverse, so the tracker is evaluated there directly
    # (windowed gather + GEMM in _jones_at_grid_points) instead of filtering
    # all N samples and discarding grid_step-1 of every grid_step outputs.
    # The batched inverse follows the RLS-style double-precision convention,
    # hence the complex128 (G, C, K) stack.
    if G * num_taps <= 24 * K * N:
        Tg = _jones_at_grid_points(xw, h, f_used, grid_np, sampling_rate)
    else:
        # Dense grid (grid_step ≪ num_taps): per-grid-point windows would touch
        # far more elements than the record itself, so keep the full-rate
        # batched-FIR formulation.  The phase ramp 2π·f·n/fs reaches ~1e7 rad
        # over a long capture, so it MUST be formed and wrapped in float64 -
        # float32 loses the integer turn count and corrupts every tone phasor.
        # The wrapped phase, the exp, the mix-down, and the averaging LPF are
        # well-conditioned, so they run in working precision, with FFT-FIR
        # round-off (~√N_fft·ε ≈ 5e-5) far below the demux crosstalk floor.
        two_pi = 2.0 * np.pi
        real_dtype = xp.float64 if xw.dtype == xp.complex128 else xp.float32
        f_arr = xp.asarray([float(fj) for fj in f_used], dtype=xp.float64)  # (K,)
        n = xp.arange(N, dtype=xp.float64)
        ph = two_pi * f_arr[:, None] * n[None, :] / sampling_rate  # (K, N) float64
        ph -= xp.round(ph / two_pi) * two_pi  # wrap in float64 (essential)
        carrier = xp.exp(-1j * ph.astype(real_dtype))  # (K, N) working precision
        mixed = xw[:, None, :] * carrier[None, :, :]  # (C, K, N)
        # One batched linear-phase FIR over (C·K) rows instead of K calls.
        T_t = cast(ArrayType, fir_filter(mixed.reshape(C * K, N), h, axis=-1)).reshape(
            C, K, N
        )
        idx = xp.asarray(grid_np)
        Tg = xp.moveaxis(T_t[:, :, idx], 2, 0).astype(xp.complex128)  # (G, C, K)
    Th = xp.conj(xp.swapaxes(Tg, -1, -2))  # (G, K, C)
    gram = Th @ Tg  # (G, K, K)
    # Tikhonov regularisation keeps the batched inverse well-conditioned when the
    # instantaneous SOP nearly aligns two streams (gram -> singular).
    diag_mean = xp.real(xp.trace(gram, axis1=-2, axis2=-1)) / K  # (G,)
    eye = xp.eye(K, dtype=xp.complex128)
    ridge = (1e-9 * diag_mean)[:, None, None] * eye[None, :, :]
    Wg = xp.linalg.inv(gram + ridge) @ Th  # (G, K, C) - batched, CPU+GPU

    if not apply:
        # Matrix-only mode: skip the O(N) interpolate-and-apply.  normalize and
        # trim_edges act on the applied signal, so they are no-ops here.
        logger.info(
            "demultiplex_polarization_tones_dynamic (matrix-only): f_tones=%s Hz, "
            "refine=%s, track_bw=%.3g Hz, taps=%d, grid_step=%d, G=%d "
            "[C=%d, K=%d, N=%d]",
            [f"{f:.3g}" for f in f_used],
            refine_tones,
            track_bandwidth,
            num_taps,
            grid_step,
            G,
            C,
            K,
            N,
        )
        return Wg, grid_positions

    # Interpolate W(n) to full rate and apply it (block-vectorised GEMMs in the
    # shared helper) as a fixed forward pass.
    demuxed = apply_interpolated_matrix(samples, Wg, grid_positions)  # (K, N)

    # Drop the FIR edge transient (the data is untouched; only W is unreliable
    # there).  ``valid`` reports the retained range in original coordinates.
    valid = slice(guard, N - guard)
    demuxed = demuxed[:, valid]

    if normalize:
        p_in = xp.mean(xp.abs(samples[:, valid]) ** 2)
        p_out = xp.mean(xp.abs(demuxed) ** 2, axis=-1, keepdims=True)  # (K, 1)
        scale = xp.sqrt(p_in / xp.where(p_out > 0, p_out, 1.0))
        demuxed = demuxed * scale

    demuxed = demuxed.astype(samples.dtype)

    logger.info(
        "demultiplex_polarization_tones_dynamic: f_tones=%s Hz, refine=%s, "
        "track_bw=%.3g Hz, taps=%d, grid_step=%d, G=%d [C=%d, K=%d, N=%d]",
        [f"{f:.3g}" for f in f_used],
        refine_tones,
        track_bandwidth,
        num_taps,
        grid_step,
        G,
        C,
        K,
        N,
    )

    out: tuple[Any, ...] = (demuxed,)
    if trim_edges:
        out = out + (valid,)
    if return_matrix:
        out = out + (Wg, grid_positions)
    if context.signal is not None:
        out = (context.return_value(out[0]), *out[1:])
    return out[0] if len(out) == 1 else out
