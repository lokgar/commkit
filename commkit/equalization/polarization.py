"""Pilot-tone polarization demultiplexing.

Each transmitted stream carries a CW pilot tone at its own frequency.
:func:`demultiplex_polarization_tones_static` estimates one Jones matrix from
the tones; :func:`demultiplex_polarization_tones_dynamic` tracks it over time
(a :class:`JonesTrack`), and :func:`apply_interpolated_matrix` applies such a
track to the samples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast, overload

import numpy as np

from .. import _fft
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..filtering import fir_filter, fir_taps
from ..logger import logger

__all__ = [
    "JonesTrack",
    "apply_interpolated_matrix",
    "demultiplex_polarization_tones_dynamic",
    "demultiplex_polarization_tones_static",
]

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
    from 0 to 1 across the cell).  This is the apply half of the dynamic pilot demux,
    but it is a generic operator: given any per-grid-point matrix stack (an inverse
    ``W(n)``, a polar unitary ``Qᴴ(n)``, a butterfly seed, ...) it interpolates and
    applies it as a fixed forward pass.

    Within each uniform cell the product reduces to two batched ``(K, C)``
    matrix products, ``M[g]·X`` and ``dM·(X·t)``, instead of building a per-sample
    ``(L, K, C)`` matrix and an einsum mat-vec, so the cost is a handful of batched
    kernels rather than an ``O(N)`` Python/launch-bound loop.  The ``O(N)`` data
    path runs in ``complex64`` (a well-conditioned mix with no long accumulation),
    so a ``matrix_grid`` built by a double-precision inverse/SVD keeps its
    precision while the bulk traffic moves at half the bandwidth.

    Parameters
    ----------
    samples : array_like or Signal
        Input channels, ``(C, N)``.  A :class:`Signal` returns a new
        :class:`Signal`.
    matrix_grid : array_like
        ``(G, K, C)`` matrices at the grid points, mapping the ``C`` inputs to
        ``K`` outputs.
    grid_positions : array_like
        ``(G,)`` sample indices at which ``matrix_grid`` was evaluated; the
        interior must be uniformly spaced (the last point may be pinned to
        ``N - 1``).

    Returns
    -------
    ndarray or Signal
        ``M(n) · samples[:, n]``, ``(K, N)``, same dtype and device as
        ``samples``.
    """
    signal_adapter = adapt_signal(samples, function_name="apply_interpolated_matrix()")
    samples = signal_adapter.array

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
    return signal_adapter.wrap_samples(out.astype(samples.dtype, copy=False))


# -----------------------------------------------------------------------------
# TONE-BASED POLARIZATION DEMULTIPLEXING (Signal-aware)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class JonesTrack:
    """Time-varying unmixing estimate of
    :func:`demultiplex_polarization_tones_dynamic`.

    Attributes
    ----------
    matrix_grid : ndarray
        ``(G, K, C)`` stack of unmixing matrices (``complex128``) at the grid
        points; apply with :func:`apply_interpolated_matrix`.
    grid_positions : ndarray
        ``(G,)`` sample indices (``float64``) of the grid points.
    valid : slice
        Samples of the record that the demuxed output keeps (all of them
        unless ``trim_edges=True``).
    """

    matrix_grid: ArrayType
    grid_positions: ArrayType
    valid: slice


_EXTRACT_CHUNK = 1 << 20  # samples per block in the chunked tone-phasor GEMM


def _tone_phasor_matrix(xw: ArrayType, freqs: Any, sampling_rate: float) -> ArrayType:
    r"""Tone-phasor matrix ``T[i, j] = (1/N) Σ_n xw[i, n]·exp(-j2π f_j n/fs)``.

    The whole-record accumulation is precision-sensitive (it feeds a matrix
    inverse), but running the O(N) GEMM in complex128 would put the entire data
    path on the slow FP64 units.  Instead the record is processed in
    ``_EXTRACT_CHUNK`` blocks: each block's ``(C, chunk) @ (chunk, K)`` product
    runs in the signal's working precision (complex64 for complex64 input) and
    the small ``(C, K)`` per-block partials are accumulated in complex128, so
    the rounding error no longer grows with N beyond a block, at complex64
    bandwidth.
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
    freqs: Any,
    sampling_rate: float,
    search_band: float,
) -> list[float]:
    """Refine each tone frequency on the receive channel where it is strongest.

    Two stages replace one ``BiasTone`` estimate per tone, each of which ran
    its own zero-padded FFT of the full record:

    1. Coarse: one batched FFT, in the working precision, of the distinct
       strongest channels only, and the log-parabolic peak fit on the
       device.  It is good to a fraction of a bin but ill-conditioned for a
       tone exactly on a bin, whose neighbors are sinc nulls.
    2. Fine: the phase slope between two half-records.  The exact tone
       phasor of each half at the coarse frequency (chunked matrix
       products) advances between the segment centroids by the residual
       offset ``δ = ∠(z_b·conj(z_a))·fs/(2π·Δc)``, unambiguous for
       ``|δ| < 1`` bin (which stage 1 guarantees) and far more accurate than
       either peak fit.  The mix-down needs it: a tone-frequency error
       leaves a slow residual rotation on the demultiplexed output.
    """
    from ..frequency import _refine_tones_from_spectrum

    xw, xp, _ = dispatch(xw)
    N = int(xw.shape[-1])
    K = len(list(freqs))
    best_ch = to_device(xp.argmax(xp.abs(T), axis=0), "cpu")  # (K,) host ints
    uniq, inv = np.unique(np.asarray(best_ch), return_inverse=True)
    xr = xw[xp.asarray(uniq)]  # (U, N) unique strongest channels only
    X = _fft.fft(xr, axis=-1)
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
    freqs: Any,
    grid_np: np.ndarray,
    grid: ArrayType,
    sampling_rate: float,
) -> ArrayType:
    r"""Tone-tracked Jones estimate evaluated at the grid points only.

    The output of the mix-down and the centered tracking FIR at sample ``g``
    factors as::

        T[c, k, g] = e^{-j2π f_k g/fs} · Σ_q h̃_k[q] · r_c[g - lead + q],
        h̃_k[q]    = h[L-1-q] · e^{-j2π f_k (q - lead)/fs},   lead = ⌈(L-1)/2⌉,

    a bandpass filter at ``+f_k``.  So instead of building the ``(C, K, N)``
    mixed array and filtering every sample to keep one in ``grid_step``, this
    gathers the ``(chunk, L)`` sample windows around the grid points and
    applies all K modulated tap vectors in one matrix product per chunk,
    without temporaries of the record's size.  Zero padding at the record
    ends reproduces ``mode='same'`` exactly, so the ``num_taps//2`` edge
    guard is unchanged.

    Parameters
    ----------
    xw : ndarray
        Received samples in the working precision (complex64 or complex128),
        ``(C, N)``.
    h : numpy.ndarray
        Tracking low-pass FIR taps, ``(L,)``.
    freqs : sequence of float
        The K tracked tone frequencies in Hz.
    grid_np : numpy.ndarray
        Integer sample indices of the inversion grid, ``(G,)``, on the host
        (for the edge-window search).
    grid : ndarray
        The same indices, int64, on the device of ``xw``.
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    ndarray
        Jones stack, complex128, ``(G, C, K)``, ready for the batched Gram
        inverse.
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

    gp = grid.astype(xp.float64)
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
    starts = (grid - lead).astype(idx_dtype)  # (G,) on the device
    Tg_work = xp.empty((C, G, K), dtype=xw.dtype)

    def _fill(a: int, b: int, masked: bool) -> None:
        for s0 in range(a, b, chunk):
            s1 = min(s0 + chunk, b)
            idx = starts[s0:s1, None] + off[None, :]  # (chunk, L)
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


def _check_tone_inputs(
    samples: ArrayType, tone_frequencies: Any, sampling_rate: float, name: str
) -> list[float]:
    """Validate the input of the tone demultiplexers; return the tone list.

    ``samples`` must be ``(C, N)``, with between 1 and C tones, each strictly
    inside ``(-fs/2, fs/2)``.
    """
    if samples.ndim != 2:
        raise ValueError(
            f"{name} requires a 2-D (C, N) MIMO input; got ndim={samples.ndim}."
        )
    C = samples.shape[0]
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
    return f_tones


def demultiplex_polarization_tones_static(
    samples: ArrayType | Signal,
    *,
    tone_frequencies: Sequence[float],
    sampling_rate: float | None = None,
    refine_tones: bool = True,
    search_band: float | None = None,
    normalize: bool = True,
    return_matrix: bool = False,
) -> ArrayType | Signal | tuple[ArrayType | Signal, ArrayType]:
    r"""Demultiplex polarizations once from distinct per-stream CW pilot tones.

    Undoes a frequency-flat polarization (or spatial) mixing by inverting the
    channel's Jones matrix, read from pilot tones at a distinct frequency on
    each transmitted stream (see :func:`~commkit.spectral.add_pilot_tone`
    with one ``frequency`` per channel).

    **Principle.**  With tone ``f_j`` carried only by stream ``j`` and a
    frequency-flat mixing ``r = J s`` (e.g.
    :func:`~commkit.impairments.apply_polarization_mixing`), the complex
    amplitude of tone ``f_j`` in received channel ``i`` is
    ``T[i, j] = J[i, j] · α_j``, with ``α_j`` the transmitted tone amplitude
    of stream ``j``.  The measured matrix of tone phasors is therefore
    ``T = J · diag(α)``, and ``W = pinv(T)`` unmixes::

        W r = diag(1/α) · J^{-1} J s = diag(1/α) · s,

    which recovers each stream up to a complex scale ``1/α_j`` (removed by
    ``normalize`` and the downstream CPR).  Each tone labels its stream, so
    output row ``j`` is always the stream of ``tone_frequencies[j]``: there
    is no permutation ambiguity, unlike after blind equalization (see
    :func:`commkit.recovery.resolve_channel_permutation`).

    **Cost.**  Tones from ``add_pilot_tone`` sit on the FFT grid, so one DFT
    bin is the exact (orthogonal, maximum-likelihood) estimate of each tone
    phasor.  The whole operation is two small matrix products (``T`` from a
    ``(C, N)·(N, K)`` product, the output from ``(K, C)·(C, N)``) and one
    ``K x K`` inverse: no iteration, and far cheaper than a full FFT.

    **Scope.**  The SOP must be frequency-flat and constant: the
    whole-record DFT bin estimates one Jones matrix.  If the SOP drifts
    noticeably over the capture (a long record or a low symbol rate, whose
    duration exceeds the SOP coherence time), the averaged phasors are
    biased and attenuated, and the one-shot inverse leaves crosstalk that
    grows with time: use :func:`demultiplex_polarization_tones_dynamic`
    instead.  For PMD (DGD), seed a butterfly equalizer (``cma``,
    ``block_lms``) with the returned unmixer (``return_matrix=True``).

    Parameters
    ----------
    samples : array_like or Signal
        Received MIMO samples, ``(C, N)``.  A :class:`Signal` returns a new
        demultiplexed :class:`Signal`.
    tone_frequencies : sequence of float
        The ``K <= C`` distinct tone frequencies in Hz, one per stream, in the
        order of the transmitted streams.  Output row ``j`` is the stream of
        ``tone_frequencies[j]``.
    sampling_rate : float, optional
        Sampling rate ``f_s`` in Hz.  Taken from the Signal; required for
        array input.  A value that disagrees with the Signal raises.
    refine_tones : bool, default True
        Refine each tone's frequency to a fraction of a bin with the
        ``BiasTone`` estimator (on the receive channel where the tone is
        strongest) before extracting it, which absorbs a residual carrier
        frequency offset that moved the tone off its bin.  ``False``
        extracts at ``tone_frequencies`` exactly.
    search_band : float, optional
        Half-width in Hz of each tone's peak search with
        ``refine_tones=True``; defaults to ``4 * f_s / N`` (a few FFT bins).
        Widen it if the carrier offset can be larger, but keep it inside the
        guard between tone and data, so the data band never wins the search.
    normalize : bool, default True
        Rescale each output row to the mean input power per channel, which
        removes the scale ``1/α_j`` and keeps the library's power
        convention.
    return_matrix : bool, default False
        Also return the ``(K, C)`` unmixing matrix ``W``.

    Returns
    -------
    demuxed : ndarray or Signal
        Demultiplexed streams, ``(K, N)``: one row per transmitted stream
        (``K`` tones), not per receive channel, so the ``C`` inputs map to
        ``K`` streams (in the usual dual-pol case ``K == C == 2``).  Same
        complex dtype and device as the input; row ``j`` carried
        ``tone_frequencies[j]``.
    W : ndarray
        Only with ``return_matrix=True``: the ``(K, C)`` complex128 unmixing
        matrix, the estimated inverse Jones matrix up to a scale per stream;
        a seed for a butterfly equalizer.

    Raises
    ------
    ValueError
        If ``samples`` is not 2-D, ``tone_frequencies`` is empty, ``K > C``, or
        any tone frequency lies outside ``(-f_s/2, f_s/2)``.

    See Also
    --------
    commkit.spectral.add_pilot_tone : Add the per-stream tones at the transmitter.
    demultiplex_polarization_tones_dynamic : Demultiplexing of a drifting SOP.
    """
    signal_adapter = adapt_signal(
        samples, function_name="demultiplex_polarization_tones_static()"
    )
    samples = signal_adapter.array
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)

    samples, xp, _ = dispatch(samples)
    f_tones = _check_tone_inputs(
        samples,
        tone_frequencies,
        sampling_rate,
        "demultiplex_polarization_tones_static",
    )
    C, N = samples.shape
    K = len(f_tones)

    # The KxK inverse is precision-sensitive and stays in
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
        wrapped = signal_adapter.wrap_samples(demuxed)
        return wrapped, W
    return signal_adapter.wrap_samples(demuxed)


def demultiplex_polarization_tones_dynamic(
    samples: ArrayType | Signal,
    *,
    tone_frequencies: Sequence[float],
    sampling_rate: float | None = None,
    track_bandwidth: float,
    num_taps: int | None = None,
    grid_step: int | None = None,
    refine_tones: bool = True,
    search_band: float | None = None,
    normalize: bool = True,
    trim_edges: bool = False,
    return_matrix: bool = False,
    apply: bool = True,
) -> ArrayType | Signal | JonesTrack | tuple[ArrayType | Signal, JonesTrack]:
    r"""Demultiplex drifting polarizations from distinct per-stream CW pilot tones.

    The counterpart of :func:`demultiplex_polarization_tones_static` for a
    drifting SOP.  Where that function reads one Jones matrix from a
    whole-record DFT bin, this one tracks a slowly rotating frequency-flat
    mixing ``r(n) = J(n) s(n)`` by following each pilot tone in time.

    **Principle.**  Tone ``f_j`` is a CW carried only by stream ``j``
    (amplitude ``α_j``).  Mixing receive channel ``i`` down by ``f_j`` and
    low-pass filtering isolates that tone's slowly varying phasor::

        z_ij(n) = LPF{ r_i(n) · exp(-j2π f_j n / f_s) } ≈ J_ij(n) · α_j,

    because every other tone ``f_k`` (``k ≠ j``) lands at ``f_k - f_j`` and
    the data band moves away from DC, so the low-pass rejects them.  Doing
    this for all ``K`` tones (one mix-down per distinct tone frequency)
    gives a continuous estimate of the whole Jones matrix
    ``T(n) = J(n) diag(α)``, ``(C, K, N)``.  Inverting ``T`` on a decimated
    time grid and interpolating back to the full rate gives a per-sample
    unmixer ``W(n) = pinv(T(n))`` with::

        W(n) r(n) = diag(1/α) · s(n),

    which recovers each stream up to the scale ``1/α_j``, as in the static
    case.  Each tone labels its stream here too, so there is no permutation
    ambiguity: output row ``j`` is the stream of ``tone_frequencies[j]``.

    **Tracking bandwidth.**  ``track_bandwidth``, the low-pass cutoff, is the
    one design parameter, bounded on both sides:

    * at least the SOP rotation rate, or ``W(n)`` lags the true ``J(n)`` and
      crosstalk remains;
    * at most the guard to the nearest other tone and to the data band, or
      those leak into ``z_ij`` and corrupt the estimate; the hard ceiling is
      about half the smallest tone spacing.

    An SOP that rotates faster than the tone spacing allows to track is a
    real limit of the design: the tones are too close for that drift.  A
    warning is logged when ``2·track_bandwidth`` (plus the FIR transition)
    reaches the nearest tone spacing.

    Parameters
    ----------
    samples : array_like or Signal
        Received samples, ``(C, N)``.  A Signal gives a new demultiplexed
        Signal when ``apply=True``.
    tone_frequencies : sequence of float
        The ``K`` distinct per-stream tone frequencies in Hz, in transmitted
        stream order; ``K <= C``.  Output row ``j`` is the stream of
        ``tone_frequencies[j]``.
    sampling_rate : float, optional
        Sampling rate ``f_s`` in Hz.  Taken from the Signal; required for
        array input.  A value that disagrees with the Signal raises.
    track_bandwidth : float
        One-sided cutoff of the tracking low-pass in Hz, the polarization
        tracking bandwidth.  Set it a few times above the expected SOP
        rotation rate and well below the smallest tone spacing (see above).
    num_taps : int, optional
        Length of the tracking low-pass FIR.  Defaults to about
        ``3.3·f_s/track_bandwidth`` (the Hamming transition width that
        resolves ``track_bandwidth``), made odd and kept below ``N``.  More
        taps reject neighboring tones more sharply but lengthen the edge
        transients.
    grid_step : int, optional
        Spacing in samples of the grid on which ``T(n)`` is inverted.
        Defaults to ``max(1, floor(f_s/(4·track_bandwidth)))``, four grid
        points per tracking period.  ``W`` is interpolated linearly between
        grid points, so a finer grid costs more inversions for slightly
        better tracking.
    refine_tones : bool, default True
        Refine each tone frequency below the DFT bin spacing with the
        ``BiasTone`` estimator, on the receive channel where the tone is
        strongest, before mixing down.  This absorbs a residual carrier
        frequency offset.
    search_band : float, optional
        Half-width in Hz of the per-tone peak search when
        ``refine_tones=True``.  Defaults to ``4·f_s/N``.
    normalize : bool, default True
        Scale each output row so its mean power equals the mean per-channel
        input power.  This removes the ``1/α_j`` scale and keeps the library's
        power invariant.  With ``trim_edges=True`` the power is measured over
        the retained interior only.
    trim_edges : bool, default False
        Drop the samples within ``g = num_taps//2`` of each record end.  The
        tracking FIR runs as a centered (``'same'``) convolution, so the Jones
        estimate there averages in zero padding and leaves residual
        crosstalk.  The data itself is never filtered, so timing is
        unaffected.  The output is then the interior ``(K, N - 2·g)``, and
        ``JonesTrack.valid`` gives the retained range in the original
        sample indices (align references as ``ref[..., valid]``).  Array
        input only: trimming would misalign a Signal's reference.
    return_matrix : bool, default False
        Also return the :class:`JonesTrack`: the unmixer stack on the
        decimated grid, the sample positions of its grid points (usable to
        seed a time-varying butterfly equalizer) and the ``valid`` range.
        The stack always spans the full record, also with
        ``trim_edges=True``.
    apply : bool, default True
        Interpolate ``W(n)`` to the full rate and apply it.  With ``False``,
        skip the ``O(N)`` interpolation and product and return only the
        :class:`JonesTrack`, for example to test PDL or unitarity first and
        then apply a different factor (the polar unitary ``Qᴴ(n)``).
        ``normalize`` and ``trim_edges`` act on the applied output, so they
        have no effect with ``apply=False``.

    Returns
    -------
    demuxed : ndarray or Signal
        Demultiplexed streams, ``(K, N)``, or ``(K, N - 2·(num_taps//2))``
        with ``trim_edges=True``; same complex dtype and device as the
        input.  Row ``j`` is the stream of ``tone_frequencies[j]``.
    track : JonesTrack
        Returned as ``(demuxed, track)`` with ``return_matrix=True``, and
        alone with ``apply=False``.

    Raises
    ------
    ValueError
        If ``samples`` is not 2-D, ``tone_frequencies`` is empty, ``K > C``,
        ``track_bandwidth`` is not positive, or a tone frequency lies outside
        ``(-f_s/2, f_s/2)``.

    See Also
    --------
    demultiplex_polarization_tones_static : Faster demultiplexing for a static SOP.
    commkit.spectral.add_pilot_tone : Add the per-stream tones at the transmitter.
    """
    signal_adapter = adapt_signal(
        samples, function_name="demultiplex_polarization_tones_dynamic()"
    )
    samples = signal_adapter.array
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)

    samples, xp, _ = dispatch(samples)
    f_tones = _check_tone_inputs(
        samples,
        tone_frequencies,
        sampling_rate,
        "demultiplex_polarization_tones_dynamic",
    )
    C, N = samples.shape
    K = len(f_tones)
    if trim_edges and signal_adapter.signal is not None:
        raise ValueError(
            "demultiplex_polarization_tones_dynamic(): trim_edges drops samples "
            "and would misalign the Signal's reference; use array input."
        )
    if not (track_bandwidth > 0):
        raise ValueError(f"track_bandwidth must be positive; got {track_bandwidth}.")

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
    h = fir_taps(
        sampling_rate=sampling_rate,
        num_taps=num_taps,
        cutoff=track_bandwidth,
        btype="low",
    )

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
                "spacing %.3g Hz - neighboring tones may leak into the Jones "
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
    grid = xp.arange(0, N, grid_step, dtype=xp.int64)  # grid_np, built on the device
    if int(grid.shape[0]) != G:
        grid = xp.concatenate([grid, xp.full(1, N - 1, dtype=xp.int64)])
    grid_positions = grid.astype(xp.float64)

    # --- Track the Jones matrix at the grid points ---------------------------
    # T[i, j, g] = LPF{ r_i(n) · exp(-j2π f_j n/fs) }(g) ≈ J_ij(g)·α_j, with a
    # centered ('same') linear-phase FIR, so the LPF group delay is compensated
    # and the estimate aligns in time with the input.  Only the G grid points
    # feed the batched inverse, so the tracker is evaluated there directly
    # (windowed gather + GEMM in _jones_at_grid_points) instead of filtering
    # all N samples and discarding grid_step-1 of every grid_step outputs.
    # The batched inverse follows the RLS-style double-precision convention,
    # hence the complex128 (G, C, K) stack.
    if G * num_taps <= 24 * K * N:
        Tg = _jones_at_grid_points(xw, h, f_used, grid_np, grid, sampling_rate)
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
        T_t = cast(ArrayType, fir_filter(mixed.reshape(C * K, N), h)).reshape(C, K, N)
        Tg = xp.moveaxis(T_t[:, :, grid], 2, 0).astype(xp.complex128)  # (G, C, K)
    Th = xp.conj(xp.swapaxes(Tg, -1, -2))  # (G, K, C)
    gram = Th @ Tg  # (G, K, K)
    # Tikhonov regularization keeps the batched inverse well-conditioned when the
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
        return JonesTrack(Wg, grid_positions, slice(0, N))

    # Interpolate W(n) to full rate and apply it (block-vectorized GEMMs in the
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

    wrapped = signal_adapter.wrap_samples(demuxed)
    if return_matrix:
        return wrapped, JonesTrack(Wg, grid_positions, valid)
    return wrapped
