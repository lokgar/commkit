"""Polarization demultiplexing.

:func:`estimate_polarization` reads the unmixing (inverse Jones) matrix, a
:class:`PolarizationEstimate`, with the method :class:`PolarizationTones`
(returning a :class:`PolarizationTonesEstimate`):
each transmitted stream carries a CW pilot tone at its own frequency, read
once for the record or tracked over time.  :func:`correct_polarization`
applies the estimate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from .. import _fft
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import S, adapt_signal
from ..core.signal import Signal
from ..filtering import fir_filter, fir_taps
from ..logger import logger

__all__ = [
    "PolarizationEstimate",
    "PolarizationTones",
    "PolarizationTonesEstimate",
    "correct_polarization",
    "estimate_polarization",
]

# -----------------------------------------------------------------------------
# TIME-VARYING MATRIX APPLY (Signal-aware)
# -----------------------------------------------------------------------------
# Interpolate-and-apply core of correct_polarization().


def _apply_interpolated_matrix(
    samples: ArrayType,
    matrix_grid: ArrayType,
    grid_positions: ArrayType,
) -> ArrayType:
    r"""Apply a time-varying matrix ``M(n)`` to ``samples``, interpolated from a grid.

    Output sample ``n`` is ``M(n) · samples[:, n]``, where ``M(n)`` is the exact
    linear blend ``M[g] + (M[g+1]-M[g])·t`` inside grid cell ``g`` (``t`` ramps
    from 0 to 1 across the cell).

    Within each uniform cell the product reduces to two batched ``(K, C)``
    matrix products, ``M[g]·X`` and ``dM·(X·t)``, instead of building a per-sample
    ``(L, K, C)`` matrix and an einsum mat-vec, so the cost is a handful of batched
    kernels rather than an ``O(N)`` Python/launch-bound loop.  The data path
    runs in the samples' precision (complex64 for single-precision input): the
    matrices, typically from a double-precision inverse or SVD, are cast to it.

    Parameters
    ----------
    samples : array_like
        Input channels, ``(C, N)``.
    matrix_grid : array_like
        ``(G, K, C)`` matrices at the grid points, ``G >= 2``, mapping the
        ``C`` inputs to ``K`` outputs.
    grid_positions : array_like
        ``(G,)`` sample indices at which ``matrix_grid`` was evaluated; the
        interior must be uniformly spaced (the last point may be pinned to
        ``N - 1``).

    Returns
    -------
    ndarray
        ``M(n) · samples[:, n]``, ``(K, N)``, same dtype and device as
        ``samples``.
    """
    samples, xp, _ = dispatch(samples)
    C, N = samples.shape
    cdtype = xp.result_type(samples.dtype, xp.complex64)
    rdtype = xp.empty(0, dtype=cdtype).real.dtype
    M = xp.asarray(matrix_grid).astype(cdtype, copy=False)  # (G, K, C)
    gp = xp.asarray(grid_positions, dtype=xp.float64)  # (G,)
    G, K = int(M.shape[0]), int(M.shape[1])
    xc = samples.astype(cdtype, copy=False)
    step = int(round(float(gp[1] - gp[0]))) if G > 1 else N
    out = xp.empty((K, N), dtype=cdtype)

    nblk = (N - 1) // step if step > 0 else 0  # full uniform cells over [0, nblk·step)
    bulk = nblk * step
    if nblk > 0:
        Xb = xc[:, :bulk].reshape(C, nblk, step).transpose(1, 0, 2)  # (nblk, C, step)
        M0 = M[:nblk]  # (nblk, K, C)
        dM = M[1 : nblk + 1] - M0
        t = (xp.arange(step, dtype=rdtype) / step).astype(cdtype)
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
        frac = ((nn - gp[lo]) / (gp[lo + 1] - gp[lo])).astype(cdtype)
        M_full = M[lo] + (M[lo + 1] - M[lo]) * frac[:, None, None]  # (L, K, C)
        out[:, bulk:] = xp.einsum("lkc,cl->kl", M_full, xc[:, bulk:])
    return out.astype(samples.dtype, copy=False)


# -----------------------------------------------------------------------------
# TONE-BASED POLARIZATION DEMULTIPLEXING (Signal-aware)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PolarizationEstimate:
    """Unmixing (inverse Jones) matrices over time, applied by
    :func:`correct_polarization`.

    Returned by :func:`estimate_polarization`; build one directly to
    apply any other per-grid-point matrix stack (a polar unitary ``Qᴴ(n)``,
    a butterfly seed, ...).

    Attributes
    ----------
    matrix_grid : ndarray
        ``(G, K, C)`` unmixing matrices (``complex128`` from the estimator)
        mapping ``C`` received channels to ``K`` streams: one matrix
        (``G = 1``) for a static estimate, else the matrices at
        ``grid_positions``, linearly interpolated in between.
    grid_positions : ndarray
        ``(G,)`` sample indices (``float64``) of the matrices; the interior
        uniformly spaced, the last point may be pinned to ``N - 1``.
        ``[0]`` for a static estimate.
    edge : int, default 0
        Samples at each end of the record where a tracked estimate averaged
        in the zero padding of its tracking filter;
        ``correct_polarization(trim_edges=True)`` drops them (align a
        reference as ``ref[..., edge : N - edge]``).
    """

    matrix_grid: ArrayType
    grid_positions: ArrayType
    edge: int = 0

    @property
    def matrix(self) -> ArrayType:
        """The ``(K, C)`` unmixing matrix of a static estimate, a seed for a
        butterfly equalizer (``cma``, ``block_lms``).

        Raises
        ------
        ValueError
            For a tracked estimate (more than one matrix).
        """
        if self.matrix_grid.shape[0] != 1:
            raise ValueError(
                f"PolarizationEstimate.matrix: the track has {self.matrix_grid.shape[0]} "
                "matrices; matrix_grid holds them all."
            )
        return self.matrix_grid[0]


@dataclass(frozen=True)
class PolarizationTonesEstimate(PolarizationEstimate):
    """Result of :class:`PolarizationTones`.

    The constructor takes the same fields: those of
    :class:`PolarizationEstimate`, then the attribute below, by keyword.

    Attributes
    ----------
    tone_frequencies : tuple of float
        The tone frequencies the estimate used, after refinement, in Hz;
        row ``j`` of the unmixing matrices recovers the stream of tone ``j``.
    """

    tone_frequencies: tuple[float, ...] = field(kw_only=True)


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

    Two stages replace one ``SpectralPeak`` estimate per tone, each of which ran
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
    from .._peaks import refine_tones

    xw, xp, _ = dispatch(xw)
    N = int(xw.shape[-1])
    K = len(list(freqs))
    best_ch = to_device(xp.argmax(xp.abs(T), axis=0), "cpu")  # (K,) host ints
    uniq, inv = np.unique(np.asarray(best_ch), return_inverse=True)
    xr = xw[xp.asarray(uniq)]  # (U, N) unique strongest channels only
    X = _fft.fft(xr, axis=-1)
    f_coarse = refine_tones(
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


@dataclass(frozen=True)
class PolarizationTones:
    r"""Polarization unmixing from distinct per-stream CW pilot tones, for
    :func:`estimate_polarization` and :func:`correct_polarization`.

    Reads the inverse of a frequency-flat polarization (or spatial) mixing
    from pilot tones at a distinct frequency on each transmitted stream (see
    :func:`~commkit.spectral.add_pilot_tone` with one ``frequency`` per
    channel), once for the whole record or, with ``track_bandwidth``,
    tracked over time.

    **Principle.**  With tone ``f_j`` carried only by stream ``j`` and a
    frequency-flat mixing ``r = J s`` (e.g.
    :func:`~commkit.impairments.apply_polarization_mixing`), the complex
    amplitude of tone ``f_j`` in received channel ``i`` is
    ``T[i, j] = J[i, j] · α_j``, with ``α_j`` the transmitted tone amplitude
    of stream ``j``.  The measured matrix of tone phasors is therefore
    ``T = J · diag(α)``, and ``W = pinv(T)`` unmixes::

        W r = diag(1/α) · J^{-1} J s = diag(1/α) · s,

    which recovers each stream up to a complex scale ``1/α_j`` (removed by
    ``correct_polarization(normalize=True)`` and the downstream CPR).  Each
    tone labels its stream, so output row ``j`` is always the stream of
    ``tone_frequencies[j]``: there is no permutation ambiguity, unlike after
    blind equalization (see :func:`commkit.recovery.resolve_channel_permutation`).

    **Static** (``track_bandwidth=None``).  Tones from ``add_pilot_tone`` sit
    on the FFT grid, so one DFT bin is the exact (orthogonal, maximum-
    likelihood) estimate of each tone phasor: a ``(C, N)·(N, K)`` product and
    one ``K x K`` inverse, far cheaper than a full FFT.  The SOP must be
    constant over the record: if it drifts (a long record, or a low symbol
    rate whose capture outlasts the SOP coherence time), the averaged
    phasors are biased and the one-shot inverse leaves crosstalk that grows
    with time.  For PMD (DGD), seed a butterfly equalizer (``cma``,
    ``block_lms``) with :attr:`PolarizationEstimate.matrix`.

    **Tracked** (``track_bandwidth`` in Hz).  Mixing receive channel ``i``
    down by ``f_j`` and low-pass filtering isolates that tone's slowly
    varying phasor::

        z_ij(n) = LPF{ r_i(n) · exp(-j2π f_j n / f_s) } ≈ J_ij(n) · α_j,

    because every other tone lands at ``f_k - f_j`` and the data band moves
    away from DC.  ``T(n)`` is inverted on a decimated grid, and
    :func:`correct_polarization` interpolates ``W(n)`` back to the full
    rate.  ``track_bandwidth``, the low-pass cutoff, is bounded on both
    sides: at least the SOP rotation rate, or ``W(n)`` lags the true
    ``J(n)``; at most the guard to the nearest other tone and to the data
    band (about half the smallest tone spacing), or those leak into the
    estimate.  A warning is logged when ``2·track_bandwidth`` plus the FIR
    transition reaches the nearest tone spacing.

    Parameters
    ----------
    tone_frequencies : sequence of float
        The ``K <= C`` distinct tone frequencies in Hz, one per stream, in the
        order of the transmitted streams.
    track_bandwidth : float, optional
        One-sided cutoff of the tracking low-pass in Hz, the polarization
        tracking bandwidth: set it a few times above the expected SOP
        rotation rate and well below the smallest tone spacing.  ``None``
        estimates one matrix for the record.
    num_taps : int, optional
        Tracked only: length of the tracking low-pass FIR.  Defaults to
        about ``3.3·f_s/track_bandwidth`` (the Hamming transition width that
        resolves ``track_bandwidth``), made odd and kept below ``N``.
    grid_step : int, optional
        Tracked only: spacing in samples of the grid on which ``T(n)`` is
        inverted.  Defaults to ``max(1, floor(f_s/(4·track_bandwidth)))``,
        four grid points per tracking period.
    refine_tones : bool, default True
        Refine each tone's frequency to a fraction of a bin with the
        ``SpectralPeak`` estimator (on the receive channel where the tone is
        strongest) first, which absorbs a residual carrier frequency offset
        that moved the tone off its bin.  ``False`` uses
        ``tone_frequencies`` exactly.
    search_band : float, optional
        Half-width in Hz of each tone's peak search with
        ``refine_tones=True``; defaults to ``4 * f_s / N`` (a few FFT bins).
        Keep it inside the guard between tone and data.


    Raises
    ------
    ValueError
        If ``tone_frequencies`` is empty, ``track_bandwidth`` is not
        positive, or ``num_taps`` or ``grid_step`` come without
        ``track_bandwidth``.

    See Also
    --------
    commkit.spectral.add_pilot_tone : Add the per-stream tones at the transmitter.
    """

    tone_frequencies: tuple[float, ...]
    track_bandwidth: float | None = None
    num_taps: int | None = None
    grid_step: int | None = None
    refine_tones: bool = True
    search_band: float | None = None

    def __post_init__(self) -> None:
        tones = tuple(float(f) for f in self.tone_frequencies)
        object.__setattr__(self, "tone_frequencies", tones)
        if not tones:
            raise ValueError(
                "PolarizationTones: tone_frequencies must contain at least one "
                "frequency."
            )
        if self.track_bandwidth is None and (
            self.num_taps is not None or self.grid_step is not None
        ):
            raise ValueError(
                "PolarizationTones: num_taps and grid_step shape the tracking "
                "filter; give track_bandwidth too."
            )
        if self.track_bandwidth is not None and not self.track_bandwidth > 0:
            raise ValueError(
                "PolarizationTones: track_bandwidth must be positive; got "
                f"{self.track_bandwidth}."
            )


#: Polarization estimators by method type.
PolarizationMethod = PolarizationTones


def estimate_polarization(
    samples: ArrayType | Signal,
    method: PolarizationMethod,
    *,
    sampling_rate: float | None = None,
) -> PolarizationEstimate:
    """Estimate the polarization unmixing of MIMO samples.

    Parameters
    ----------
    samples : array_like or Signal
        Received MIMO samples, ``(C, N)``.
    method : PolarizationTones
        The estimator and its settings.
    sampling_rate : float, optional
        Sampling rate ``f_s`` in Hz.  Taken from the Signal; required for
        array input.  A value that disagrees with the Signal raises.

    Returns
    -------
    PolarizationEstimate
        The subclass of the method, :class:`PolarizationTonesEstimate` for
        :class:`PolarizationTones`: one ``(K, C)`` matrix (static) or the
        matrices on the grid (tracked), complex128, with the edge of the
        tracking filter and the refined tone frequencies.

    Raises
    ------
    ValueError
        If ``samples`` is not 2-D, there are more tones than channels, or a
        tone frequency lies outside ``(-f_s/2, f_s/2)``.
    TypeError
        If ``method`` is not a polarization method.
    """
    signal_adapter = adapt_signal(samples, function_name="estimate_polarization()")
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    x, _, _ = dispatch(signal_adapter.array)
    return _estimate(x, method, sampling_rate)


def _estimate(
    x: ArrayType, method: PolarizationMethod, sampling_rate: float
) -> PolarizationEstimate:
    """Dispatch on the method's type (array input, rate resolved)."""
    estimator = _ESTIMATORS.get(type(method))
    if estimator is None:
        raise TypeError(
            "estimate_polarization(): method must be PolarizationTones, got "
            f"{type(method).__name__}."
        )
    return estimator(x, method, sampling_rate)


def _estimate_tones(
    samples: ArrayType, method: PolarizationTones, sampling_rate: float
) -> PolarizationEstimate:
    """The pilot-tone estimator of :class:`PolarizationTones`."""
    tone_frequencies = method.tone_frequencies
    track_bandwidth = method.track_bandwidth
    num_taps = method.num_taps
    grid_step = method.grid_step
    refine_tones = method.refine_tones
    search_band = method.search_band
    _, xp, _ = dispatch(samples)
    f_tones = _check_tone_inputs(
        samples, tone_frequencies, sampling_rate, "estimate_polarization"
    )
    C, N = samples.shape
    K = len(f_tones)

    # Every O(N) pass runs in the signal's working precision (complex64 unless
    # the caller supplied complex128): the tone phasors are accumulated
    # block-wise with complex128 partials (_tone_phasor_matrix), while the
    # precision-sensitive inverses stay complex128.
    xw = (
        samples
        if samples.dtype == xp.complex128
        else samples.astype(xp.complex64, copy=False)
    )
    df = sampling_rate / N
    if refine_tones:
        if search_band is None:
            search_band = 4.0 * df
        # Sub-bin refine each tone on the channel where it is strongest (one
        # batched FFT of the unique channels; single host transfer).
        T0 = _tone_phasor_matrix(xw, f_tones, sampling_rate)  # (C, K)
        f_used = _refine_tone_frequencies(xw, T0, f_tones, sampling_rate, search_band)
    else:
        f_used = f_tones

    if track_bandwidth is None:
        T = _tone_phasor_matrix(xw, f_used, sampling_rate)  # (C, K) complex128
        # pinv covers the over-determined C > K case and equals the inverse
        # when square.
        W = xp.linalg.pinv(T)  # (K, C)
        logger.info(
            "estimate_polarization (tones, static): f_tones=%s Hz, refine=%s "
            "[C=%d, K=%d, N=%d]",
            [f"{f:.3g}" for f in f_used],
            refine_tones,
            C,
            K,
            N,
        )
        return PolarizationTonesEstimate(
            W[None],
            xp.zeros(1, dtype=xp.float64),
            0,
            tone_frequencies=tuple(map(float, f_used)),
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

    if K > 1:
        sorted_f = sorted(f_tones)
        d_min = min(b - a for a, b in zip(sorted_f, sorted_f[1:]))
        transition = 3.3 * sampling_rate / num_taps
        if 2.0 * track_bandwidth + transition >= d_min:
            logger.warning(
                "estimate_polarization: tracking bandwidth "
                "(%.3g Hz, FIR transition %.3g Hz) approaches the nearest tone "
                "spacing %.3g Hz - neighboring tones may leak into the Jones "
                "estimate. Reduce track_bandwidth or widen the tone spacing.",
                track_bandwidth,
                transition,
                d_min,
            )

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

    logger.info(
        "estimate_polarization (tones, tracked): f_tones=%s Hz, refine=%s, "
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
    # A centered FIR averages zero padding into the num_taps//2 samples at
    # each end.
    return PolarizationTonesEstimate(
        Wg, grid_positions, num_taps // 2, tone_frequencies=tuple(map(float, f_used))
    )


def correct_polarization(
    samples: S,
    how: PolarizationEstimate | PolarizationMethod | ArrayType,
    *,
    sampling_rate: float | None = None,
    normalize: bool = True,
    trim_edges: bool = False,
) -> S:
    """Unmix polarizations: ``y[:, n] = W(n) · x[:, n]``.

    Parameters
    ----------
    samples : array_like or Signal
        Received MIMO samples, ``(C, N)``.  A :class:`Signal` returns a new
        demultiplexed :class:`Signal`.
    how : PolarizationEstimate, PolarizationTones or array_like
        An estimate of :func:`estimate_polarization` (or any
        :class:`PolarizationEstimate`), a method, which estimates first, or
        one ``(K, C)`` unmixing matrix.  A tracked estimate is interpolated
        linearly between its grid points.
    sampling_rate : float, optional
        Sampling rate in Hz, for a method ``how``.  Taken from the Signal;
        required for array input with a method.  A value that disagrees with
        the Signal raises.
    normalize : bool, default True
        Rescale each output row to the mean input power per channel, which
        removes the tone-amplitude scale of a tone estimate and keeps the
        library's power convention.  With ``trim_edges=True`` the power is
        measured over the retained samples.
    trim_edges : bool, default False
        Drop the ``how.edge`` samples at each end, where a tracked estimate
        averaged in the zero padding of its tracking filter and leaves
        residual crosstalk; the output is then ``(K, N - 2·edge)``.  The data
        itself is never filtered, so timing is unaffected.  Array input
        only: trimming would misalign a Signal's reference.

    Returns
    -------
    ndarray or Signal
        Demultiplexed streams, ``(K, N)``: one row per transmitted stream,
        row ``j`` the stream of the estimate's ``tone_frequencies[j]``.  Same
        complex dtype and device as the input.

    Raises
    ------
    ValueError
        If ``samples`` is not ``(C, N)`` with the ``C`` of ``how``, or
        ``trim_edges`` is set for a Signal.
    """
    signal_adapter = adapt_signal(samples, function_name="correct_polarization()")
    x, xp, _ = dispatch(signal_adapter.array)
    if isinstance(how, PolarizationEstimate):
        track = how
    elif type(how) in _ESTIMATORS:
        fs = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
        track = _estimate(x, cast(PolarizationMethod, how), fs)
    else:
        matrix = xp.asarray(how)
        if matrix.ndim != 2:
            raise ValueError(
                "correct_polarization(): how must be a PolarizationEstimate, a method "
                "or a (K, C) "
                f"matrix, got shape {matrix.shape}."
            )
        track = PolarizationEstimate(matrix[None], xp.zeros(1, dtype=xp.float64))
    if trim_edges and signal_adapter.signal is not None:
        raise ValueError(
            "correct_polarization(): trim_edges drops samples and would "
            "misalign the Signal's reference; use array input."
        )
    M = xp.asarray(track.matrix_grid)
    if x.ndim != 2 or x.shape[0] != M.shape[-1]:
        raise ValueError(
            f"correct_polarization(): samples must be (C, N) with C = "
            f"{M.shape[-1]} channels, got shape {x.shape}."
        )
    N = x.shape[1]
    if M.shape[0] == 1:
        # One matrix: a per-sample mat-vec with no long accumulation, in the
        # working precision (complex64 unless the input is complex128).
        xw = x if x.dtype == xp.complex128 else x.astype(xp.complex64, copy=False)
        demuxed = M[0].astype(xw.dtype) @ xw  # (K, N)
    else:
        demuxed = _apply_interpolated_matrix(x, M, track.grid_positions)
    guard = track.edge if trim_edges else 0
    valid = slice(guard, N - guard)
    demuxed = demuxed[:, valid]
    if normalize:
        x_kept = x[:, valid]
        # Single-pass BLAS reduction - no full-record |x|² temporary.
        p_in = xp.vdot(x_kept, x_kept).real / x_kept.size
        p_out = xp.mean(xp.abs(demuxed) ** 2, axis=-1, keepdims=True)  # (K, 1)
        demuxed = demuxed * xp.sqrt(p_in / xp.where(p_out > 0, p_out, 1.0))
    return signal_adapter.wrap_samples(demuxed.astype(x.dtype, copy=False))


_ESTIMATORS: dict[type, Callable[..., PolarizationEstimate]] = {
    PolarizationTones: _estimate_tones
}
