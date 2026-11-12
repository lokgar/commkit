"""Carrier frequency offset estimation and correction.

:func:`estimate_frequency_offset` measures the offset with the estimator its
method object chooses: :class:`MthPower` (blind, the spectral peak of the
M-th power), :class:`MengaliMorelli` (blind, multi-lag autocorrelation),
:class:`PilotSlope` (the phase slope of known symbols) or
:class:`SpectralPeak` (a CW pilot tone).  A ``block_size`` on any method but
``PilotSlope`` tracks a time-varying offset.  The estimate is a
:class:`FrequencyOffsetEstimate`, as the subclass holding the method's
diagnostics (:class:`MthPowerEstimate`, :class:`MengaliMorelliEstimate`,
:class:`PilotSlopeEstimate`, :class:`BlockwiseFrequencyEstimate`);
:func:`correct_frequency_offset` removes an estimate, an offset in Hz, or
what a method estimates.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _fft
from ._array import as_2d, broadcast_channels, restore_1d
from ._peaks import parabolic_peak_offset
from .backend import ArrayType, dispatch, to_device
from .core._signal_adapter import S, adapt_signal, require_integer_sps
from .core.signal import Signal
from .logger import logger
from .math import _derotate, _linear_trend_slope


def _get_numba_mm_bootstrap() -> Callable[..., np.ndarray]:
    """The Mengali-Morelli bootstrap kernel."""
    from . import _numba_frequency

    kernel: Callable[..., np.ndarray] = _numba_frequency.mm_bootstrap
    return kernel


__all__ = [
    "BlockwiseFrequencyEstimate",
    "FrequencyOffsetEstimate",
    "MengaliMorelli",
    "MengaliMorelliEstimate",
    "MthPower",
    "MthPowerEstimate",
    "PilotSlope",
    "PilotSlopeEstimate",
    "SpectralPeak",
    "correct_frequency_offset",
    "estimate_frequency_offset",
]


# -----------------------------------------------------------------------------
# METHOD OBJECTS
# -----------------------------------------------------------------------------
# One frozen object per estimation method.  ``block_size`` turns any of
# the blind methods into a blockwise tracker: every block of every channel is
# estimated in one batched call, and ``correct_frequency_offset`` interpolates
# the block estimates (PCHIP) into a phase trajectory.


def _check_blocks(block_size: int | None, overlap: float) -> None:
    if block_size is not None and block_size < 4:
        raise ValueError(f"block_size must be >= 4 samples, got {block_size}.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")


@dataclass(frozen=True)
class MthPower:
    """Blind M-th power spectral estimator.

    Raising the samples to the power ``M`` removes the modulation and leaves a
    tone at ``M·Δf``; its spectral peak is refined by sub-bin interpolation.
    Lock range ``±fs/(2M)``.

    Parameters
    ----------
    power : int, optional
        ``M``.  Defaults to the constellation's rotational symmetry (4 for
        QAM, M for M-PSK, 2 for bipolar PAM).
    search_range : (float, float), optional
        ``(f_min, f_max)`` in Hz to search, mapped to ``[M·f_min, M·f_max]``.
    nfft : int, optional
        FFT size; defaults to the next power of 2 of the record (or block).
    interpolation : {"jacobsen", "parabolic"}, default "jacobsen"
        Sub-bin interpolation: Jacobsen's complex three-bin estimator (unbiased
        for a rectangular window) or a parabola through the magnitudes.
    block_size : int, optional
        Estimate per block of this many samples (blockwise tracking).
    overlap : float, default 0.5
        Fractional overlap of consecutive blocks, in ``[0, 1)``.

    Notes
    -----
    For a ``(C, N)`` record the channel magnitude spectra are summed to find
    one shared peak bin, then each channel is refined at that bin.  With
    ``block_size`` every block and channel is searched on its own.
    """

    power: int | None = None
    search_range: tuple[float, float] | None = None
    nfft: int | None = None
    interpolation: str = "jacobsen"
    block_size: int | None = None
    overlap: float = 0.5

    def __post_init__(self) -> None:
        if self.power is not None and self.power < 1:
            raise ValueError(f"power must be >= 1, got {self.power}.")
        if self.interpolation not in ("jacobsen", "parabolic"):
            raise ValueError(
                f"interpolation must be 'jacobsen' or 'parabolic', got "
                f"{self.interpolation!r}."
            )
        _check_blocks(self.block_size, self.overlap)


@dataclass(frozen=True)
class MengaliMorelli:
    """Mengali-Morelli multi-lag autocorrelation estimator.

    Combines the autocorrelation phase at lags ``1 .. L`` with MVUE weights
    ``m²|R[m]|²`` after bootstrapping the phase from lag 1; Cramér-Rao
    efficient with a lock range of ``±fs/(2M)``.

    Parameters
    ----------
    power : int, optional
        ``M`` for blind pre-processing ``x^M``.  Defaults to the
        constellation's rotational symmetry, or 1 (a constant-envelope or
        already derotated signal) when there is no constellation.  For the
        data-aided form, pass ``x * conj(known)`` with ``power=1``.
    max_lag : int, optional
        ``L``; defaults to ``N // 4``, clamped to ``[1, N // 2]``.
    block_size : int, optional
        Estimate per block of this many samples (blockwise tracking).
    overlap : float, default 0.5
        Fractional overlap of consecutive blocks, in ``[0, 1)``.

    Notes
    -----
    The autocorrelation is computed on the samples' device.  The bootstrap
    over the lags is a sequential recursion (each lag is unwrapped against
    the estimate of the lags before it), run by a Numba kernel on the
    host: GPU input copies the ``(R, L)`` autocorrelation, one row per
    channel (and per block with ``block_size``), to the host once per call.
    """

    power: int | None = None
    max_lag: int | None = None
    block_size: int | None = None
    overlap: float = 0.5

    def __post_init__(self) -> None:
        if self.power is not None and self.power < 1:
            raise ValueError(f"power must be >= 1, got {self.power}.")
        if self.max_lag is not None and self.max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {self.max_lag}.")
        _check_blocks(self.block_size, self.overlap)


@dataclass(frozen=True, eq=False)
class PilotSlope:
    """Least-squares phase slope of known pilot symbols.

    The pilot phase ``angle(r · conj(s))`` is unwrapped and fitted with a
    line (optionally weighted by ``|r|²``); its slope is ``2π·Δf``.  Lock
    range ``±fs/(2·max_gap)`` for the largest gap between pilot indices.

    Parameters
    ----------
    indices : array_like of int
        Sample indices of the pilots, increasing, ``(P,)``.
    values : array_like
        Transmitted pilot symbols, ``(P,)`` shared or ``(C, P)`` per channel.
    snr_weighted : bool, default True
        Weight each pilot by its received power (WLSQ) instead of plain OLS.
    """

    indices: np.ndarray
    values: np.ndarray
    snr_weighted: bool = True

    def __post_init__(self) -> None:
        indices = np.array(to_device(self.indices, "cpu")).astype(np.intp)
        values = np.array(to_device(self.values, "cpu"))
        if indices.ndim != 1 or indices.size < 2:
            raise ValueError("indices must be 1-D with at least two pilots.")
        if np.any(np.diff(indices) <= 0):
            raise ValueError("indices must be strictly increasing.")
        if values.shape[-1] != indices.size or values.ndim not in (1, 2):
            raise ValueError(
                f"values must have shape (P,) or (C, P) with P={indices.size}, "
                f"got {values.shape}."
            )
        for name, arr in (("indices", indices), ("values", values)):
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    @classmethod
    def from_frame(
        cls,
        frame: Any,
        *,
        sps: int = 1,
        include_preamble: bool = True,
        snr_weighted: bool = True,
    ) -> "PilotSlope":
        """The pilots of a ``SingleCarrierFrame``.

        Parameters
        ----------
        frame : SingleCarrierFrame
            A frame with pilots.
        sps : int, default 1
            Samples per symbol of the record to estimate on: the pilot of
            symbol ``k`` is sample ``k * sps``, the symbol-centered phase of
            a matched-filtered, timing-corrected record.
        include_preamble : bool, default True
            Count from the frame start (the preamble's first symbol, as after
            ``correct_timing(mode="slice")``); ``False`` counts from the end
            of the preamble.
        snr_weighted : bool, default True
            As for the constructor.

        Returns
        -------
        PilotSlope

        Raises
        ------
        ValueError
            If the frame has no pilots.
        """
        sps = require_integer_sps(sps, "PilotSlope.from_frame()")
        if frame.pilot_symbols is None:
            raise ValueError("PilotSlope.from_frame(): the frame has no pilots.")
        indices = frame.pilot_indices(include_preamble)
        return cls(indices * sps, np.asarray(frame.pilot_symbols), snr_weighted)


@dataclass(frozen=True)
class SpectralPeak:
    """Frequency of the spectral peak: a CW pilot (bias) tone, no nonlinearity.

    Finds the spectral peak, optionally inside ``target_frequency ±
    search_band``, and refines it by a log-parabolic fit of the three bins
    around it.  No nonlinearity is applied, so the result is independent of
    the modulation.

    Parameters
    ----------
    target_frequency : float, optional
        Center of the search window in Hz; give it with ``search_band``.
    search_band : float, optional
        Half-width of the search window in Hz.
    block_size : int, optional
        Estimate per block of this many samples (blockwise tracking).
    overlap : float, default 0.5
        Fractional overlap of consecutive blocks, in ``[0, 1)``.
    """

    target_frequency: float | None = None
    search_band: float | None = None
    block_size: int | None = None
    overlap: float = 0.5

    def __post_init__(self) -> None:
        if (self.target_frequency is None) != (self.search_band is None):
            raise ValueError(
                "target_frequency and search_band must both be given or both omitted."
            )
        _check_blocks(self.block_size, self.overlap)


FrequencyMethod = MthPower | MengaliMorelli | PilotSlope | SpectralPeak


# -----------------------------------------------------------------------------
# ESTIMATE
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FrequencyOffsetEstimate:
    """Result of :func:`estimate_frequency_offset`: the result of
    ``SpectralPeak``, and the fields every method gives.

    The constructor takes the same fields.  The other methods return a
    subclass with their diagnostics: :class:`MthPowerEstimate`,
    :class:`MengaliMorelliEstimate`, :class:`PilotSlopeEstimate`, and
    :class:`BlockwiseFrequencyEstimate` for any method with a
    ``block_size``.  Per-channel fields follow the rank rule (0-d for
    ``(N,)`` input, ``(C,)`` for ``(C, N)``) and stay on the input's device.

    Attributes
    ----------
    value : ndarray
        Frequency offset in Hz (the mean of the block estimates for a
        blockwise method).
    weights : ndarray
        Per-channel reliability weights used by :meth:`combined` (peak
        magnitude, autocorrelation energy or pilot power).
    """

    value: ArrayType
    weights: ArrayType

    def combined(self) -> "FrequencyOffsetEstimate":
        """One estimate for all channels: the weighted mean over channels.

        For channels that share one laser (polarization diversity).  The
        result is 0-d and is applied to every channel by
        :func:`correct_frequency_offset`; the per-channel diagnostics are
        dropped (a blockwise estimate keeps its combined blocks).
        """
        _, xp, _ = dispatch(self.value)
        if self.value.ndim == 0:
            return self
        w = self.weights / xp.sum(self.weights)
        return FrequencyOffsetEstimate(
            value=xp.sum(w * self.value), weights=xp.sum(self.weights)
        )


@dataclass(frozen=True)
class MthPowerEstimate(FrequencyOffsetEstimate):
    """Result of :class:`MthPower`.

    The constructor takes the same fields: those of
    :class:`FrequencyOffsetEstimate`, then the attributes below.

    Attributes
    ----------
    power : int
        Exponent ``M`` used.
    spectrum : ndarray
        ``|FFT(x^M)|`` over the bins of the search range (all ``nfft`` bins
        without one), in ascending frequency, ``(K,)`` or ``(C, K)``.
    spectrum_frequencies : numpy.ndarray
        The frequencies of ``spectrum``'s bins in Hz, ``(K,)``, ascending and
        evenly spaced (divide by ``power`` for ``Δf``).
    """

    power: int
    spectrum: ArrayType
    spectrum_frequencies: np.ndarray


@dataclass(frozen=True)
class MengaliMorelliEstimate(FrequencyOffsetEstimate):
    """Result of :class:`MengaliMorelli`.

    The constructor takes the same fields: those of
    :class:`FrequencyOffsetEstimate`, then the attributes below.

    Attributes
    ----------
    power : int
        Exponent ``M`` used.
    autocorrelation : ndarray
        Unbiased autocorrelation at lags ``1 .. L``.
    """

    power: int
    autocorrelation: ArrayType


@dataclass(frozen=True)
class PilotSlopeEstimate(FrequencyOffsetEstimate):
    """Result of :class:`PilotSlope`.

    The constructor takes the same fields: those of
    :class:`FrequencyOffsetEstimate`, then the attributes below.

    Attributes
    ----------
    pilot_phase : ndarray
        Unwrapped pilot phase in radians, ``(P,)`` or ``(C, P)``.
    pilot_indices : numpy.ndarray
        The pilot sample indices.
    """

    pilot_phase: ArrayType
    pilot_indices: np.ndarray


@dataclass(frozen=True)
class BlockwiseFrequencyEstimate(FrequencyOffsetEstimate):
    """Result of a method with a ``block_size``: a time-varying offset.

    The constructor takes the same fields: those of
    :class:`FrequencyOffsetEstimate`, then the attributes below.
    :func:`correct_frequency_offset` interpolates the blocks.

    Attributes
    ----------
    block_centers : numpy.ndarray
        Block center sample indices, ``(B,)``.
    block_values : ndarray
        Estimate per block in Hz, ``(B,)`` or ``(C, B)``.
    """

    block_centers: np.ndarray
    block_values: ArrayType

    def combined(self) -> "BlockwiseFrequencyEstimate":
        """One estimate for all channels: the weighted mean over channels,
        of the value and of each block (``(B,)``).
        """
        _, xp, _ = dispatch(self.value)
        if self.value.ndim == 0:
            return self
        w = self.weights / xp.sum(self.weights)
        return BlockwiseFrequencyEstimate(
            value=xp.sum(w * self.value),
            weights=xp.sum(self.weights),
            block_centers=self.block_centers,
            block_values=xp.sum(w[:, None] * self.block_values, axis=0),
        )


# -----------------------------------------------------------------------------
# ESTIMATOR KERNELS (rows of a (R, N) array; R = channels or blocks)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rows:
    """Per-row values and weights (on device) plus the method's diagnostics."""

    values: ArrayType
    weights: ArrayType
    power: int | None = None
    spectrum: ArrayType | None = None
    spectrum_frequencies: np.ndarray | None = None
    autocorrelation: ArrayType | None = None
    pilot_phase: ArrayType | None = None
    pilot_indices: np.ndarray | None = None


def _resolve_power(power: int | None, constellation: Any, default: int | None) -> int:
    if power is not None:
        return int(power)
    if constellation is not None:
        m = int(constellation.rotational_symmetry)
        if m > 4:
            logger.debug(
                "M=%s-th power raises the noise variance by M²; prefer pilot-"
                "aided estimation for constellations with this symmetry.",
                m,
            )
        return m
    if default is None:
        raise ValueError(
            "The M-th power method needs power= or a constellation (from the "
            "Signal or constellation=)."
        )
    return default


def _mth_power(
    x: ArrayType,
    fs: float,
    method: MthPower,
    constellation: Any,
    shared_rows: bool,
) -> _Rows:
    """M-th power estimate per row; rows share one peak bin if ``shared_rows``."""
    xp = dispatch(x)[1]
    R, N = x.shape
    if N < 8:
        raise ValueError(
            f"Signal too short for spectral FOE (N={N}). Minimum 8 samples required."
        )
    M = _resolve_power(method.power, constellation, None)
    nfft = method.nfft or 1 << int(np.ceil(np.log2(N)))

    # In the samples' precision: complex128 changed no estimate by more than
    # 0.005 Hz (16-QAM, 2^16 symbols, 15 dB; the error itself is 726 Hz rms).
    x_m = x**M

    X_m = _fft.fft(x_m, n=nfft, axis=-1)  # (R, nfft)
    freqs_np = np.fft.fftfreq(nfft, d=1.0 / fs)  # reported on the host
    mag = xp.abs(X_m)
    if method.search_range is not None:
        tone_lo = M * min(method.search_range)
        tone_hi = M * max(method.search_range)
        if not np.any((freqs_np >= tone_lo) & (freqs_np <= tone_hi)):
            raise ValueError(
                f"search_range {method.search_range} Hz produces an empty search "
                f"window in the M={M} scaled spectrum."
            )
        freqs = xp.fft.fftfreq(nfft, d=1.0 / fs)
        mask = (freqs >= tone_lo) & (freqs <= tone_hi)
        mag = xp.where(mask[None, :], mag, xp.zeros_like(mag))

    if shared_rows:
        k_peak = xp.broadcast_to(xp.argmax(xp.sum(mag, axis=0)), (R,))
    else:
        k_peak = xp.argmax(mag, axis=-1)
    k_prev = (k_peak - 1) % nfft  # circular: correct at the Nyquist edge
    k_next = (k_peak + 1) % nfft
    rows = xp.arange(R)
    if method.interpolation == "jacobsen":
        X_km1, X_k0, X_kp1 = X_m[rows, k_prev], X_m[rows, k_peak], X_m[rows, k_next]
        d_vec = 2.0 * X_k0 - X_km1 - X_kp1
        ok = xp.abs(d_vec) > 1e-30
        safe_d = xp.where(ok, d_vec, xp.ones_like(d_vec))
        mu_raw = ((X_km1 - X_kp1) / safe_d).real
        mu = xp.clip(xp.where(ok, mu_raw, xp.zeros_like(mu_raw)), -0.5, 0.5)
    else:
        mu = parabolic_peak_offset(
            mag[rows, k_prev], mag[rows, k_peak], mag[rows, k_next], xp, log=False
        )
    # The peak bin's frequency, as fftfreq computes it: signed bin times fs/nfft.
    k_signed = xp.where(k_peak > (nfft - 1) // 2, k_peak - nfft, k_peak)
    f_bin = k_signed.astype(xp.float64) * (1.0 / (nfft * (1.0 / fs)))
    values = (f_bin + mu * (fs / nfft)) / M
    # The diagnostic spectrum in ascending frequency, cut to the search range
    # (the mask zeroed every other bin), gathered on the device.
    f_sorted = np.fft.fftshift(freqs_np)
    lo, hi = 0, nfft
    if method.search_range is not None:
        lo = int(np.searchsorted(f_sorted, tone_lo, side="left"))
        hi = int(np.searchsorted(f_sorted, tone_hi, side="right"))
    bins = (xp.arange(lo, hi) + (nfft + 1) // 2) % nfft  # fftshift order
    return _Rows(
        values=values.astype(xp.float64),
        weights=mag[rows, k_peak].astype(xp.float64),
        power=M,
        spectrum=mag[:, bins],
        spectrum_frequencies=f_sorted[lo:hi],
    )


def _mengali_morelli(
    x: ArrayType, fs: float, method: MengaliMorelli, constellation: Any
) -> _Rows:
    xp = dispatch(x)[1]
    R, N = x.shape
    M = _resolve_power(method.power, constellation, 1)
    y = x**M if M > 1 else x
    L = method.max_lag if method.max_lag is not None else N // 4
    L = max(1, min(L, N // 2))

    # Autocorrelation at all lags 1..L via Wiener-Khinchin (2 FFTs), linear
    # because of the zero-padding to nfft >= N + L.
    nfft_r = 1 << int(np.ceil(np.log2(N + L)))
    Y_r = _fft.fft(y, n=nfft_r, axis=-1)
    R_all = _fft.ifft(Y_r * xp.conj(Y_r), axis=-1)
    # Lags in the data's precision: a float64 array would promote the
    # quarter-record autocorrelation to complex128 (NEP 50).
    lags = xp.arange(1, L + 1, dtype=R_all.real.dtype)
    R_rows = R_all[:, 1 : L + 1] / (N - lags[None, :])  # unbiased, (R, L)

    # The bootstrap is a sequential scan over the lags; one transfer, then one
    # Numba call over all rows.
    R_np = np.asarray(to_device(R_rows, "cpu"))
    amp = np.abs(R_np)
    values = _get_numba_mm_bootstrap()(np.angle(R_np), amp, float(M), float(fs))
    weights = np.sum(amp**2, axis=-1)
    return _Rows(
        values=xp.asarray(values),
        weights=xp.asarray(weights),
        power=M,
        autocorrelation=R_rows,
    )


def _pilot_symbols(x: ArrayType, fs: float, method: PilotSlope) -> _Rows:
    xp = dispatch(x)[1]
    C, N = x.shape
    idx_np = method.indices
    if idx_np[-1] >= N:
        raise ValueError(
            f"Pilot index {int(idx_np[-1])} is outside the record (N={N})."
        )
    values = broadcast_channels(xp.asarray(method.values), C, xp, name="values")

    # Demodulated pilot phase: angle(r · conj(s)) = 2π·Δf·t + φ₀ + noise.
    r_pilots = x[:, xp.asarray(idx_np)]  # (C, P)
    phi = xp.angle(r_pilots * xp.conj(values))
    # Unwrap in float64: float32 rounding causes spurious slips.
    phi_u = xp.unwrap(phi.astype(xp.float64), axis=-1)
    t = xp.asarray(idx_np.astype(np.float64)) / fs  # (P,)

    pwr = xp.abs(r_pilots) ** 2  # (C, P)
    if method.snr_weighted:
        # WLSQ with per-channel weights |r|² (normalized to unit mean).
        v = pwr / (xp.mean(pwr, axis=-1, keepdims=True) + 1e-30)
        v_sum = xp.sum(v, axis=-1, keepdims=True)
        t_c = t[None, :] - xp.sum(v * t[None, :], axis=-1, keepdims=True) / v_sum
        phi_c = phi_u - xp.sum(v * phi_u, axis=-1, keepdims=True) / v_sum
        t_var = xp.sum(v * t_c**2, axis=-1)
        safe = xp.where(xp.abs(t_var) > 1e-30, t_var, xp.ones_like(t_var))
        slopes = xp.sum(v * phi_c * t_c, axis=-1) / safe
    else:
        slopes = _linear_trend_slope(phi_u, x=t, xp=xp)  # rad/s, centered OLS

    max_gap = int(np.max(np.diff(idx_np)))
    logger.debug(
        "FOE (pilots, %s): P=%s, max_gap=%s samples, lock range ±%.1f Hz",
        "WLSQ" if method.snr_weighted else "OLS",
        idx_np.size,
        max_gap,
        fs / (2 * max_gap),
    )
    return _Rows(
        values=slopes / (2.0 * np.pi),
        weights=xp.mean(pwr, axis=-1).astype(xp.float64),
        pilot_phase=phi_u,
        pilot_indices=idx_np,
    )


def _bias_tone(x: ArrayType, fs: float, method: SpectralPeak) -> _Rows:
    xp = dispatch(x)[1]
    R, N = x.shape
    if N < 4:
        raise ValueError(
            f"Segment too short for bias tone search (N={N}). Minimum 4 samples required."
        )
    nfft = 1 << int(np.ceil(np.log2(N)))
    mag = xp.abs(_fft.fft(x, n=nfft, axis=-1))  # (R, nfft)
    freqs = xp.fft.fftfreq(nfft, d=1.0 / fs)
    search = mag
    if method.target_frequency is not None:
        assert method.search_band is not None
        lo = method.target_frequency - abs(method.search_band)
        hi = method.target_frequency + abs(method.search_band)
        mask = (freqs >= lo) & (freqs <= hi)
        if not bool(xp.any(mask)):
            raise ValueError(
                f"target_frequency={method.target_frequency} ± search_band="
                f"{method.search_band} produces an empty search window for "
                f"fs={fs} Hz, nfft={nfft}."
            )
        search = xp.where(mask[None, :], mag, xp.zeros_like(mag))
    k = xp.argmax(search, axis=-1)
    rows = xp.arange(R)
    # The peak and its neighbours in float64, not the whole spectrum.
    left, peak, right = (
        mag[rows, j].astype(xp.float64) for j in ((k - 1) % nfft, k, (k + 1) % nfft)
    )
    delta = parabolic_peak_offset(left, peak, right, xp, log=True)
    values = freqs[k] + delta * (fs / nfft)
    return _Rows(values=values, weights=peak)


def _estimate_rows(
    x: ArrayType,
    fs: float,
    method: FrequencyMethod,
    constellation: Any,
    shared_rows: bool,
) -> _Rows:
    return _ROW_ESTIMATORS[type(method)](x, fs, method, constellation, shared_rows)


# Row estimator per method type, all called as
# ``(x, fs, method, constellation, shared_rows)``.
_ROW_ESTIMATORS: dict[type, Callable[..., _Rows]] = {
    MthPower: _mth_power,
    MengaliMorelli: lambda x, fs, m, c, shared: _mengali_morelli(x, fs, m, c),
    PilotSlope: lambda x, fs, m, c, shared: _pilot_symbols(x, fs, m),
    SpectralPeak: lambda x, fs, m, c, shared: _bias_tone(x, fs, m),
}


def _mth_power_result(
    values: Any, weights: Any, rows: _Rows, was_1d: bool
) -> MthPowerEstimate:
    assert rows.power is not None and rows.spectrum_frequencies is not None
    spectrum = restore_1d(was_1d, rows.spectrum)
    return MthPowerEstimate(
        values, weights, rows.power, spectrum, rows.spectrum_frequencies
    )


def _mengali_morelli_result(
    values: Any, weights: Any, rows: _Rows, was_1d: bool
) -> MengaliMorelliEstimate:
    assert rows.power is not None
    autocorrelation = restore_1d(was_1d, rows.autocorrelation)
    return MengaliMorelliEstimate(values, weights, rows.power, autocorrelation)


def _pilot_slope_result(
    values: Any, weights: Any, rows: _Rows, was_1d: bool
) -> PilotSlopeEstimate:
    assert rows.pilot_indices is not None
    pilot_phase = restore_1d(was_1d, rows.pilot_phase)
    return PilotSlopeEstimate(values, weights, pilot_phase, rows.pilot_indices)


def _spectral_peak_result(
    values: Any, weights: Any, rows: _Rows, was_1d: bool
) -> FrequencyOffsetEstimate:
    return FrequencyOffsetEstimate(values, weights)


#: The result of a static (whole-record) estimate, built from the method's rows.
_RESULTS: dict[type, Callable[[Any, Any, _Rows, bool], FrequencyOffsetEstimate]] = {
    MthPower: _mth_power_result,
    MengaliMorelli: _mengali_morelli_result,
    PilotSlope: _pilot_slope_result,
    SpectralPeak: _spectral_peak_result,
}


def _block_starts(n: int, block_size: int, overlap: float) -> tuple[list[int], int]:
    """Block start samples and block length (the whole record if too short)."""
    step = max(1, round(block_size * (1.0 - overlap)))
    starts = list(range(0, n - block_size + 1, step))
    if not starts:
        return [0], n
    return starts, block_size


# -----------------------------------------------------------------------------
# FREQUENCY OFFSET ESTIMATION AND CORRECTION (Signal-aware)
# -----------------------------------------------------------------------------


def estimate_frequency_offset(
    samples: ArrayType | Signal,
    method: FrequencyMethod,
    *,
    sampling_rate: float | None = None,
    constellation: Any = None,
) -> FrequencyOffsetEstimate:
    """Estimate the carrier frequency offset.

    Parameters
    ----------
    samples : array_like or Signal
        Complex samples, ``(N,)`` or ``(C, N)``.
    method : MthPower, MengaliMorelli, PilotSlope or SpectralPeak
        Estimation method; its ``block_size`` (blind methods) makes the
        estimate blockwise.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.
    constellation : Constellation, optional
        Sets the M-th power exponent when the method has no ``power``.
        Defaults to the Signal's ``constellation``.

    Returns
    -------
    FrequencyOffsetEstimate
        Offset per channel in Hz, as the subclass holding the method's
        diagnostics: :class:`MthPowerEstimate`,
        :class:`MengaliMorelliEstimate` or :class:`PilotSlopeEstimate`, and
        :class:`BlockwiseFrequencyEstimate` for any method with a
        ``block_size``; the base for ``SpectralPeak``.

    Examples
    --------
    >>> est = estimate_frequency_offset(sig, MthPower(search_range=(-1e9, 1e9)))
    >>> sig = correct_frequency_offset(sig, est)
    >>> sig = correct_frequency_offset(sig, MengaliMorelli(block_size=4096))
    """
    if not isinstance(method, FrequencyMethod):
        raise TypeError(
            "estimate_frequency_offset(): method must be MthPower, MengaliMorelli, "
            f"PilotSlope or SpectralPeak, got {type(method).__name__}."
        )
    signal_adapter = adapt_signal(samples, function_name="estimate_frequency_offset()")
    fs = float(signal_adapter.resolve_fact("sampling_rate", sampling_rate))
    constellation = signal_adapter.resolve_choice("constellation", constellation)
    return _estimate_frequency_offset(signal_adapter.array, method, fs, constellation)


def _estimate_frequency_offset(
    samples: ArrayType, method: FrequencyMethod, fs: float, constellation: Any
) -> FrequencyOffsetEstimate:
    """:func:`estimate_frequency_offset` on an array with resolved facts."""
    x, xp, _ = dispatch(samples)
    x, was_1d = as_2d(x, name="samples")
    C, N = x.shape

    block_size = getattr(method, "block_size", None)
    if block_size is None:
        rows = _estimate_rows(x, fs, method, constellation, shared_rows=True)
        values, weights = restore_1d(was_1d, rows.values, rows.weights)
        est: FrequencyOffsetEstimate = _RESULTS[type(method)](
            values, weights, rows, was_1d
        )
    else:
        assert not isinstance(method, PilotSlope)
        starts, length = _block_starts(N, block_size, method.overlap)
        B = len(starts)
        # All blocks of all channels in one batched call: rows (C * B, L).
        blocks = xp.stack([x[:, s : s + length] for s in starts], axis=1)
        rows = _estimate_rows(
            blocks.reshape(C * B, length), fs, method, constellation, shared_rows=False
        )
        block_values = rows.values.reshape(C, B)
        block_weights = rows.weights.reshape(C, B)
        values, weights, block_values = restore_1d(
            was_1d,
            xp.mean(block_values, axis=-1),
            xp.mean(block_weights, axis=-1),
            block_values,
        )
        est = BlockwiseFrequencyEstimate(
            value=values,
            weights=weights,
            block_centers=np.array([s + length / 2.0 for s in starts]),
            block_values=block_values,
        )

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "FOE (%s): %s Hz",
            type(method).__name__,
            np.round(np.atleast_1d(to_device(est.value, "cpu")), 2).tolist(),
        )
    return est


def correct_frequency_offset(
    samples: S,
    how: FrequencyOffsetEstimate | FrequencyMethod | float | ArrayType,
    *,
    sampling_rate: float | None = None,
) -> S:
    """Remove a carrier frequency offset by complex mixing.

    A constant offset is removed exactly, ``y[n] = x[n]·exp(-j2πΔf·n/fs)``
    with ``n = 0`` at the first sample and no bin quantization.  A blockwise
    estimate is interpolated with PCHIP between block centers (held constant
    outside them), integrated into a phase trajectory and removed.

    Parameters
    ----------
    samples : array_like or Signal
        Samples, ``(N,)`` or ``(C, N)``.
    how : FrequencyOffsetEstimate, method object, float or array_like
        An estimate, a method (estimated first, with the Signal's
        constellation), or the offset in Hz: scalar for all channels or
        ``(C,)`` per channel.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.

    Returns
    -------
    ndarray or Signal
        Corrected samples, same shape, dtype and device (complex for real
        input).

    Notes
    -----
    A constant correction restarts its phasor at ``n = 0``; correcting
    sub-blocks separately breaks phase continuity.
    """
    signal_adapter = adapt_signal(samples, function_name="correct_frequency_offset()")
    fs = float(signal_adapter.resolve_fact("sampling_rate", sampling_rate))
    x, xp, _ = dispatch(signal_adapter.array)
    if isinstance(how, FrequencyMethod):
        constellation = signal_adapter.resolve_choice("constellation", None)
        how = _estimate_frequency_offset(x, how, fs, constellation)

    if isinstance(how, BlockwiseFrequencyEstimate):
        result = _correct_blockwise(x, fs, how.block_centers, how.block_values, xp)
        return signal_adapter.wrap_samples(result)

    offset = how.value if isinstance(how, FrequencyOffsetEstimate) else how
    return signal_adapter.wrap_samples(_correct_static(x, fs, offset, xp))


def _correct_static(x: ArrayType, fs: float, offset: Any, xp: Any) -> ArrayType:
    offset_arr = xp.asarray(offset, dtype=xp.float64)
    per_channel = offset_arr.ndim >= 1 and offset_arr.size > 1
    n = x.shape[-1]
    # Phase step per sample, kept on the device (no sync for a device
    # scalar): (C, 1) per channel, 0-d otherwise.
    if per_channel:
        step = offset_arr.reshape(-1)[: x.shape[0], None] * (2.0 * np.pi / fs)
    else:
        step = offset_arr.reshape(-1)[0] * (2.0 * np.pi / fs)
    # float64 ramp, exact for any N; wrapped in float64, rotated in the
    # samples' precision.
    return _derotate(x, step * xp.arange(n, dtype=xp.float64))


def _correct_blockwise(
    x: ArrayType,
    fs: float,
    block_centers: np.ndarray,
    block_values: ArrayType,
    xp: Any,
) -> ArrayType:
    """Remove a blockwise frequency estimate, interpolated to every sample.

    The block values are interpolated (PCHIP, held constant outside the
    block centers) and integrated into a float64 phase trajectory on the
    samples' device: SciPy's interpolator on the CPU, CuPy's on the GPU.
    """
    x2, was_1d = as_2d(x, name="samples")
    C, N = x2.shape
    df = xp.asarray(block_values, dtype=xp.float64)
    df = df[None, :] if df.ndim == 1 else df  # (C_interp, B); 1 row is shared
    B = df.shape[-1]
    if B == 1:
        df_dense = xp.broadcast_to(df, (df.shape[0], N))
    else:
        if xp is np:
            from scipy.interpolate import PchipInterpolator
        else:
            from cupyx.scipy.interpolate import PchipInterpolator
        centers = xp.asarray(block_centers, dtype=xp.float64)
        n_grid = xp.arange(N, dtype=xp.float64)
        n_clamped = xp.clip(n_grid, float(block_centers[0]), float(block_centers[-1]))
        df_dense = PchipInterpolator(centers, df, axis=-1)(n_clamped)  # (C_interp, N)
    theta = (2.0 * np.pi / fs) * xp.cumsum(df_dense, axis=-1)
    if theta.shape[0] == 1 and C > 1:
        theta = xp.broadcast_to(theta, (C, N))

    corrected = _derotate(x2, theta)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "correct_frequency_offset (blockwise): C=%s, B=%s blocks, total phase "
            "drift=%.3f rad",
            C,
            B,
            float(theta[0, -1]),
        )
    return restore_1d(was_1d, corrected)


# -----------------------------------------------------------------------------
# SHARED TONE REFINEMENT (used by recovery and equalization)
# -----------------------------------------------------------------------------
