"""
Carrier frequency offset estimation and correction.

estimate_frequency_offset(x, method) measures the offset with one of the
method objects MthPower (blind spectral), MengaliMorelli (multi-lag
autocorrelation), PilotSymbols (pilot phase slope) or BiasTone (a CW
pilot tone); a block_size on the blind methods tracks a time-varying
offset.  correct_frequency_offset(x, how) removes an estimate, an offset
in Hz, or what a method estimates.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._array import as_2d, broadcast_channels, restore_1d
from .backend import ArrayType, dispatch, to_device
from .core._signal_adapter import S, adapt_signal
from .core.signal import Signal
from .helpers import linear_trend_slope
from .logger import logger
from .timing import _parabolic_peak_offset


def _modulation_power_m(modulation: str, order: int) -> int:
    """
    Return the exponent M for M-th power spectral methods.

    Parameters
    ----------
    modulation : str
        Modulation type string (case-insensitive).
    order : int
        Modulation order.

    Returns
    -------
    int
        M = ``order`` for PSK; M = 4 for QAM and other schemes.

    Notes
    -----
    M=4 is exact only for **square** QAM constellations (4, 16, 64, 256, ...)
    which have perfect 4-fold rotational symmetry.  For cross-QAM (32, 128,
    512-QAM) the 4th power leaves residual modulation spurs; a warning is
    emitted.  For PAM/ASK the M-th power law does not apply; prefer
    pilot-aided or data-aided estimators.
    """
    mod = modulation.lower()
    if "psk" in mod:
        if order > 4:
            logger.warning(
                "%s-PSK: M=%sth-power raises noise variance by M² - "
                "VV/FOE reliability degrades severely for order > 4. "
                "Prefer BPS or pilot-aided CPR for 8-PSK and higher.",
                order,
                order,
            )
        return order  # M-th power exactly removes M-PSK modulation

    if "qam" in mod:
        side = int(order**0.5)
        if side * side == order:
            return 4  # Square QAM: 4-fold rotational symmetry, 4th power is exact
        # Cross-QAM (32, 128, 512-QAM): 4-fold symmetry is only approximate
        logger.warning(
            "%s-QAM is not square - 4th-power FOE/CPR will have residual "
            "modulation spurs. Prefer pilot-aided or data-aided estimation.",
            order,
        )
        return 4

    # PAM, ASK, or unrecognised scheme
    logger.warning(
        "Modulation '%s' (order %s): M=4 is a heuristic. 4th-power methods "
        "are unreliable for non-QAM/PSK formats. Prefer pilot-aided or "
        "data-aided estimation.",
        modulation,
        order,
    )
    return 4


# Lazy-compiled Numba kernel for the M&M iterative bootstrap.
_NUMBA_MM: dict[str, Callable[..., float]] = {}


def _get_numba_mm_bootstrap() -> Callable[..., float]:
    """JIT-compile and cache the Numba M&M iterative bootstrap kernel.

    Returns
    -------
    callable
        Numba-compiled ``_mm_bootstrap_loop``.
    """
    if "mm" not in _NUMBA_MM:
        import numba

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _mm_bootstrap_loop(
            theta: np.ndarray, amp: np.ndarray, M_val: float, fs: float
        ) -> float:
            """Iterative Mengali-Morelli bootstrap compiled to machine code.

            Predicts each lag's phase from the running weighted frequency
            estimate accumulated from all previous lags, then folds it
            into the weighted sum.  Sequential data dependency prevents
            vectorisation; Numba removes Python-interpreter overhead.

            Parameters
            ----------
            theta : (L,) float64
                Wrapped phase of R[m] at each lag (output of np.angle).
            amp : (L,) float64
                Magnitude |R[m]| at each lag (output of np.abs).
            M_val : float64
                Modulation power (1 for data-aided/generic, order for PSK,
                4 for QAM).
            fs : float64
                Sampling rate in Hz.

            Returns
            -------
            float64
                Estimated frequency offset in Hz.
            """
            two_pi = 2.0 * np.pi
            L = len(theta)

            # Lag 1 initialisation (m=1, m²=1)
            Theta_0 = theta[0]
            w0 = amp[0] * amp[0]  # m² · |R|² at m=1
            f_hat = Theta_0 * fs / (two_pi * M_val)
            w_sum = w0 if w0 > 1e-30 else 1e-30
            wf_sum = w_sum * f_hat

            for m_idx in range(1, L):
                m_val = float(m_idx + 1)
                predicted = two_pi * f_hat * M_val * m_val / fs
                diff = predicted - theta[m_idx]
                # round() is the C math round - unboxed, no Python overhead
                correction = round(diff / two_pi)
                Theta_m = theta[m_idx] + two_pi * correction
                f_m = Theta_m * fs / (two_pi * m_val * M_val)
                w_m = m_val * m_val * amp[m_idx] * amp[m_idx]
                w_sum += w_m
                wf_sum += w_m * f_m
                f_hat = wf_sum / w_sum

            return float(f_hat)

        _NUMBA_MM["mm"] = _mm_bootstrap_loop

    kernel: Callable[..., float] = _NUMBA_MM["mm"]
    return kernel


__all__ = [
    "BiasTone",
    "FrequencyOffsetEstimate",
    "MengaliMorelli",
    "MthPower",
    "PilotSymbols",
    "correct_frequency_offset",
    "estimate_frequency_offset",
]


# -----------------------------------------------------------------------------
# METHOD OBJECTS
# -----------------------------------------------------------------------------
# One frozen object per estimation method (D16).  ``block_size`` turns any of
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
class PilotSymbols:
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


@dataclass(frozen=True)
class BiasTone:
    """Frequency of a CW pilot (bias) tone in the spectrum.

    Finds the spectral peak, optionally inside ``target_frequency ±
    search_band``, and refines it by a log-parabolic fit of the three bins
    around it.  No nonlinearity is applied, so the result is independent of
    the modulation.

    Parameters
    ----------
    target_frequency : float, optional
        Centre of the search window in Hz; give it with ``search_band``.
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


FrequencyMethod = MthPower | MengaliMorelli | PilotSymbols | BiasTone


# -----------------------------------------------------------------------------
# ESTIMATE
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FrequencyOffsetEstimate:
    """Result of :func:`estimate_frequency_offset`.

    Per-channel fields follow the rank rule (0-d for ``(N,)`` input, ``(C,)``
    for ``(C, N)``) and stay on the input's device.

    Attributes
    ----------
    value : array_like
        Frequency offset in Hz (the mean of the block estimates for a
        blockwise method).
    weights : array_like
        Per-channel reliability weights used by :meth:`combined` (peak
        magnitude, autocorrelation energy or pilot power).
    block_centers : numpy.ndarray or None
        Blockwise only: block centre sample indices, ``(B,)``.
    block_values : array_like or None
        Blockwise only: estimate per block in Hz, ``(B,)`` or ``(C, B)``.
    power : int or None
        Exponent ``M`` used (M-th power and Mengali-Morelli).
    spectrum : array_like or None
        M-th power: ``|FFT(x^M)|`` after the search mask, ``(nfft,)`` or
        ``(C, nfft)``.
    spectrum_frequencies : numpy.ndarray or None
        M-th power: FFT bin frequencies of ``spectrum`` in Hz (``fftfreq``
        order; divide by ``power`` for ``Δf``).
    autocorrelation : array_like or None
        Mengali-Morelli: unbiased autocorrelation at lags ``1 .. L``.
    pilot_phase : array_like or None
        Pilots: unwrapped pilot phase in radians, ``(P,)`` or ``(C, P)``.
    pilot_indices : numpy.ndarray or None
        Pilots: the pilot sample indices.
    """

    value: ArrayType
    weights: ArrayType
    block_centers: np.ndarray | None = None
    block_values: ArrayType | None = None
    power: int | None = None
    spectrum: ArrayType | None = None
    spectrum_frequencies: np.ndarray | None = None
    autocorrelation: ArrayType | None = None
    pilot_phase: ArrayType | None = None
    pilot_indices: np.ndarray | None = None

    def combined(self) -> "FrequencyOffsetEstimate":
        """One estimate for all channels: the weighted mean over channels.

        For channels that share one laser (polarization diversity).  The
        result is 0-d (block values ``(B,)``) and is applied to every channel
        by :func:`correct_frequency_offset`.
        """
        _, xp, _ = dispatch(self.value)
        if self.value.ndim == 0:
            return self
        w = self.weights / xp.sum(self.weights)
        block_values = (
            None
            if self.block_values is None
            else xp.sum(w[:, None] * self.block_values, axis=0)
        )
        return FrequencyOffsetEstimate(
            value=xp.sum(w * self.value),
            weights=xp.sum(self.weights),
            block_centers=self.block_centers,
            block_values=block_values,
            power=self.power,
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
            logger.warning(
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

    # Promote for numerical accuracy during the power computation.
    s_c = x.astype(xp.complex128 if x.dtype == xp.complex64 else x.dtype)
    x_m = s_c**M
    # Non-constant-envelope symbols leave an amplitude residual at DC.
    if constellation is not None and np.ptp(np.abs(constellation.points)) > 1e-9:
        x_m = x_m - xp.mean(x_m, axis=-1, keepdims=True)

    X_m = xp.fft.fft(x_m, n=nfft, axis=-1)  # (R, nfft)
    freqs_np = np.fft.fftfreq(nfft, d=1.0 / fs)
    mag = xp.abs(X_m)
    if method.search_range is not None:
        tone_lo = M * min(method.search_range)
        tone_hi = M * max(method.search_range)
        freqs = xp.asarray(freqs_np)
        mask = (freqs >= tone_lo) & (freqs <= tone_hi)
        if not bool(xp.any(mask)):
            raise ValueError(
                f"search_range {method.search_range} Hz produces an empty search "
                f"window in the M={M} scaled spectrum."
            )
        mag = xp.where(mask[None, :], mag, xp.zeros_like(mag))
    mag[:, 0] = 0.0  # Δf = 0 is degenerate; also any residual DC

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
        mu = _parabolic_peak_offset(
            mag[rows, k_prev], mag[rows, k_peak], mag[rows, k_next], xp, log=False
        )
    f_bin = xp.asarray(freqs_np)[k_peak]
    values = (f_bin + mu * (fs / nfft)) / M
    return _Rows(
        values=values.astype(xp.float64),
        weights=mag[rows, k_peak].astype(xp.float64),
        power=M,
        spectrum=mag,
        spectrum_frequencies=freqs_np,
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
    Y_r = xp.fft.fft(y, n=nfft_r, axis=-1)
    R_all = xp.fft.ifft(Y_r * xp.conj(Y_r), axis=-1)
    lags = xp.arange(1, L + 1, dtype=xp.float64)
    R_rows = R_all[:, 1 : L + 1] / (N - lags[None, :])  # unbiased, (R, L)

    # The bootstrap is a sequential scan; one transfer, then Numba per row.
    R_np = np.asarray(to_device(R_rows, "cpu"))
    kernel = _get_numba_mm_bootstrap()
    values = np.array(
        [
            float(kernel(np.angle(R_np[r]), np.abs(R_np[r]), float(M), float(fs)))
            for r in range(R)
        ]
    )
    weights = np.sum(np.abs(R_np) ** 2, axis=-1)
    return _Rows(
        values=xp.asarray(values),
        weights=xp.asarray(weights),
        power=M,
        autocorrelation=R_rows,
    )


def _pilot_symbols(x: ArrayType, fs: float, method: PilotSymbols) -> _Rows:
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
        slopes = linear_trend_slope(phi_u, x=t, xp=xp)  # rad/s, centred OLS

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


def _bias_tone(x: ArrayType, fs: float, method: BiasTone) -> _Rows:
    xp = dispatch(x)[1]
    R, N = x.shape
    if N < 4:
        raise ValueError(
            f"Segment too short for bias tone search (N={N}). Minimum 4 samples required."
        )
    nfft = 1 << int(np.ceil(np.log2(N)))
    mag = xp.abs(xp.fft.fft(x, n=nfft, axis=-1))  # (R, nfft)
    freqs_np = np.fft.fftfreq(nfft, d=1.0 / fs)
    search = mag
    if method.target_frequency is not None:
        assert method.search_band is not None
        lo = method.target_frequency - abs(method.search_band)
        hi = method.target_frequency + abs(method.search_band)
        freqs = xp.asarray(freqs_np)
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
    m64 = mag.astype(xp.float64)
    delta = _parabolic_peak_offset(
        m64[rows, (k - 1) % nfft], m64[rows, k], m64[rows, (k + 1) % nfft], xp, log=True
    )
    values = xp.asarray(freqs_np)[k] + delta * (fs / nfft)
    return _Rows(values=values, weights=m64[rows, k])


def _estimate_rows(
    x: ArrayType,
    fs: float,
    method: FrequencyMethod,
    constellation: Any,
    shared_rows: bool,
) -> _Rows:
    if isinstance(method, MthPower):
        return _mth_power(x, fs, method, constellation, shared_rows)
    if isinstance(method, MengaliMorelli):
        return _mengali_morelli(x, fs, method, constellation)
    if isinstance(method, PilotSymbols):
        return _pilot_symbols(x, fs, method)
    return _bias_tone(x, fs, method)


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
    """
    Estimate the carrier frequency offset.

    Parameters
    ----------
    samples : array_like or Signal
        Complex samples, ``(N,)`` or ``(C, N)``.
    method : MthPower, MengaliMorelli, PilotSymbols or BiasTone
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
        Offset per channel in Hz plus the method's diagnostics.

    Examples
    --------
    >>> est = estimate_frequency_offset(sig, MthPower(search_range=(-1e9, 1e9)))
    >>> sig = correct_frequency_offset(sig, est)
    >>> sig = correct_frequency_offset(sig, MengaliMorelli(block_size=4096))
    """
    if not isinstance(method, FrequencyMethod):
        raise TypeError(
            "estimate_frequency_offset(): method must be MthPower, MengaliMorelli, "
            f"PilotSymbols or BiasTone, got {type(method).__name__}."
        )
    signal_adapter = adapt_signal(samples, function_name="estimate_frequency_offset()")
    fs = float(signal_adapter.resolve_fact("sampling_rate", sampling_rate))
    constellation = signal_adapter.resolve_choice("constellation", constellation)
    x, xp, _ = dispatch(signal_adapter.array)
    x, was_1d = as_2d(x, name="samples")
    C, N = x.shape

    block_size = getattr(method, "block_size", None)
    if block_size is None:
        rows = _estimate_rows(x, fs, method, constellation, shared_rows=True)
        values, weights = restore_1d(was_1d, rows.values, rows.weights)
        spectrum = None if rows.spectrum is None else restore_1d(was_1d, rows.spectrum)
        autocorr = (
            None
            if rows.autocorrelation is None
            else restore_1d(was_1d, rows.autocorrelation)
        )
        phase = (
            None if rows.pilot_phase is None else restore_1d(was_1d, rows.pilot_phase)
        )
        est = FrequencyOffsetEstimate(
            value=values,
            weights=weights,
            power=rows.power,
            spectrum=spectrum,
            spectrum_frequencies=rows.spectrum_frequencies,
            autocorrelation=autocorr,
            pilot_phase=phase,
            pilot_indices=rows.pilot_indices,
        )
    else:
        assert not isinstance(method, PilotSymbols)
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
        est = FrequencyOffsetEstimate(
            value=values,
            weights=weights,
            block_centers=np.array([s + length / 2.0 for s in starts]),
            block_values=block_values,
            power=rows.power,
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
    """
    Remove a carrier frequency offset by complex mixing.

    A constant offset is removed exactly, ``y[n] = x[n]·exp(-j2πΔf·n/fs)``
    with ``n = 0`` at the first sample and no bin quantization.  A blockwise
    estimate is interpolated with PCHIP between block centres (held constant
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
    array_like or Signal
        Corrected samples, same shape, dtype and device (complex for real
        input).

    Notes
    -----
    A constant correction restarts its phasor at ``n = 0``; correcting
    sub-blocks separately breaks phase continuity.
    """
    signal_adapter = adapt_signal(samples, function_name="correct_frequency_offset()")
    fs = float(signal_adapter.resolve_fact("sampling_rate", sampling_rate))
    if isinstance(how, FrequencyMethod):
        how = estimate_frequency_offset(samples, how, sampling_rate=fs)
    x, xp, _ = dispatch(signal_adapter.array)

    if isinstance(how, FrequencyOffsetEstimate) and how.block_values is not None:
        assert how.block_centers is not None
        result = _correct_blockwise(x, fs, how.block_centers, how.block_values, xp)
        return signal_adapter.wrap_samples(result)

    offset = how.value if isinstance(how, FrequencyOffsetEstimate) else how
    return signal_adapter.wrap_samples(_correct_static(x, fs, offset, xp))


def _correct_static(x: ArrayType, fs: float, offset: Any, xp: Any) -> ArrayType:
    offset_arr = xp.asarray(offset)
    per_channel = offset_arr.ndim >= 1 and offset_arr.size > 1
    n = x.shape[-1]
    t = xp.arange(n, dtype=xp.float64) / fs  # float64: exact for any N

    if xp.iscomplexobj(x):
        target_dtype = x.dtype
    else:
        target_dtype = xp.complex64 if x.dtype == xp.float32 else xp.complex128

    # Wrap the phase to [-π, π] in float64 before casting to float32 for a
    # complex64 target: a large unwrapped ramp in float32 loses ~|φ|·2⁻²³ rad.
    dtype_real = xp.float32 if target_dtype == xp.complex64 else xp.float64
    two_pi = 2.0 * np.pi
    if per_channel:
        C = x.shape[0]
        offsets = xp.asarray(offset_arr.reshape(-1)[:C], dtype=xp.float64)
        phase = -2.0 * xp.pi * offsets[:, None] * t[None, :]  # (C, N)
    else:
        phase = -2.0 * xp.pi * float(offset_arr.reshape(-1)[0]) * t  # (N,)
    phase_w = (phase - xp.round(phase / two_pi) * two_pi).astype(dtype_real)
    mixer = xp.exp(1j * phase_w).astype(target_dtype)
    if not per_channel and x.ndim > 1:
        mixer = mixer.reshape((1,) * (x.ndim - 1) + (-1,))
    return x * mixer


def _correct_blockwise(
    x: ArrayType,
    fs: float,
    block_centers: np.ndarray,
    block_values: ArrayType,
    xp: Any,
) -> ArrayType:
    x2, was_1d = as_2d(x, name="samples")
    C, N = x2.shape
    df = np.asarray(to_device(block_values, "cpu"), dtype=np.float64)
    df = df[None, :] if df.ndim == 1 else df  # (C_interp, B); 1 row is shared
    B = df.shape[-1]
    n_grid = np.arange(N, dtype=np.float64)
    theta = np.empty((df.shape[0], N), dtype=np.float64)
    if B > 1:
        from scipy.interpolate import PchipInterpolator

        n_clamped = np.clip(n_grid, block_centers[0], block_centers[-1])
    for c in range(df.shape[0]):
        if B == 1:
            df_dense = np.full(N, df[c, 0])
        else:
            df_dense = PchipInterpolator(block_centers, df[c])(n_clamped)
        theta[c] = (2.0 * np.pi / fs) * np.cumsum(df_dense)
    if df.shape[0] == 1 and C > 1:
        theta = np.broadcast_to(theta, (C, N))

    phase = xp.asarray(theta)
    two_pi = 2.0 * np.pi
    phase_w = (phase - xp.round(phase / two_pi) * two_pi).astype(xp.float32)
    corrected = x2 * xp.exp(-1j * phase_w).astype(x2.dtype)
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


def _refine_tones_from_spectrum(
    X: ArrayType,
    sampling_rate: float,
    targets: Sequence[float],
    search_band: float,
    rows: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Batched log-parabolic tone refinement on a precomputed full-record FFT.

    Device-side counterpart of ``BiasTone`` for callers that already hold
    the record's spectrum: refines each ``targets[t]`` by an argmax search over
    ``targets[t] ± search_band`` in the spectrum row ``rows[t]``, followed by
    the same three-bin log-parabolic sub-bin fit.  No zero-padding is applied
    (the sub-bin fit compensates for the coarser bin grid), no extra FFTs are
    run, and all per-tone work stays on device - the refined frequencies come
    back in a **single** host transfer.

    Parameters
    ----------
    X : (R, N) array
        Full FFT (``xp.fft.fft`` bin order) of the rows to search.
    sampling_rate : float
        Sampling rate in Hz.
    targets : sequence of float
        Nominal tone frequencies to refine (length T).
    search_band : float
        Half-width of the per-tone search window in Hz.
    rows : sequence of int, optional
        Spectrum row for each target.  Defaults to ``range(T)`` (requires
        ``R == T``).

    Returns
    -------
    (T,) np.ndarray
        Refined frequencies in Hz (float64, on host).
    """
    X, xp, _ = dispatch(X)
    X, _ = as_2d(X, name="X")
    N = X.shape[-1]
    df = sampling_rate / N
    targets = [float(f) for f in targets]
    if rows is None:
        rows = list(range(len(targets)))
    neigh = xp.asarray([-1, 0, 1], dtype=xp.int64)

    refined = []
    for f_t, row in zip(targets, rows):
        # Signed candidate bins covering [f_t - band, f_t + band] on the N-grid.
        b_lo = max(int(np.ceil((f_t - abs(search_band)) / df)), -(N // 2))
        b_hi = min(int(np.floor((f_t + abs(search_band)) / df)), (N - 1) // 2)
        if b_hi < b_lo:
            raise ValueError(
                f"target_frequency={f_t} ± search_band={search_band} produces an "
                f"empty search window for fs={sampling_rate} Hz, N={N}."
            )
        cand = xp.arange(b_lo, b_hi + 1, dtype=xp.int64)  # signed bins
        spec_row = X[int(row)]
        mag2 = xp.abs(spec_row[cand % N]) ** 2
        kb = cand[xp.argmax(mag2)]  # 0-d signed peak bin, stays on device
        # Three-bin log-parabolic fit.
        mags = xp.abs(spec_row[(kb + neigh) % N]).astype(xp.float64)
        delta = _parabolic_peak_offset(mags[0], mags[1], mags[2], xp, log=True)
        refined.append((kb.astype(xp.float64) + delta) * df)

    return to_device(xp.stack(refined), "cpu")
