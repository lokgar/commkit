"""Linear filters: FIR and IIR design and application.

Design: :func:`fir_taps` (window method) and the IIR designs in second-order
sections, :func:`butterworth_sos`, :func:`chebyshev1_sos`,
:func:`chebyshev2_sos`, :func:`elliptic_sos` and :func:`bessel_sos` (host
NumPy arrays).  Application, on the device of the samples:
:func:`fir_filter` (FFT convolution, optionally circular),
:func:`ols_fir_filter` (overlap-save, for long filters), :func:`iir_filter`
and :func:`matched_filter`.
"""

from typing import Any

import numpy as np
import scipy

from .. import _fft
from .._array import as_2d, restore_1d
from .._circular import convolve_periodic
from .._overlap_save import ols_backward, ols_forward
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import S, adapt_signal
from ..logger import logger
from ..math import normalize
from .pulses import Pulse

__all__ = [
    "bessel_sos",
    "butterworth_sos",
    "chebyshev1_sos",
    "chebyshev2_sos",
    "elliptic_sos",
    "fir_filter",
    "fir_taps",
    "iir_filter",
    "matched_filter",
    "ols_fir_filter",
]


def fir_taps(
    *,
    sampling_rate: float,
    num_taps: int,
    cutoff: float | tuple[float, float],
    btype: str = "low",
    window: str = "hamming",
) -> ArrayType:
    """Design an FIR filter with the window method (``scipy.signal.firwin``).

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the signal in Hz.
    num_taps : int
        Number of filter coefficients.  For ``btype in {"high", "bandstop"}``
        this should typically be odd to avoid a zero at the Nyquist frequency.
    cutoff : float or (float, float)
        Cutoff frequency in Hz.  A scalar for ``btype in {"low", "high"}``;
        a ``(low, high)`` pair for ``btype in {"band", "bandstop"}``.
    btype : {"low", "high", "band", "bandstop"}, default "low"
        Filter shape, as for :func:`butterworth_sos` and the other IIR
        designs.
    window : str, default "hamming"
        Window, as accepted by ``scipy.signal.firwin`` (``"hamming"``,
        ``"blackman"``, ...).

    Returns
    -------
    ndarray
        Host ``float64`` taps with 0 dB passband gain, ``(num_taps,)``.
    """
    pass_zero = btype in ("low", "bandstop")
    logger.debug(
        "Designing FIR: btype=%s, cutoff=%s Hz, taps=%s.", btype, cutoff, num_taps
    )
    h = scipy.signal.firwin(
        num_taps, cutoff, window=window, fs=sampling_rate, pass_zero=pass_zero
    )
    return h


# -----------------------------------------------------------------------------
# FILTER DESIGN - IIR SOS GENERATORS (array-only)
# -----------------------------------------------------------------------------
# butterworth_sos, chebyshev1_sos, chebyshev2_sos, elliptic_sos, bessel_sos:
#   Classic IIR filter families in second-order-sections (SOS) form - the
#   IIR-design counterpart to fir_taps() above, one function per filter
#   *family* (since each is a genuinely different algorithm) rather than
#   per shape.  ``btype`` selects the shape (as in scipy's own
#   butter/cheby1/cheby2/ellip/bessel and fir_taps() above), matching how
#   these designs are parameterized in practice rather than splitting one
#   function per shape.
#
# Like the FIR tap generators, these build coefficients from parameters
# alone - no signal input, so none are Signal-aware.  Apply the resulting
# ``sos`` array to a signal with the generic iir_filter() below, the same
# way fir_filter() applies any of the FIR tap generators above.


def _iir_wn(
    cutoff: float | tuple[float, float], btype: str, nyq: float
) -> float | tuple[float, float]:
    """Normalize cutoff(s) in Hz to scipy's ``Wn`` convention (frac. of Nyquist)."""
    if btype in ("band", "bandstop"):
        low, high = cutoff  # type: ignore[misc]
        return (float(low) / nyq, float(high) / nyq)
    return float(cutoff) / nyq  # type: ignore[arg-type]


def butterworth_sos(
    *,
    sampling_rate: float,
    cutoff: float | tuple[float, float],
    order: int = 4,
    btype: str = "low",
) -> np.ndarray:
    """Design a Butterworth IIR filter in second-order-sections (SOS) form.

    Maximally flat passband and monotonic roll-off, the general-purpose IIR
    design.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the signal in Hz.
    cutoff : float or (float, float)
        Cutoff frequency in Hz.  A scalar for ``btype in {"low", "high"}``;
        a ``(low, high)`` pair for ``btype in {"band", "bandstop"}``.
    order : int, default 4
        Filter order.
    btype : {"low", "high", "band", "bandstop"}, default "low"
        Filter shape.

    Returns
    -------
    ndarray
        SOS coefficients, ``(n_sections, 6)``.
    """
    nyq = 0.5 * float(sampling_rate)
    Wn = _iir_wn(cutoff, btype, nyq)
    logger.debug(
        "Designing Butterworth SOS: btype=%s, cutoff=%s Hz, order=%s.",
        btype,
        cutoff,
        order,
    )
    return np.asarray(scipy.signal.butter(order, Wn, btype=btype, output="sos"))


def chebyshev1_sos(
    *,
    sampling_rate: float,
    cutoff: float | tuple[float, float],
    order: int = 4,
    btype: str = "low",
    ripple: float = 1.0,
) -> np.ndarray:
    """Design a Chebyshev Type I IIR filter in SOS form.

    Sharper roll-off than Butterworth for the same order, at the cost of
    passband ripple.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the signal in Hz.
    cutoff : float or (float, float)
        Cutoff frequency in Hz, see :func:`butterworth_sos`.
    order : int, default 4
        Filter order.
    btype : {"low", "high", "band", "bandstop"}, default "low"
        Filter shape.
    ripple : float, default 1.0
        Maximum passband ripple, in dB.

    Returns
    -------
    ndarray
        SOS coefficients, ``(n_sections, 6)``.
    """
    nyq = 0.5 * float(sampling_rate)
    Wn = _iir_wn(cutoff, btype, nyq)
    logger.debug(
        "Designing Chebyshev-I SOS: btype=%s, cutoff=%s Hz, order=%s, ripple=%s dB.",
        btype,
        cutoff,
        order,
        ripple,
    )
    return np.asarray(scipy.signal.cheby1(order, ripple, Wn, btype=btype, output="sos"))


def chebyshev2_sos(
    *,
    sampling_rate: float,
    cutoff: float | tuple[float, float],
    order: int = 4,
    btype: str = "low",
    attenuation: float = 40.0,
) -> np.ndarray:
    """Design a Chebyshev Type II (inverse Chebyshev) IIR filter in SOS form.

    Monotonic passband and equiripple stopband: Type I's passband ripple
    moves to the stopband.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the signal in Hz.
    cutoff : float or (float, float)
        Cutoff frequency in Hz, see :func:`butterworth_sos`.
    order : int, default 4
        Filter order.
    btype : {"low", "high", "band", "bandstop"}, default "low"
        Filter shape.
    attenuation : float, default 40.0
        Minimum stopband attenuation, in dB.

    Returns
    -------
    ndarray
        SOS coefficients, ``(n_sections, 6)``.
    """
    nyq = 0.5 * float(sampling_rate)
    Wn = _iir_wn(cutoff, btype, nyq)
    logger.debug(
        "Designing Chebyshev-II SOS: btype=%s, cutoff=%s Hz, order=%s, attenuation=%s dB.",
        btype,
        cutoff,
        order,
        attenuation,
    )
    return np.asarray(
        scipy.signal.cheby2(order, attenuation, Wn, btype=btype, output="sos")
    )


def elliptic_sos(
    *,
    sampling_rate: float,
    cutoff: float | tuple[float, float],
    order: int = 4,
    btype: str = "low",
    ripple: float = 1.0,
    attenuation: float = 40.0,
) -> np.ndarray:
    """Design an Elliptic (Cauer) IIR filter in SOS form.

    Sharpest roll-off per order of the classic families, at the cost of
    ripple in both passband and stopband.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the signal in Hz.
    cutoff : float or (float, float)
        Cutoff frequency in Hz, see :func:`butterworth_sos`.
    order : int, default 4
        Filter order.
    btype : {"low", "high", "band", "bandstop"}, default "low"
        Filter shape.
    ripple : float, default 1.0
        Maximum passband ripple, in dB.
    attenuation : float, default 40.0
        Minimum stopband attenuation, in dB.

    Returns
    -------
    ndarray
        SOS coefficients, ``(n_sections, 6)``.
    """
    nyq = 0.5 * float(sampling_rate)
    Wn = _iir_wn(cutoff, btype, nyq)
    logger.debug(
        "Designing Elliptic SOS: btype=%s, cutoff=%s Hz, order=%s, ripple=%s dB, "
        "attenuation=%s dB.",
        btype,
        cutoff,
        order,
        ripple,
        attenuation,
    )
    return np.asarray(
        scipy.signal.ellip(order, ripple, attenuation, Wn, btype=btype, output="sos")
    )


def bessel_sos(
    *,
    sampling_rate: float,
    cutoff: float | tuple[float, float],
    order: int = 4,
    btype: str = "low",
    norm: str = "phase",
) -> np.ndarray:
    """Design a Bessel/Thomson IIR filter in SOS form.

    Maximally flat group delay (nearly linear phase in the passband) instead
    of a sharp magnitude roll-off, for when the waveform shape matters more
    than stopband rejection.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the signal in Hz.
    cutoff : float or (float, float)
        Cutoff frequency in Hz, see :func:`butterworth_sos`.
    order : int, default 4
        Filter order.
    btype : {"low", "high", "band", "bandstop"}, default "low"
        Filter shape.
    norm : {"phase", "delay", "mag"}, default "phase"
        Critical frequency normalization, passed to ``scipy.signal.bessel``.

    Returns
    -------
    ndarray
        SOS coefficients, ``(n_sections, 6)``.
    """
    nyq = 0.5 * float(sampling_rate)
    Wn = _iir_wn(cutoff, btype, nyq)
    logger.debug(
        "Designing Bessel SOS: btype=%s, cutoff=%s Hz, order=%s, norm=%s.",
        btype,
        cutoff,
        order,
        norm,
    )
    return np.asarray(
        scipy.signal.bessel(order, Wn, btype=btype, output="sos", norm=norm)
    )


# -----------------------------------------------------------------------------
# FILTERING OPERATIONS (Signal-aware)
# -----------------------------------------------------------------------------
# ols_fir_filter: Public OLS FIR convolution (long-tap / memory-bounded)
# fir_filter: Generic FIR filtering operation (short-to-medium taps) - applies
#   any of the FIR tap generators above.
# matched_filter: Apply matched filter (time-reversed conjugate of pulse shape)
# iir_filter: Generic IIR filtering operation (SOS form, causal or zero-phase)
#   - applies any of the IIR SOS generators above, the IIR sibling of
#   fir_filter().
#
# shape_pulse (TX symbol -> waveform synthesis) lives in core/generation.py,
# not here: it is a signal-construction primitive, not a transform on an
# existing Signal's samples.


def ols_fir_filter(
    samples: S,
    taps: ArrayType,
    *,
    fft_size: int | None = None,
    center: bool = True,
) -> S:
    """Apply an FIR filter by overlap-save, for long filters and long records.

    Overlap-save (OLS) processes the record in FFT blocks of fixed size, so
    memory stays bounded for long impulse responses (chromatic dispersion,
    group-delay equalizers) where one FFT of the whole record would exhaust
    GPU memory.  For short filters on moderate records, :func:`fir_filter` is
    as fast and simpler.

    Parameters
    ----------
    samples : array_like or Signal
        Samples, ``(N,)`` or ``(C, N)``.  A Signal gives a new Signal.
    taps : array_like
        FIR coefficients, ``(L,)``.
    fft_size : int, optional
        FFT block size, a power of 2.  Defaults to
        ``max(1024, next_power_of_2(4·L))``, so the guard of a quarter block
        is at least ``L`` samples long.
    center : bool, default True
        Align the output as :func:`fir_filter` does (SciPy's
        ``mode='same'``, centered on tap ``L // 2``):
        ``y[n] = sum_k x[n + L//2 - k]·taps[k]``, which removes the group
        delay of a symmetric filter before symbol sampling.  With ``False``
        the output is the causal convolution ``y[n] = sum_k x[n-k]·taps[k]``
        (``numpy.convolve(x, taps, mode='full')[:N]``), for a raw causal
        response such as a step response, or when the caller aligns the
        filter itself.

    Returns
    -------
    ndarray or Signal
        Filtered samples, same shape and dtype as ``samples`` (real for real
        samples and taps).

    Raises
    ------
    ValueError
        If ``fft_size`` is not a power of 2 or ``fft_size // 4 < len(taps)``.

    Notes
    -----
    A symmetric guard of ``fft_size // 4`` samples is discarded from each
    block edge, so ``fft_size // 4 >= len(taps)`` must hold.

    With ``center=True`` the input is padded at the end by ``L // 2`` zeros
    and as many leading output samples are dropped, which costs at most one
    extra block.
    """
    signal_adapter = adapt_signal(samples, function_name="ols_fir_filter()")
    x = signal_adapter.array

    x, xp, _ = dispatch(x)
    taps = xp.asarray(taps)
    is_real = not xp.iscomplexobj(x) and not xp.iscomplexobj(taps)
    out_dtype = x.dtype  # capture before any reshape

    # Signal drives precision: cast taps to match signal so float64 tap
    # generators do not silently upcast complex64 signals via FFT multiply.
    target_tap_dtype = x.real.dtype if not xp.iscomplexobj(taps) else x.dtype
    if taps.dtype != target_tap_dtype:
        taps = taps.astype(target_tap_dtype)

    L = len(taps)
    half = L // 2

    x, was_1d = as_2d(x, name="samples")

    N = x.shape[-1]

    N_fft = fft_size
    if N_fft is None:
        N_fft = max(1024, 1 << (max(1, 4 * L) - 1).bit_length())
    elif N_fft & (N_fft - 1) or N_fft // 4 < L:
        raise ValueError(
            f"fft_size must be a power of 2 with fft_size // 4 >= len(taps) "
            f"({L}), got {fft_size}."
        )

    logger.debug(
        "ols_fir_filter: L=%s, N=%s, N_fft=%s, num_ch=%s, center=%s",
        L,
        N,
        N_fft,
        x.shape[0],
        center,
    )

    H = _fft.fft(taps, n=N_fft)  # frequency response of the filter

    if center:
        # Post-pad by half so the OLS can compute full_conv[half : half+N].
        # This matches scipy's mode='same' (center-aligned, group-delay compensated),
        # which is required for correct eye-opening after pulse-shaped filtering.
        samples_ext = xp.pad(x, ((0, 0), (0, half)))
        Y, meta = ols_forward(samples_ext, N_fft)
        X_hat_f = Y * H
        out_ext = ols_backward(X_hat_f, meta)  # shape: (num_ch, N + half)
        out = out_ext[:, half:]  # trim leading half -> shape: (num_ch, N)
    else:
        Y, meta = ols_forward(x, N_fft)
        X_hat_f = Y * H
        out = ols_backward(X_hat_f, meta)

    if is_real:
        out = out.real  # strip IFFT imaginary noise for real inputs
    elif out.dtype != out_dtype:
        out = out.astype(
            out_dtype
        )  # guard complex inputs (e.g. complex64 -> complex128)
    return signal_adapter.wrap_samples(restore_1d(was_1d, out))


def fir_filter(samples: S, taps: ArrayType, *, periodic: bool = False) -> S:
    """Apply an FIR filter along the time (last) axis.

    FFT convolution, ``mode="same"``: the output is centered on tap
    ``len(taps) // 2``, so a symmetric filter adds no delay.  With
    ``periodic=True`` the convolution is circular; away from the record ends
    both give the same samples.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` returns a
        new filtered :class:`Signal`.
    taps : array_like
        Filter coefficients (impulse response), ``(L,)``.  Cast to the
        precision of ``samples``.
    periodic : bool, default False
        Treat ``samples`` as one period of a periodic sequence (circular
        convolution): the output loops seamlessly, as on a streaming AWG,
        and has no filter transients at its ends.  Any ``L`` works.

    Returns
    -------
    ndarray or Signal
        Filtered samples, same shape and dtype as ``samples``.
    """
    signal_adapter = adapt_signal(samples, function_name="fir_filter()")
    x, xp, sp = dispatch(signal_adapter.array)
    taps = xp.asarray(taps)
    if taps.ndim != 1:
        raise ValueError(f"taps must be 1-D, got shape {taps.shape}.")
    logger.debug("Applying FIR filter via convolution (%s taps).", taps.size)

    # Signal drives precision: cast taps to match signal dtype so numpy/scipy
    # type-promotion rules do not silently upcast float32/complex64 signals.
    target_tap_dtype = x.real.dtype if not xp.iscomplexobj(taps) else x.dtype
    if taps.dtype != target_tap_dtype:
        taps = taps.astype(target_tap_dtype)

    if periodic:
        return signal_adapter.wrap_samples(convolve_periodic(x, taps))
    taps_nd = taps.reshape((1,) * (x.ndim - 1) + (-1,))
    result = sp.signal.convolve(x, taps_nd, mode="same", method="fft")

    # Belt-and-suspenders: scipy may still promote internally (version-dependent)
    if result.dtype != x.dtype:
        result = result.astype(x.dtype)
    return signal_adapter.wrap_samples(result)


def matched_filter(
    samples: S,
    *,
    pulse: Pulse | ArrayType | None = None,
    taps_normalization: str = "unit_energy",
) -> S:
    """Matched filter: convolve with the time-reversed conjugate of the pulse.

    Maximizes the SNR at the symbol instants in AWGN.

    Parameters
    ----------
    samples : array_like or Signal
        Received samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` returns
        a new filtered :class:`Signal`.
    pulse : Pulse or array_like, optional
        Transmit pulse, as a pulse object (``RRC(0.35)``) or its taps.
        Defaults to the Signal's ``pulse``; required for array input.  A
        pulse object needs the samples per symbol, so array input takes taps.
    taps_normalization : {"unit_energy", "unity_gain"}, default "unit_energy"
        Normalization of the matched-filter taps: unit energy (``Σ|h|² = 1``)
        or unit DC gain (``Σh = 1``).

    Returns
    -------
    ndarray or Signal
        Filtered samples, same shape and dtype as ``samples``.

    Raises
    ------
    ValueError
        If there is no pulse, or a pulse object is given with array input.
    """
    signal_adapter = adapt_signal(samples, function_name="matched_filter()")
    pulse = signal_adapter.resolve_choice("pulse", pulse)
    if pulse is None:
        raise ValueError(
            "matched_filter() needs a pulse: pass pulse= (a Pulse or taps) or a "
            "Signal that has one."
        )
    if isinstance(pulse, Pulse):
        sig = signal_adapter.signal
        if sig is None:
            raise ValueError(
                "matched_filter(): a Pulse needs the samples per symbol; pass "
                "pulse.taps(sps) for array input."
            )
        pulse = pulse.taps(sig.sps)
    if taps_normalization not in ("unit_energy", "unity_gain"):
        raise ValueError(
            f"Unknown taps_normalization: {taps_normalization!r}. "
            "Use 'unity_gain' or 'unit_energy'."
        )

    x, xp, _ = dispatch(signal_adapter.array)
    pulse_taps = xp.asarray(pulse)
    logger.debug("Applying matched filter (%s taps).", pulse_taps.size)
    matched_taps = normalize(xp.conj(pulse_taps[::-1]), mode=taps_normalization)
    return signal_adapter.wrap_samples(fir_filter(x, matched_taps))


_NUMBA_SOS: dict[str, Any] = {}


def _get_numba_sos() -> Any:
    """JIT-compile and cache the Numba second-order-sections kernel.

    Returns
    -------
    callable
        Numba-compiled ``_sos_loop(sos, x, zi, y)``: filters each row of
        ``x`` (``(C, N)``) through the sections ``sos`` (``(S, 6)``,
        float64) in transposed direct form II, the form of
        ``scipy.signal.sosfilt``.  The state ``zi`` (``(C, S, 2)``, float64
        or complex128) is updated in place and sets the working precision;
        ``y`` takes the output in the dtype of ``x``.
    """
    if "sos" not in _NUMBA_SOS:
        import numba

        # No fastmath: without reassociation, double-precision input gives
        # SciPy's sosfilt result bit for bit.
        @numba.njit(cache=True, nogil=True)
        def _sos_loop(sos: Any, x: Any, zi: Any, y: Any) -> None:
            C, N = x.shape
            S = sos.shape[0]
            for c in range(C):
                for n in range(N):
                    # Promote the sample to the state's precision (float64
                    # or complex128): adding the zero of a state element.
                    v = zi[c, 0, 0] * 0.0 + x[c, n]
                    for k in range(S):
                        out = sos[k, 0] * v + zi[c, k, 0]
                        zi[c, k, 0] = sos[k, 1] * v - sos[k, 4] * out + zi[c, k, 1]
                        zi[c, k, 1] = sos[k, 2] * v - sos[k, 5] * out
                        v = out
                    y[c, n] = v

        _NUMBA_SOS["sos"] = _sos_loop

    return _NUMBA_SOS["sos"]


def _sosfilt(sos: np.ndarray, x: np.ndarray, zi: np.ndarray) -> np.ndarray:
    """Causal SOS filter of host ``(C, N)`` data from the initial state ``zi``.

    ``zi`` is ``(C, S, 2)`` in float64 or complex128 and is not modified.
    Returns ``(C, N)`` in the dtype of ``x``.
    """
    y = np.empty_like(x)
    _get_numba_sos()(sos, np.ascontiguousarray(x), zi.copy(), y)
    return y


def _zero_phase_padding(sos: np.ndarray, num_samples: int) -> int:
    """Samples of odd extension at each end of a zero-phase SOS filter.

    SciPy's default ``padlen`` of ``sosfiltfilt``: ``3·ntaps``, with
    ``ntaps = 2·n_sections + 1`` less the sections without a ``b2`` or
    ``a2`` term.

    Raises
    ------
    ValueError
        If the record is not longer than the padding.
    """
    ntaps = 2 * int(sos.shape[0]) + 1
    ntaps -= min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    edge = 3 * ntaps
    if num_samples <= edge:
        raise ValueError(
            f"iir_filter(): samples must be longer than the zero-phase padding "
            f"of {edge} samples for these sections, got {num_samples}."
        )
    return edge


def _sosfiltfilt(sos: np.ndarray, x: np.ndarray, state_dtype: type) -> np.ndarray:
    """Zero-phase SOS filter of host ``(C, N)`` data, as ``scipy.signal.sosfiltfilt``.

    Odd extension by ``3·ntaps`` samples at each end, then a forward and a
    backward pass, each started from the steady state ``sosfilt_zi`` scaled
    by its first sample (SciPy's defaults ``padtype="odd"`` and
    ``padlen=None``).

    Raises
    ------
    ValueError
        If the record is not longer than the extension.
    """
    edge = _zero_phase_padding(sos, x.shape[-1])
    # Odd extension in the data's precision (2·x[0] - x[k] with a Python 2).
    left = 2 * x[:, :1] - x[:, edge:0:-1]
    right = 2 * x[:, -1:] - x[:, -2 : -(edge + 2) : -1]
    ext = np.concatenate([left, x, right], axis=-1)
    zi = scipy.signal.sosfilt_zi(sos).astype(state_dtype)  # (S, 2), small
    y = _sosfilt(sos, ext, zi[None] * ext[:, :1, None].astype(state_dtype))
    y = np.ascontiguousarray(y[:, ::-1])
    y = _sosfilt(sos, y, zi[None] * y[:, :1, None].astype(state_dtype))
    return y[:, ::-1][:, edge:-edge]


def iir_filter(
    samples: S,
    sos: ArrayType,
    *,
    zero_phase: bool = True,
) -> S:
    """Apply an IIR filter in second-order sections (SOS) along the time axis.

    The IIR counterpart of :func:`fir_filter`.  It applies sections designed
    separately, by :func:`butterworth_sos`, :func:`chebyshev1_sos`,
    :func:`chebyshev2_sos`, :func:`elliptic_sos`, :func:`bessel_sos` or any
    other SOS design.

    Parameters
    ----------
    samples : array_like or Signal
        Samples, ``(N,)`` or ``(C, N)``.  A Signal gives a new Signal.
    sos : array_like
        Second-order sections, ``(n_sections, 6)``, with ``sos[:, 3] == 1``.
    zero_phase : bool, default True
        Filter forward and backward, as ``scipy.signal.sosfiltfilt``: no
        phase distortion or group delay, but non-causal, so the whole record
        is needed.  With ``False`` the filter runs causally, as
        ``scipy.signal.sosfilt``, and its frequency-dependent group delay
        stays in the output, as in a real-time receiver.

    Returns
    -------
    ndarray or Signal
        Filtered samples, same shape, dtype and device as ``samples``.

    Raises
    ------
    ValueError
        If ``sos`` is not ``(n_sections, 6)`` with ``sos[:, 3] == 1``, or,
        with ``zero_phase=True``, the record is not longer than the
        zero-phase padding: ``3·(2·n_sections + 1)`` samples, fewer when every
        section has ``b2 = 0`` or every one ``a2 = 0``.

    Notes
    -----
    The sections and the filter state are float64 (complex128 for complex
    samples).  At very low normalized cutoffs (phase-drift extraction) the
    poles crowd near ``z = 1``, and single-precision sections or state are
    not stable: the causal output of a 4th-order Butterworth at
    ``1e-4·f_s`` is 17 % off in float32.

    On the CPU the samples keep their precision: a Numba kernel reads and
    writes them in their own dtype, and the odd extension of the zero-phase
    filter is computed in it too.  On the GPU the filter runs on the device
    with ``cupyx.scipy.signal``, whose block-parallel algorithm computes in
    the dtype of the samples and sections together, so single-precision
    samples are filtered in double precision and cast back: a temporary
    double-precision copy of the record.
    """
    signal_adapter = adapt_signal(samples, function_name="iir_filter()")
    x, xp, sp = dispatch(signal_adapter.array)
    sos_np = np.asarray(to_device(sos, "cpu"), dtype=np.float64)
    if sos_np.ndim != 2 or sos_np.shape[1] != 6:
        raise ValueError(
            f"iir_filter(): sos must have shape (n_sections, 6), got {sos_np.shape}."
        )
    if not np.allclose(sos_np[:, 3], 1.0):
        raise ValueError("iir_filter(): sos[:, 3] (each section's a0) must be 1.")

    logger.debug(
        "Applying IIR filter (%s SOS sections, zero_phase=%s).",
        sos_np.shape[0],
        zero_phase,
    )

    # Single and double precision pass through; other dtypes (integers) are
    # filtered in double precision.
    complex_input = np.iscomplexobj(x)
    if x.dtype not in (np.float32, np.float64, np.complex64, np.complex128):
        x = x.astype(np.complex128 if complex_input else np.float64)
    state_dtype = np.complex128 if complex_input else np.float64
    if zero_phase:
        _zero_phase_padding(sos_np, x.shape[-1])
    if xp is not np:
        # Deliberate double precision on the GPU (see Notes and AGENTS.md).
        x_work = x.astype(state_dtype, copy=False)
        sos_dev = xp.asarray(sos_np)
        if zero_phase:
            result = sp.signal.sosfiltfilt(sos_dev, x_work, axis=-1)
        else:
            result = sp.signal.sosfilt(sos_dev, x_work, axis=-1)
        return signal_adapter.wrap_samples(result.astype(x.dtype, copy=False))
    x_host, was_1d = as_2d(np.asarray(to_device(x, "cpu")), name="samples")
    if zero_phase:
        y = _sosfiltfilt(sos_np, x_host, state_dtype)
    else:
        zi = np.zeros((x_host.shape[0], sos_np.shape[0], 2), dtype=state_dtype)
        y = _sosfilt(sos_np, x_host, zi)
    return signal_adapter.wrap_samples(restore_1d(was_1d, np.ascontiguousarray(y)))
