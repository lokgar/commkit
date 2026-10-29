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

from .. import _cuda, _fft
from .._array import as_2d, restore_1d
from .._circular import convolve_periodic
from .._overlap_save import ols_apply
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


def _design_sos(
    name: str,
    design: Any,
    args: tuple[Any, ...],
    *,
    sampling_rate: float,
    cutoff: float | tuple[float, float],
    btype: str,
    **kwargs: Any,
) -> np.ndarray:
    """Second-order sections of the SciPy IIR ``design`` for cutoffs in Hz.

    Calls ``design(*args, Wn, btype=btype, output="sos", **kwargs)`` with the
    cutoffs normalized to the Nyquist frequency.
    """
    Wn = _iir_wn(cutoff, btype, 0.5 * float(sampling_rate))
    logger.debug(
        "Designing %s SOS: btype=%s, cutoff=%s Hz, args=%s %s.",
        name,
        btype,
        cutoff,
        args,
        kwargs,
    )
    return np.asarray(design(*args, Wn, btype=btype, output="sos", **kwargs))


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
    return _design_sos(
        "Butterworth",
        scipy.signal.butter,
        (order,),
        sampling_rate=sampling_rate,
        cutoff=cutoff,
        btype=btype,
    )


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
    return _design_sos(
        "Chebyshev-I",
        scipy.signal.cheby1,
        (order, ripple),
        sampling_rate=sampling_rate,
        cutoff=cutoff,
        btype=btype,
    )


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
    return _design_sos(
        "Chebyshev-II",
        scipy.signal.cheby2,
        (order, attenuation),
        sampling_rate=sampling_rate,
        cutoff=cutoff,
        btype=btype,
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
    return _design_sos(
        "Elliptic",
        scipy.signal.ellip,
        (order, ripple, attenuation),
        sampling_rate=sampling_rate,
        cutoff=cutoff,
        btype=btype,
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
    return _design_sos(
        "Bessel",
        scipy.signal.bessel,
        (order,),
        sampling_rate=sampling_rate,
        cutoff=cutoff,
        btype=btype,
        norm=norm,
    )


def _taps_like(taps: ArrayType, x: ArrayType, xp: Any, function_name: str) -> ArrayType:
    """``taps`` in the precision of ``x``; complex taps need complex samples.

    The samples set the precision, so float64 tap designs do not promote
    single-precision data.  The output keeps the dtype of the samples, which
    for real samples has no room for the imaginary part of complex taps.
    """
    if xp.iscomplexobj(taps) and not xp.iscomplexobj(x):
        raise ValueError(
            f"{function_name}: complex taps need complex samples, got {x.dtype} "
            f"samples; pass samples.astype(complex64) for a complex output."
        )
    target = x.dtype if xp.iscomplexobj(taps) else x.real.dtype
    return taps if taps.dtype == target else taps.astype(target)


def ols_fir_filter(
    samples: S,
    taps: ArrayType,
    *,
    nfft: int | None = None,
    causal: bool = False,
) -> S:
    """Apply an FIR filter by overlap-save, for long filters and long records.

    Overlap-save (OLS) processes the record in FFT blocks of fixed size,
    transformed in batches of at most 2^24 spectrum elements, so the working
    memory beyond the input and output stays bounded for long records and
    long impulse responses (chromatic dispersion, group-delay equalizers),
    where one FFT of the whole record would exhaust GPU memory.  For short filters on moderate records, :func:`fir_filter` is
    as fast and simpler.

    Parameters
    ----------
    samples : array_like or Signal
        Samples, ``(N,)`` or ``(C, N)``.  A Signal gives a new Signal.
    taps : array_like
        FIR coefficients, ``(L,)``.
    nfft : int, optional
        FFT block size, a power of 2.  Defaults to
        ``max(1024, next_power_of_2(4·L))``, so the guard of a quarter block
        is at least ``L`` samples long.
    causal : bool, default False
        By default the output is aligned as :func:`fir_filter` aligns it
        (SciPy's ``mode='same'``, centered on tap ``L // 2``):
        ``y[n] = sum_k x[n + L//2 - k]·taps[k]``, which removes the group
        delay of a symmetric filter before symbol sampling.  With ``True``
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
        If ``nfft`` is not a power of 2 or ``nfft // 4 < len(taps)``,
        or ``taps`` is complex and ``samples`` real.

    Notes
    -----
    A symmetric guard of ``nfft // 4`` samples is discarded from each
    block edge, so ``nfft // 4 >= len(taps)`` must hold.

    Without ``causal`` the input is padded at the end by ``L // 2`` zeros
    and as many leading output samples are dropped, which costs at most one
    extra block.
    """
    signal_adapter = adapt_signal(samples, function_name="ols_fir_filter()")
    x = signal_adapter.array

    x, xp, _ = dispatch(x)
    taps = _taps_like(xp.asarray(taps), x, xp, "ols_fir_filter()")
    is_real = not xp.iscomplexobj(x) and not xp.iscomplexobj(taps)
    out_dtype = x.dtype  # capture before any reshape

    L = len(taps)
    half = L // 2

    x, was_1d = as_2d(x, name="samples")

    N = x.shape[-1]

    N_fft = nfft
    if N_fft is None:
        N_fft = max(1024, 1 << (max(1, 4 * L) - 1).bit_length())
    elif N_fft & (N_fft - 1) or N_fft // 4 < L:
        raise ValueError(
            f"nfft must be a power of 2 with nfft // 4 >= len(taps) ({L}), got {nfft}."
        )

    logger.debug(
        "ols_fir_filter: L=%s, N=%s, N_fft=%s, num_ch=%s, causal=%s",
        L,
        N,
        N_fft,
        x.shape[0],
        causal,
    )

    H = _fft.fft(taps, n=N_fft)  # frequency response of the filter

    if not causal:
        # Post-pad by half so the OLS can compute full_conv[half : half+N].
        # This matches scipy's mode='same' (center-aligned, group-delay compensated),
        # which is required for correct eye-opening after pulse-shaped filtering.
        samples_ext = xp.pad(x, ((0, 0), (0, half)))
        out_ext = ols_apply(samples_ext, N_fft, lambda Y: Y * H)  # (num_ch, N + half)
        out = out_ext[:, half:]  # trim leading half -> shape: (num_ch, N)
    else:
        out = ols_apply(x, N_fft, lambda Y: Y * H)

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

    Raises
    ------
    ValueError
        If ``taps`` is not 1-D, or is complex and ``samples`` real.
    """
    signal_adapter = adapt_signal(samples, function_name="fir_filter()")
    x, xp, sp = dispatch(signal_adapter.array)
    taps = xp.asarray(taps)
    if taps.ndim != 1:
        raise ValueError(f"taps must be 1-D, got shape {taps.shape}.")
    logger.debug("Applying FIR filter via convolution (%s taps).", taps.size)
    taps = _taps_like(taps, x, xp, "fir_filter()")

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


# Samples per block and blocks per group of the GPU filter: enough threads
# for long records, and short sequential carries (G steps per group, one
# step per group across the record).
_IIR_BLOCK = 256
_IIR_GROUP = 32
_IIR_TRANSITIONS: dict[tuple[bytes, int, int], np.ndarray] = {}


def _iir_transitions(sos: np.ndarray, length: int, count: int) -> np.ndarray:
    """Zero-input state transitions of the cascade over ``j·length`` samples.

    ``(count + 1, 2S, 2S)`` float64, ``j = 0 .. count``, state index
    ``2k + i`` for ``z_i`` of section ``k``: column ``m`` of matrix ``j`` is
    the state that the unit state ``m`` reaches through ``j·length`` zeros.
    Run by the CPU kernel itself, segment after segment, so they round as
    the sequential filter does (a matrix power was 100x less accurate at a
    cutoff of 1e-5·fs).  Cached per sections, length and count.
    """
    key = (sos.tobytes(), length, count)
    if key not in _IIR_TRANSITIONS:
        S = sos.shape[0]
        D = 2 * S
        z = np.eye(D).reshape(D, S, 2)  # row m: the unit state m
        out = np.empty((count + 1, D, D))
        out[0] = np.eye(D)
        zeros, scratch = np.zeros((D, length)), np.empty((D, length))
        for j in range(1, count + 1):
            _get_numba_sos()(sos, zeros, z, scratch)
            out[j] = z.reshape(D, D).T
        if len(_IIR_TRANSITIONS) > 64:
            _IIR_TRANSITIONS.clear()
        _IIR_TRANSITIONS[key] = out
    return _IIR_TRANSITIONS[key]


def _sosfilt(sos: np.ndarray, x: ArrayType, zi: ArrayType, xp: Any) -> ArrayType:
    """Causal SOS filter of ``(C, N)`` data from the initial state ``zi``.

    ``sos`` is float64 on the host; ``zi`` is ``(C, S, 2)`` in float64 or
    complex128 on the device of ``x`` and is not modified.  The Numba kernel
    on the CPU, the block-parallel ``iir`` kernel on the GPU, both in double
    precision.  Returns ``(C, N)`` in the dtype of ``x``.
    """
    x = xp.ascontiguousarray(x)
    if xp is not np:
        block = min(_IIR_BLOCK, max(1, x.shape[-1]))
        group = max(1, min(_IIR_GROUP, -(-x.shape[-1] // block)))
        transitions = _iir_transitions(sos, block, group)
        return _cuda.get_kernel("iir")(
            sos, x, xp.ascontiguousarray(zi), block, group, transitions
        )
    y = np.empty_like(x)
    _get_numba_sos()(sos, x, zi.copy(), y)
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


def _sosfiltfilt(
    sos: np.ndarray, x: ArrayType, state_dtype: type, xp: Any
) -> ArrayType:
    """Zero-phase SOS filter of ``(C, N)`` data, as ``scipy.signal.sosfiltfilt``.

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
    ext = xp.concatenate([left, x, right], axis=-1)
    zi = xp.asarray(scipy.signal.sosfilt_zi(sos).astype(state_dtype))  # (S, 2)
    y = _sosfilt(sos, ext, zi[None] * ext[:, :1, None].astype(state_dtype), xp)
    y = _sosfilt(sos, y[:, ::-1], zi[None] * y[:, -1:, None].astype(state_dtype), xp)
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

    The samples keep their precision on both devices: the kernels read and
    write them in their own dtype, and the odd extension of the zero-phase
    filter is computed in it too.  On the CPU a Numba kernel filters each
    channel sample by sample (double input gives SciPy's result bit for
    bit).  On the GPU a CUDA kernel filters blocks of the record in parallel
    and carries the state across them; it agrees with SciPy to about 1e-14
    at ordinary cutoffs and 2e-10 at ``1e-5·f_s``.
    """
    signal_adapter = adapt_signal(samples, function_name="iir_filter()")
    x, xp, _ = dispatch(signal_adapter.array)
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
    x2, was_1d = as_2d(x, name="samples")
    if zero_phase:
        y = _sosfiltfilt(sos_np, x2, state_dtype, xp)
    else:
        zi = xp.zeros((x2.shape[0], sos_np.shape[0], 2), dtype=state_dtype)
        y = _sosfilt(sos_np, x2, zi, xp)
    return signal_adapter.wrap_samples(restore_1d(was_1d, xp.ascontiguousarray(y)))
