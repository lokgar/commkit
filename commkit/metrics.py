"""Performance metrics: EVM, SNR, BER, SER, GMI and MI.

Metrics are the reporting layer.  They take received symbols (or bits, or
LLRs) at one sample per symbol and their reference, as arrays or as a 1-SPS
:class:`~commkit.core.Signal` whose ``reference`` holds the transmitted
symbols and bits.  They return host values: a ``float`` for ``(N,)`` input
and a ``(C,)`` NumPy array for ``(C, N)`` input.

Metrics never align, rotate or permute the received symbols against the
reference: synchronization, phase-ambiguity and channel-permutation
resolution are explicit receiver steps.  Lengths that disagree, empty
selections and frame Signals (use :func:`~commkit.core.extract_payload`)
raise.

Units and scaling
-----------------
Units are carried by argument names (``esn0_db``, ``noise_var``); there is
no units object.

==========================  =================================================
Quantity                    Definition in commkit
==========================  =================================================
Sample power ``P``          ``E[|x|^2]`` per complex sample.  At ``sps``
                            samples per symbol a normalized waveform has
                            ``P = 1/sps``; at the symbol rate ``P = 1``.
Symbol energy ``Es``        ``sps * P``: 1 for normalized waveforms and for
                            symbols on a constellation's scale.  Every
                            factory constellation, shaped ones included, has
                            unit average power under its pmf.
``Es/N0`` (``esn0_db``)     Symbol energy over noise spectral density.
                            ``apply_awgn`` adds complex noise of variance
                            ``sps * P / (Es/N0)`` per sample, which is
                            ``Es/N0`` per symbol after a matched filter and
                            decimation.
Noise variance              ``noise_var = sigma^2 = E[|n|^2]`` per complex
(``noise_var``)             symbol on the constellation's scale, so
                            ``sigma^2 = 10 ** (-esn0_db / 10)`` at unit
                            ``Es``.  Each quadrature (I, Q) carries
                            ``sigma^2 / 2``.
SNR (:func:`snr`, dB)       ``Es / E[|r/g - s|^2]`` at the symbol rate, with
                            ``g`` the data-aided gain (below); equal to
                            ``Es/N0`` for AWGN after a matched filter.
EVM (:func:`evm`, %)        ``100 sqrt(E[|r/g - s|^2] / Es)``, RMS, relative
                            to the reference's average power; ``= 100 *
                            10 ** (-SNR_dB / 20)``.  Blind EVM is relative
                            to the constellation's average power.
MI, GMI (bits/symbol)       At most ``H(X)``: ``log2(M)`` when uniform, the
                            entropy of the pmf when shaped.  GMI is the
                            bit-metric decoding rate
                            ``H(X) - sum_b H(B_b | Y)``.
==========================  =================================================

Which scale each metric reads:

* ``evm`` and ``snr`` normalize the reference to unit average power and
  divide the received symbols by the data-aided gain
  ``g = |<r s*>| / <|s|^2>`` (per channel, after ``num_skip_symbols``).
  Unlike normalizing to unit total power, which shrinks the symbols by
  ``1/sqrt(1 + 1/SNR)``, this is unbiased at any SNR.  Only the magnitude
  is used, so a rotation stays an error.
* ``mi`` and ``gmi`` of a Signal with a reference divide by the same gain,
  which puts the symbols on the constellation's scale whatever their
  normalization; ``noise_var`` is then on that scale.  Arrays are read as
  given.
* ``ser`` and ``ber`` decide on the symbols as given, against the
  constellation's points: decision thresholds are the receiver's.
"""

import logging
import math
from typing import Any

import numpy as np

from ._array import to_report_scalar
from .backend import ArrayType, dispatch, to_device
from .core._signal_adapter import SignalAdapter, adapt_signal
from .core.signal import Signal
from .logger import logger
from .mapping import Constellation
from .math import linear_to_db, normalize

__all__ = ["ber", "evm", "gmi", "mi", "ser", "snr"]


def _unit_reference(tx: ArrayType, xp: Any) -> tuple[ArrayType, ArrayType]:
    """``tx`` at unit average power per channel, and its power ``(..., 1)``.

    A reference already at unit power (within 1e-3) is kept as it is; the
    power is measured once and remeasured only after normalizing.
    """
    pwr = xp.mean(xp.abs(tx) ** 2, axis=-1, keepdims=True)
    if not xp.allclose(pwr, 1.0, atol=1e-3):
        tx = normalize(tx, axis=-1, mode="average_power")
        pwr = xp.mean(xp.abs(tx) ** 2, axis=-1, keepdims=True)
    return tx, pwr


def _log_per_channel(fmt: str, *arrays: ArrayType, extra: tuple = ()) -> None:
    """Gate per-channel INFO logging behind one D2H transfer per array.

    Emits, per channel index ``ch``: ``logger.info(fmt, ch, *values, *extra)``
    where ``values`` are each array's per-channel scalar (Python ``int`` for
    integer dtypes, ``float`` otherwise) and ``extra`` are constants appended
    after the per-channel values (e.g. a fixed ``total`` symbol/bit count).
    No device transfer happens at all when INFO logging is disabled.
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    arrays_np = [to_device(a, "cpu") for a in arrays]
    n_ch = arrays_np[0].shape[0]
    for ch in range(n_ch):
        values = [
            int(a[ch]) if np.issubdtype(a.dtype, np.integer) else float(a[ch])
            for a in arrays_np
        ]
        logger.info(fmt, ch, *values, *extra)


# -----------------------------------------------------------------------------
# Input resolution shared by the metrics
# -----------------------------------------------------------------------------


def _reference(
    adapter: SignalAdapter, reference: Any, field: str, name: str, hint: str = ""
) -> Any:
    """The explicit reference, else the Signal's ``reference.<field>``."""
    if reference is not None:
        return reference
    ref = None if adapter.signal is None else adapter.signal.reference
    value = None if ref is None else getattr(ref, field)
    if value is None:
        raise ValueError(
            f"{name} needs reference {field}: pass reference= or a Signal whose "
            f"reference has {field}{hint}."
        )
    return value


def _constellation(adapter: SignalAdapter, constellation: Any, name: str) -> Any:
    """The decision constellation: the argument, else the Signal's."""
    constellation = adapter.resolve_choice("constellation", constellation)
    if constellation is None:
        raise ValueError(
            f"{name} needs a constellation: pass constellation= or a Signal that "
            "has one."
        )
    if not isinstance(constellation, Constellation):
        raise TypeError(
            f"{name}: constellation must be a Constellation, got "
            f"{type(constellation).__name__}; use e.g. Constellation.qam(16)."
        )
    return constellation


def _paired(
    rx: ArrayType, reference: Any, xp: Any, name: str
) -> tuple[ArrayType, ArrayType]:
    """Move the reference to ``rx``'s device and precision.

    The precision is matched when both are floating point, so a complex128
    reference does not promote complex64 data.  The shapes must agree.
    """
    ref = to_device(reference, "cpu" if xp is np else "gpu")
    if rx.shape != ref.shape:
        raise ValueError(
            f"{name}: shape mismatch: received {rx.shape} != reference {ref.shape}."
        )
    if ref.dtype.kind in "fc" and rx.dtype.kind in "fc":
        base = np.complex64 if ref.dtype.kind == "c" else np.float32
        ref = ref.astype(np.result_type(base, rx.real.dtype), copy=False)
    return rx, ref


def _gain(
    rx: ArrayType, tx: ArrayType, xp: Any, tx_power: ArrayType | None = None
) -> ArrayType:
    """The data-aided gain ``g = |<r s*>| / <|s|^2>`` per channel, ``(..., 1)``.

    ``g`` is the amplitude of the symbols in ``rx`` (the transmittance), so
    ``rx / g`` is on the scale of ``tx`` whatever the noise; normalizing to
    unit total power instead shrinks the symbols by ``1/sqrt(1 + 1/SNR)``.
    Only the magnitude is used: a rotation, ``pi`` included, stays visible.
    A channel without reference power or correlation gets ``g = 1``.
    ``tx_power`` is ``<|s|^2>`` ``(..., 1)`` when the caller has it.
    """
    num = xp.abs(xp.mean(rx * xp.conj(tx), axis=-1, keepdims=True))
    den = (
        xp.mean(xp.abs(tx) ** 2, axis=-1, keepdims=True)
        if tx_power is None
        else tx_power
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        g = num / den
    return xp.where((den > 0) & (g > 0), g, 1.0).astype(rx.real.dtype)


def _reference_scaled(
    adapter: SignalAdapter, rx: ArrayType, xp: Any, name: str, num_skip: int
) -> ArrayType:
    """Divide a Signal's symbols by their data-aided gain.

    The gain is estimated against the reference symbols after the first
    ``num_skip``.  Arrays and Signals without a reference are returned as
    given.
    """
    sig = adapter.signal
    if sig is None or sig.reference is None:
        return rx
    _, ref = _paired(rx, sig.reference.symbols, xp, name)
    measured = _skip(rx, num_skip, name)
    return rx / _gain(measured, _skip(ref, num_skip, name), xp)


def _skip(x: ArrayType, n: int, name: str) -> ArrayType:
    """Drop the first ``n`` entries of the last axis; nothing left raises."""
    if n < 0:
        raise ValueError(f"{name}: num_skip_symbols must be >= 0, got {n}.")
    x = x[..., n:]
    if x.shape[-1] == 0:
        raise ValueError(f"{name}: nothing to measure (no symbols after skipping).")
    return x


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def evm(
    symbols: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    constellation: Constellation | None = None,
    blind: bool = False,
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    """Error vector magnitude (RMS), in percent.

    Data-aided by default: the error is measured against the known
    transmitted symbols.  With ``blind=True`` it is measured against the
    nearest constellation point instead, as a vector signal analyzer does;
    no reference is needed, but at low SNR wrong decisions make the estimate
    optimistic.

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols at one sample per symbol, ``(N,)`` or ``(C, N)``.  A
        Signal must be at one sample per symbol; its ``reference.symbols`` is
        the default reference.
    reference : array_like, optional
        Transmitted symbols, same shape as ``symbols``.  Required for
        data-aided EVM of an array.
    constellation : Constellation, optional
        Decision constellation for ``blind=True``.  Defaults to the Signal's.
        The symbols are compared with its points as they are, so they must be
        on its scale (unit power).
    blind : bool, default False
        Measure against hard decisions instead of the reference.
    num_skip_symbols : int, default 0
        Leading symbols left out (for example equalizer training).

    Returns
    -------
    float or ndarray
        EVM in percent: a float for ``(N,)`` input, ``(C,)`` for ``(C, N)``.
        ``inf`` where the reference has no power.  In dB:
        ``20 * log10(evm / 100)``.

    Raises
    ------
    ValueError
        On a missing reference or constellation, a shape mismatch, an empty
        selection, or a frame Signal or one not at one sample per symbol.

    Notes
    -----
    Data-aided EVM normalizes the reference to unit average power per
    channel and divides the received symbols by the data-aided gain
    ``|<r s*>| / <|s|^2>``, so a common gain does not count as error at any
    SNR (normalizing to unit total power would shrink the symbols by
    ``1/sqrt(1 + 1/SNR)``).  No rotation is applied.
    """
    name = "evm()"
    adapter = adapt_signal(symbols, function_name=name)
    rx, xp, _ = dispatch(adapter.symbol_array())
    axis = -1

    if blind:
        if reference is not None:
            raise ValueError(
                f"{name}: blind=True measures against hard decisions; drop reference=."
            )
        c = _constellation(adapter, constellation, name)
        rx = _skip(rx, num_skip_symbols, name)
        # No gain correction: the caller passes symbols at the constellation's
        # scale.
        points = c.points.astype(np.result_type(np.complex64, rx.real.dtype))
        tx = xp.asarray(points)[c._nearest(rx.reshape(-1))].reshape(rx.shape)
        ref_pwr = xp.mean(xp.abs(tx) ** 2, axis=axis)
    else:
        ref = _reference(adapter, reference, "symbols", name, hint=", or blind=True")
        rx, tx = _paired(rx, ref, xp, name)
        rx = _skip(rx, num_skip_symbols, name)
        tx = _skip(tx, num_skip_symbols, name)
        tx, tx_pwr = _unit_reference(tx, xp)
        rx = rx / _gain(rx, tx, xp, tx_pwr)
        ref_pwr = tx_pwr[..., 0]

    error_power = xp.mean(xp.abs(rx - tx) ** 2, axis=axis, dtype=xp.float64)
    # The reference power is 1 after normalization.
    evm_percent = xp.sqrt(error_power) * 100.0
    evm_percent = xp.where(ref_pwr < 1e-20, xp.inf, evm_percent)
    out = to_report_scalar(evm_percent)
    if np.any(np.isinf(out)):
        logger.warning("Reference signal power near zero in one or more channels.")
    if isinstance(out, float):
        logger.info("EVM: %.2f%%", out)
    else:
        _log_per_channel("EVM Ch%s: %.2f%%", out)
    return out


def snr(
    symbols: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    """Data-aided SNR in dB: signal power over error power.

    ``SNR = 1 / E[|r / g - s|^2]`` with the reference ``s`` normalized to
    unit average power per channel and the data-aided gain
    ``g = |<r s*>| / <|s|^2>``, which is unbiased at any SNR (normalizing
    ``r`` to unit total power is not).  No rotation is applied.

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols at one sample per symbol, ``(N,)`` or ``(C, N)``.  A
        Signal must be at one sample per symbol; its ``reference.symbols`` is
        the default reference.
    reference : array_like, optional
        Transmitted symbols, same shape as ``symbols``.  Required for arrays.
    num_skip_symbols : int, default 0
        Leading symbols left out (for example equalizer training).

    Returns
    -------
    float or ndarray
        SNR in dB: a float for ``(N,)`` input, ``(C,)`` for ``(C, N)``.
        ``inf`` without error, ``-inf`` where the reference has no power.

    Raises
    ------
    ValueError
        On a missing reference, a shape mismatch, an empty selection, or a
        frame Signal or one not at one sample per symbol.
    """
    name = "snr()"
    adapter = adapt_signal(symbols, function_name=name)
    rx, xp, _ = dispatch(adapter.symbol_array())
    rx, tx = _paired(rx, _reference(adapter, reference, "symbols", name), xp, name)
    rx = _skip(rx, num_skip_symbols, name)
    tx = _skip(tx, num_skip_symbols, name)
    axis = -1

    tx, tx_pwr = _unit_reference(tx, xp)
    rx = rx / _gain(rx, tx, xp, tx_pwr)
    ref_pwr = tx_pwr[..., 0]

    noise_power = xp.mean(xp.abs(rx - tx) ** 2, axis=axis, dtype=xp.float64)
    with np.errstate(divide="ignore"):
        snr_db = linear_to_db(1.0 / noise_power, power=True)
    # No error: +inf even for a zero reference; error but no reference: -inf.
    zero_noise = noise_power < 1e-20
    snr_db = xp.where(zero_noise, xp.inf, snr_db)
    snr_db = xp.where((ref_pwr < 1e-20) & ~zero_noise, -xp.inf, snr_db)
    out = to_report_scalar(snr_db)
    if isinstance(out, float):
        logger.info("SNR: %.2f dB", out)
    else:
        _log_per_channel("SNR Ch%s: %.2f dB", out)
    return out


def ber(
    bits: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    constellation: Constellation | None = None,
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    """Bit error rate.

    Parameters
    ----------
    bits : array_like or Signal
        Received bits (0 and 1), ``(N_bits,)`` or ``(C, N_bits)``.  A Signal holds
        received *symbols* at one sample per symbol: they are hard-decided
        against the constellation and compared with ``reference.bits``.
    reference : array_like, optional
        Transmitted bits, same shape as the received bits.  Required for
        arrays.
    constellation : Constellation, optional
        Decision constellation for a Signal (defaults to the Signal's).  For
        bit arrays, needed only to convert ``num_skip_symbols`` to bits.
    num_skip_symbols : int, default 0
        Leading symbols left out, ``k = constellation.bits_per_symbol`` bits
        each.

    Returns
    -------
    float or ndarray
        BER in ``[0, 1]``: a float for 1-D input, ``(C,)`` for 2-D input.

    Raises
    ------
    ValueError
        On a missing reference or constellation, a shape mismatch, an empty
        selection, or a frame Signal or one not at one sample per symbol.
    """
    name = "ber()"
    adapter = adapt_signal(bits, function_name=name)
    received = adapter.symbol_array()
    if adapter.signal is not None or num_skip_symbols:
        c = _constellation(adapter, constellation, name)
        k = c.bits_per_symbol
        if adapter.signal is not None:
            received = c.demap(received)
    rx, xp, _ = dispatch(received)
    rx, tx = _paired(rx, _reference(adapter, reference, "bits", name), xp, name)
    if num_skip_symbols:
        rx = _skip(rx, num_skip_symbols * k, name)
        tx = _skip(tx, num_skip_symbols * k, name)
    elif rx.shape[-1] == 0:
        raise ValueError(f"{name}: nothing to measure (no bits).")

    errors = xp.sum(rx != tx, axis=-1)
    total = rx.shape[-1]
    out = to_report_scalar(errors / total)
    if isinstance(out, float):
        logger.info("BER: %.2e (%s/%s errors)", out, int(errors), total)
    else:
        _log_per_channel("BER Ch%s: %.2e (%s/%s errors)", out, errors, extra=(total,))
    return out


def ser(
    symbols: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    constellation: Constellation | None = None,
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    """Symbol error rate of minimum-distance hard decisions.

    The received and the reference symbols are both decided against the
    constellation, so rounding in the reference does not count.  SER does
    not depend on the bit labeling.

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols at one sample per symbol, ``(N,)`` or ``(C, N)``, on
        the constellation's scale.  A Signal must be at one sample per
        symbol; its ``reference.symbols`` is the default reference.
    reference : array_like, optional
        Transmitted symbols, same shape as ``symbols``.  Required for arrays.
    constellation : Constellation, optional
        Decision constellation.  Defaults to the Signal's; required for
        arrays.
    num_skip_symbols : int, default 0
        Leading symbols left out (for example equalizer training).

    Returns
    -------
    float or ndarray
        SER in ``[0, 1]``: a float for ``(N,)`` input, ``(C,)`` for
        ``(C, N)``.

    Raises
    ------
    ValueError
        On a missing reference or constellation, a shape mismatch, an empty
        selection, or a frame Signal or one not at one sample per symbol.
    """
    name = "ser()"
    adapter = adapt_signal(symbols, function_name=name)
    rx, xp, _ = dispatch(adapter.symbol_array())
    c = _constellation(adapter, constellation, name)
    rx, tx = _paired(rx, _reference(adapter, reference, "symbols", name), xp, name)
    rx = _skip(rx, num_skip_symbols, name)
    tx = _skip(tx, num_skip_symbols, name)

    dec_rx = c._nearest(rx.reshape(-1)).reshape(rx.shape)
    dec_tx = c._nearest(tx.reshape(-1)).reshape(tx.shape)
    errors = xp.sum(dec_rx != dec_tx, axis=-1)
    total = rx.shape[-1]
    out = to_report_scalar(errors / total)
    if isinstance(out, float):
        logger.info("SER: %.2e (%s/%s errors)", out, int(errors), total)
    else:
        _log_per_channel("SER Ch%s: %.2e (%s/%s errors)", out, errors, extra=(total,))
    return out


def gmi(
    llrs: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    constellation: Constellation | None = None,
    noise_var: float | None = None,
    method: str = "maxlog",
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    r"""Generalized mutual information (bit-metric decoding rate), bits/symbol.

    .. math::

        \mathrm{GMI} = H(X) - \sum_{b=0}^{k-1}
            \mathbb{E}\big[\log_2(1 + e^{-(1 - 2 c_b) L_b})\big]

    with the entropy ``H(X)`` of the constellation's prior (``k`` bits when
    uniform), the LLR ``L_b`` of bit ``b`` (positive favors 0) and the
    transmitted bit ``c_b``.  Each expectation estimates ``H(B_b | Y)``; with
    a shaped constellation the bits are not uniform, so the rate is not
    ``k - ...``.  The LLRs must include the prior, as those of
    :func:`~commkit.mapping.compute_llr` do.  The softplus is evaluated in a
    numerically stable form.

    Parameters
    ----------
    llrs : array_like or Signal
        Bit LLRs in the layout of :func:`~commkit.mapping.compute_llr`,
        ``(N * k,)`` or ``(C, N * k)`` (the ``k`` bits of a symbol adjacent).
        A Signal holds received *symbols* at one sample per symbol: they are
        divided by the data-aided gain against ``reference.symbols`` (if
        present), and their LLRs are computed with ``noise_var`` and
        ``method`` and compared with ``reference.bits``.
    reference : array_like, optional
        Transmitted bits, same shape as the LLRs.  Required for arrays.
    constellation : Constellation, optional
        Gives ``k``, and for a Signal the points, labels and prior of the
        LLRs.  Defaults to the Signal's; required for arrays.
    noise_var : float, optional
        Complex noise variance on the constellation's scale, for a Signal
        only (an LLR array already contains it).
    method : {"maxlog", "exact"}, default "maxlog"
        LLR computation for a Signal.
    num_skip_symbols : int, default 0
        Leading symbols left out (``k`` LLRs each).

    Returns
    -------
    float or ndarray
        GMI in bits per symbol: a float for 1-D input, ``(C,)`` for 2-D
        input.

    Raises
    ------
    ValueError
        On a missing reference, constellation or ``noise_var``, a shape
        mismatch, an empty selection, or a frame Signal or one not at one
        sample per symbol.
    """
    name = "gmi()"
    adapter = adapt_signal(llrs, function_name=name)
    received = adapter.symbol_array()
    c = _constellation(adapter, constellation, name)
    k = c.bits_per_symbol
    if adapter.signal is not None:
        if noise_var is None:
            raise ValueError(f"{name} needs noise_var to compute LLRs of a Signal.")
        received = _reference_scaled(
            adapter, received, dispatch(received)[1], name, num_skip_symbols
        )
        received = c.llr(received, noise_var=noise_var, method=method)
    elif noise_var is not None:
        raise ValueError(
            f"{name}: noise_var is for Signal input; LLR arrays already contain it."
        )
    llr_arr, xp, _ = dispatch(received)
    llr_arr, ref = _paired(
        llr_arr, _reference(adapter, reference, "bits", name), xp, name
    )
    if llr_arr.shape[-1] % k:
        raise ValueError(
            f"{name}: the last axis ({llr_arr.shape[-1]}) is not a multiple of "
            f"bits_per_symbol ({k})."
        )
    # Per element in the LLRs' precision (float32 from compute_llr), summed in
    # float64: single precision changed no GMI by more than 2e-8 bit (QAM 16
    # to 256), far below the Monte Carlo error.
    llr_arr = _skip(llr_arr, num_skip_symbols * k, name)
    real = llr_arr.dtype if llr_arr.dtype.kind == "f" else xp.dtype(xp.float32)
    llr_arr = llr_arr.astype(real, copy=False)
    bits_arr = _skip(ref, num_skip_symbols * k, name).astype(real)

    # -(1 - 2c) L: negative (small loss) when the LLR favors the sent bit.
    x = -llr_arr * (1.0 - 2.0 * bits_arr)
    # log2(1 + e^x) = (log1p(e^-|x|) + max(0, x)) / ln 2
    # math.log, not np.log: a NumPy float64 scalar would promote (NEP 50).
    softplus = (xp.log1p(xp.exp(-xp.abs(x))) + xp.maximum(0.0, x)) / math.log(2.0)
    # H(X) - sum_b E[...], written so that H(X)/k is exactly 1.0 when uniform.
    loss = xp.mean(softplus, axis=-1, dtype=xp.float64)
    out = to_report_scalar(k * (c.entropy / k - loss))
    if isinstance(out, float):
        logger.info("GMI: %.4f b/symbol", out)
    else:
        _log_per_channel("GMI Ch%s: %.4f b/symbol", out)
    return out


def mi(
    symbols: ArrayType | Signal,
    *,
    noise_var: float,
    constellation: Constellation | None = None,
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    r"""Mutual information of the AWGN channel with the constellation's prior.

    Monte-Carlo estimate with Gaussian likelihoods
    ``p(r | s_m) ∝ exp(-|r - s_m|^2 / sigma^2)`` and prior ``P(s_m)``:

    .. math::

        \mathrm{MI} = H(X) + \frac{1}{N} \sum_n \sum_m
            p(s_m | r_n) \log_2 p(s_m | r_n)

    It upper-bounds the GMI and is achievable with non-binary coding.

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols at one sample per symbol, ``(N,)`` or ``(C, N)``, on
        the constellation's scale.  A Signal must be at one sample per
        symbol; with a reference, its samples are first divided by the
        data-aided gain ``|<r s*>| / <|s|^2>``, which puts them on the
        constellation's scale whatever their normalization.
    noise_var : float
        Complex noise variance ``sigma^2 = E[|n|^2]`` on the constellation's
        scale.  For a unit-power constellation at Es/N0 in dB:
        ``10 ** (-esn0_db / 10)``.
    constellation : Constellation, optional
        Points and prior.  Defaults to the Signal's; required for arrays.
    num_skip_symbols : int, default 0
        Leading symbols left out (for example equalizer training).

    Returns
    -------
    float or ndarray
        MI in bits per symbol, in ``[0, H(X)]``: a float for ``(N,)`` input,
        ``(C,)`` for ``(C, N)``.

    Raises
    ------
    ValueError
        On a missing constellation, an empty selection, or a frame Signal or
        one not at one sample per symbol.
    """
    name = "mi()"
    adapter = adapt_signal(symbols, function_name=name)
    rx, xp, _ = dispatch(adapter.symbol_array())
    c = _constellation(adapter, constellation, name)
    rx = _reference_scaled(adapter, rx, xp, name, num_skip_symbols)
    # Per element in the symbols' precision (complex64 by default), summed in
    # float64: single precision changed no MI by more than 4e-8 bit (QAM 16
    # to 256), far below the Monte Carlo error.
    rx = _skip(rx, num_skip_symbols, name)
    cdtype = np.result_type(rx.dtype, np.complex64)
    rx = rx.astype(cdtype, copy=False)
    real = np.empty(0, cdtype).real.dtype
    points = xp.asarray(c.points.astype(cdtype))
    if c.pmf is not None:
        pmf = np.asarray(c.pmf, dtype=np.float64).clip(1e-300, None)
        log_prior = xp.asarray(np.log(pmf).astype(real))
    else:
        log_prior = xp.full(points.size, -np.log(points.size), dtype=real)
    inv_nv = real.type(1.0 / noise_var)
    h_x = c.entropy

    # Chunk over N to bound the (chunk, M) intermediates (~4 GB at N=1e6,
    # M=256 in one piece); the sums accumulate on the device.
    rows = rx.reshape(-1, rx.shape[-1])
    n_sym = rows.shape[-1]
    chunk = 65536
    acc = xp.zeros(rows.shape[0], dtype=xp.float64)
    for n0 in range(0, n_sym, chunk):
        diff = rows[:, n0 : n0 + chunk, None] - points  # (C, chunk, M)
        log_joint = log_prior - (diff.real**2 + diff.imag**2) * inv_nv
        shift = log_joint.max(axis=-1, keepdims=True)
        log_sum = xp.log(xp.sum(xp.exp(log_joint - shift), axis=-1)) + shift[..., 0]
        log_post = log_joint - log_sum[..., None]  # log p(s_m | r), nats
        acc += xp.sum(xp.exp(log_post) * log_post, axis=(-2, -1), dtype=xp.float64)
    mi_bits = np.clip(h_x + to_device(acc, "cpu") / n_sym / np.log(2.0), 0.0, h_x)
    out = to_report_scalar(mi_bits.reshape(rx.shape[:-1]))
    if isinstance(out, float):
        logger.info("MI: %.4f b/symbol (max %.2f)", out, h_x)
    else:
        _log_per_channel("MI Ch%s: %.4f b/symbol", out)
    return out
