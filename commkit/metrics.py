"""
Performance metrics: EVM, SNR, BER, SER, GMI and MI.

Metrics are the reporting layer.  They take received symbols (or bits, or
LLRs) at one sample per symbol and their reference, as arrays or as a 1-SPS
:class:`~commkit.core.Signal` whose ``reference`` holds the transmitted
symbols and bits.  They return host values: a ``float`` for ``(N,)`` input
and an ``np.ndarray`` of shape ``(C,)`` for ``(C, N)`` input.

Metrics never align, rotate or permute the received symbols against the
reference: synchronization, phase-ambiguity and channel-permutation
resolution are explicit receiver steps.  Lengths that disagree, empty
selections and frame Signals (use :func:`~commkit.core.extract_payload`)
raise.
"""

import logging
from typing import Any

import numpy as np

from .backend import ArrayType, dispatch, to_device
from .core._signal_adapter import SignalAdapter, adapt_signal
from .core.signal import Signal
from .logger import logger
from .mapping import Constellation
from .math import linear_to_db, normalize


def _is_normalized(arr, ax, xp):
    """True if ``arr`` has ~unit average power along ``ax`` (within 1e-3)."""
    pwr = xp.mean(xp.abs(arr) ** 2, axis=ax)
    return xp.allclose(pwr, 1.0, atol=1e-3)


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
    """The reference on ``rx``'s device; the shapes must agree."""
    ref = to_device(reference, "cpu" if xp is np else "gpu")
    if rx.shape != ref.shape:
        raise ValueError(
            f"{name}: shape mismatch: received {rx.shape} != reference {ref.shape}."
        )
    return rx, ref


def _skip(x: ArrayType, n: int, name: str) -> ArrayType:
    """Drop the first ``n`` entries of the last axis; nothing left raises."""
    if n < 0:
        raise ValueError(f"{name}: num_skip_symbols must be >= 0, got {n}.")
    x = x[..., n:]
    if x.shape[-1] == 0:
        raise ValueError(f"{name}: nothing to measure (no symbols after skipping).")
    return x


def _host(values: Any) -> float | np.ndarray:
    """A 0-d result as ``float``, per-channel results as a host float array."""
    v = np.asarray(to_device(values, "cpu"), dtype=np.float64)
    return float(v) if v.ndim == 0 else v


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
    """
    Error vector magnitude (RMS), in percent.

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
    Data-aided EVM normalizes the received symbols and the reference to unit
    average power per channel, so a common gain does not count as error.
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
        points = xp.asarray(_constellation(adapter, constellation, name).points)
        rx = _skip(rx, num_skip_symbols, name)
        from .mapping.gray import _nearest_index

        # No gain correction: the caller passes symbols at the constellation's
        # scale.  Chunked over N to bound the (N, M) distance matrix.
        tx = points[_nearest_index(rx.reshape(-1), points)].reshape(rx.shape)
    else:
        ref = _reference(adapter, reference, "symbols", name, hint=", or blind=True")
        rx, tx = _paired(rx, ref, xp, name)
        rx = _skip(rx, num_skip_symbols, name)
        tx = _skip(tx, num_skip_symbols, name)
        if not _is_normalized(rx, axis, xp):
            rx = normalize(rx, axis=axis, mode="average_power")
        if not _is_normalized(tx, axis, xp):
            tx = normalize(tx, axis=axis, mode="average_power")

    ref_pwr = xp.mean(xp.abs(tx) ** 2, axis=axis)
    low_pwr_mask = ref_pwr < 1e-20
    if xp.any(low_pwr_mask):
        logger.warning("Reference signal power near zero in one or more channels.")

    error_power = xp.mean(xp.abs(rx - tx) ** 2, axis=axis)
    # The reference power is 1 after normalization.
    evm_percent = xp.sqrt(error_power) * 100.0
    evm_percent = xp.where(low_pwr_mask, xp.inf, evm_percent)
    out = _host(evm_percent)
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
    """
    Data-aided SNR in dB: signal power over error power.

    ``SNR = 1 / E[|r - s|^2]`` with the received symbols ``r`` and the
    reference ``s`` both normalized to unit average power per channel.

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

    if not _is_normalized(rx, axis, xp):
        rx = normalize(rx, axis=axis, mode="average_power")
    if not _is_normalized(tx, axis, xp):
        tx = normalize(tx, axis=axis, mode="average_power")

    ref_pwr = xp.mean(xp.abs(tx) ** 2, axis=axis)
    noise_power = xp.mean(xp.abs(rx - tx) ** 2, axis=axis)
    with np.errstate(divide="ignore"):
        snr_db = linear_to_db(1.0 / noise_power, power=True)
    # No error: +inf even for a zero reference; error but no reference: -inf.
    zero_noise = noise_power < 1e-20
    snr_db = xp.where(zero_noise, xp.inf, snr_db)
    snr_db = xp.where((ref_pwr < 1e-20) & ~zero_noise, -xp.inf, snr_db)
    out = _host(snr_db)
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
    """
    Bit error rate.

    Parameters
    ----------
    bits : array_like or Signal
        Received bits (0/1), ``(N_bits,)`` or ``(C, N_bits)``.  A Signal holds
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
    out = _host(errors / total)
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
    """
    Symbol error rate of minimum-distance hard decisions.

    The received and the reference symbols are both decided against the
    constellation, so rounding in the reference does not count.  SER does
    not depend on the bit labelling.

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
    points = xp.asarray(_constellation(adapter, constellation, name).points)
    rx, tx = _paired(rx, _reference(adapter, reference, "symbols", name), xp, name)
    rx = _skip(rx, num_skip_symbols, name)
    tx = _skip(tx, num_skip_symbols, name)

    from .mapping.gray import _nearest_index

    # Chunked over N to bound the (N, M) distance matrix.
    dec_rx = _nearest_index(rx.reshape(-1), points).reshape(rx.shape)
    dec_tx = _nearest_index(tx.reshape(-1), points).reshape(tx.shape)
    errors = xp.sum(dec_rx != dec_tx, axis=-1)
    total = rx.shape[-1]
    out = _host(errors / total)
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
    r"""
    Generalized mutual information (bit-metric decoding rate), bits/symbol.

    .. math::

        \mathrm{GMI} = H(X) - \sum_{b=0}^{k-1}
            \mathbb{E}\big[\log_2(1 + e^{-(1 - 2 c_b) L_b})\big]

    with the entropy ``H(X)`` of the constellation's prior (``k`` bits when
    uniform), the LLR ``L_b`` of bit ``b`` (positive favours 0) and the
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
        A Signal holds received *symbols* at one sample per symbol: their LLRs
        are computed with ``noise_var`` and ``method`` and compared with
        ``reference.bits``.
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
    llr_arr = _skip(llr_arr, num_skip_symbols * k, name).astype(xp.float64)
    bits_arr = _skip(ref, num_skip_symbols * k, name).astype(xp.float64)

    # -(1 - 2c) L: negative (small loss) when the LLR favours the sent bit.
    x = -llr_arr * (1.0 - 2.0 * bits_arr)
    # log2(1 + e^x) = (log1p(e^-|x|) + max(0, x)) / ln 2
    softplus = (xp.log1p(xp.exp(-xp.abs(x))) + xp.maximum(0.0, x)) / np.log(2.0)
    # H(X) - sum_b E[...], written so that H(X)/k is exactly 1.0 when uniform.
    out = _host(k * (c.entropy / k - xp.mean(softplus, axis=-1)))
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
    r"""
    Mutual information of the AWGN channel with the constellation's prior.

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
        symbol.
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
    rx = _skip(rx, num_skip_symbols, name).astype(xp.complex128)
    points = xp.asarray(c.points, dtype=xp.complex128)
    if c.pmf is not None:
        pmf = np.asarray(c.pmf, dtype=np.float64).clip(1e-300, None)
        log_prior = xp.asarray(np.log(pmf))
    else:
        log_prior = xp.full(points.size, -np.log(points.size), dtype=xp.float64)
    h_x = c.entropy

    # Chunk over N to bound the (chunk, M) intermediates (~4 GB at N=1e6,
    # M=256 in one piece); the sums accumulate on the device.
    rows = rx.reshape(-1, rx.shape[-1])
    n_sym = rows.shape[-1]
    chunk = 65536
    acc = xp.zeros(rows.shape[0], dtype=xp.float64)
    for n0 in range(0, n_sym, chunk):
        diff = rows[:, n0 : n0 + chunk, None] - points  # (C, chunk, M)
        log_joint = log_prior - (diff.real**2 + diff.imag**2) / noise_var
        shift = log_joint.max(axis=-1, keepdims=True)
        log_sum = xp.log(xp.sum(xp.exp(log_joint - shift), axis=-1)) + shift[..., 0]
        log_post = log_joint - log_sum[..., None]  # log p(s_m | r), nats
        acc += xp.sum(xp.exp(log_post) * log_post, axis=(-2, -1))
    mi_bits = np.clip(h_x + to_device(acc, "cpu") / n_sym / np.log(2.0), 0.0, h_x)
    out = _host(mi_bits.reshape(rx.shape[:-1]))
    if isinstance(out, float):
        logger.info("MI: %.4f b/symbol (max %.2f)", out, h_x)
    else:
        _log_per_channel("MI Ch%s: %.4f b/symbol", out)
    return out
