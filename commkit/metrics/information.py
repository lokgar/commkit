"""Achievable information rates: :func:`mi`, :func:`gmi` and :func:`ngmi`.

Symbols with a reference are put on the constellation's scale by the
data-aided gain; ``noise_var`` is on that scale.  The rates are at most the
constellation's entropy ``H(X)``.
"""

import math

import numpy as np

from .._array import to_report_scalar
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..logger import logger
from ..mapping import Constellation
from ._common import (
    _constellation,
    _log_per_channel,
    _paired,
    _reference,
    _reference_scaled,
    _skip,
)

__all__ = ["gmi", "mi", "ngmi"]


# Elements of each (C, chunk, M) intermediate of mi().
_MI_ELEMENTS = 1 << 24


def _gmi_loss(
    llrs: ArrayType | Signal,
    reference: ArrayType | None,
    constellation: Constellation | None,
    noise_var: float | None,
    method: str,
    num_skip_symbols: int,
    name: str,
) -> tuple[ArrayType, Constellation]:
    """``E[log2(1 + e^{-(1 - 2c) L})]`` per channel, averaged over the bits.

    Behind :func:`gmi` (``GMI = H(X) - k * loss``) and :func:`ngmi`
    (``NGMI = 1 - loss``); returns the loss on the device, ``(...)``, and
    the constellation.
    """
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
    return xp.mean(softplus, axis=-1, dtype=xp.float64), c


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
    loss, c = _gmi_loss(
        llrs, reference, constellation, noise_var, method, num_skip_symbols, "gmi()"
    )
    # H(X) - sum_b E[...], written so that H(X)/k is exactly 1.0 when uniform.
    k = c.bits_per_symbol
    out = to_report_scalar(k * (c.entropy / k - loss))
    if isinstance(out, float):
        logger.info("GMI: %.4f b/symbol", out)
    else:
        _log_per_channel("GMI Ch%s: %.4f b/symbol", out)
    return out


def ngmi(
    llrs: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    constellation: Constellation | None = None,
    noise_var: float | None = None,
    method: str = "maxlog",
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    r"""Normalized generalized mutual information, the soft-FEC threshold metric.

    .. math::

        \mathrm{NGMI} = 1 - \frac{H(X) - \mathrm{GMI}}{m}
            = 1 - \frac{1}{m} \sum_{b=0}^{m-1}
            \mathbb{E}\big[\log_2(1 + e^{-(1 - 2 c_b) L_b})\big]

    for ``m`` bits per symbol: 1 for error-free LLRs, and the same for
    uniform and shaped constellations at equal LLR quality.  A soft-decision
    FEC of rate ``R_c`` decodes error-free above an NGMI threshold close to
    ``R_c`` (the threshold of a given code is published with it).

    Parameters
    ----------
    llrs : array_like or Signal
        As for :func:`gmi`.
    reference : array_like, optional
        Transmitted bits, same shape as the LLRs.  Required for arrays.
    constellation : Constellation, optional
        As for :func:`gmi`.
    noise_var : float, optional
        As for :func:`gmi`: for a Signal only.
    method : {"maxlog", "exact"}, default "maxlog"
        LLR computation for a Signal.
    num_skip_symbols : int, default 0
        Leading symbols left out (``m`` LLRs each).

    Returns
    -------
    float or ndarray
        NGMI, at most 1: a float for 1-D input, ``(C,)`` for 2-D input.

    Raises
    ------
    ValueError
        As :func:`gmi`.

    See Also
    --------
    gmi : The unnormalized rate in bits per symbol.
    """
    loss, _ = _gmi_loss(
        llrs, reference, constellation, noise_var, method, num_skip_symbols, "ngmi()"
    )
    out = to_report_scalar(1.0 - loss)
    if isinstance(out, float):
        logger.info("NGMI: %.4f", out)
    else:
        _log_per_channel("NGMI Ch%s: %.4f", out)
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
    # float64: single precision changed no MI by more than 5e-8 bit (QAM 4
    # to 256, Es/N0 -5 to 30 dB), far below the Monte Carlo error.
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

    # Chunk over N so that each (C, chunk, M) intermediate holds at most 2^24
    # elements (128 MB for the complex differences), whatever C and M, and
    # at most 65536 symbols (larger chunks measured slower); the sums
    # accumulate on the device.
    rows = rx.reshape(-1, rx.shape[-1])
    n_sym = rows.shape[-1]
    chunk = max(1, min(65536, _MI_ELEMENTS // (rows.shape[0] * points.size)))
    acc = xp.zeros(rows.shape[0], dtype=xp.float64)
    for n0 in range(0, n_sym, chunk):
        diff = rows[:, n0 : n0 + chunk, None] - points  # (C, chunk, M)
        log_joint = log_prior - (diff.real**2 + diff.imag**2) * inv_nv
        # With d = log_joint - max and u = exp(d), the posterior is u / Z
        # (Z = sum u), and sum_m p log p = sum(u d) / Z - log Z: one exp.
        d = log_joint - log_joint.max(axis=-1, keepdims=True)
        u = xp.exp(d)
        z = xp.sum(u, axis=-1)
        neg_entropy = xp.sum(u * d, axis=-1) / z - xp.log(z)  # nats, (C, chunk)
        acc += xp.sum(neg_entropy, axis=-1, dtype=xp.float64)
    mi_bits = np.clip(h_x + to_device(acc, "cpu") / n_sym / np.log(2.0), 0.0, h_x)
    out = to_report_scalar(mi_bits.reshape(rx.shape[:-1]))
    if isinstance(out, float):
        logger.info("MI: %.4f b/symbol (max %.2f)", out, h_x)
    else:
        _log_per_channel("MI Ch%s: %.4f b/symbol", out)
    return out
