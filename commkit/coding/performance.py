"""Error-rate curves of a code over AWGN: :func:`error_rate_curve` and :class:`ErrorRateCurve`.

:func:`error_rate_curve` runs the coded link at each Es/N0 of a grid
(random information bits, :func:`encode`, ``Constellation.map``, AWGN,
``Constellation.llr`` and :func:`decode`) in batches until a number of
block errors or of blocks, and counts the bit, block and pre-FEC bit
errors and the decoding outcome.  :class:`ErrorRateCurve` holds those
counts, also from your own simulation loop, for
:func:`commkit.plotting.plot_error_rate`.  Like the EXIT curves of
:mod:`commkit.coding.exit`, it draws the bits on the host and the noise on
the device, through the private block functions (one INFO line per call).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _random
from ..backend import _module_for, to_device
from ..logger import logger
from ..mapping import Constellation
from .base import _checked, _decode_blocks, _encode_blocks, _resolve_method
from .exit import _check_device

__all__ = ["ErrorRateCurve", "error_rate_curve"]


# Transmitted bits per batch of error_rate_curve.
_BATCH_BITS = 1 << 18


@dataclass(frozen=True, eq=False)
class ErrorRateCurve:
    """Decoding error counts of a code against Es/N0, one entry per point.

    Built by :func:`error_rate_curve`, or from the counts of your own
    simulation loop over any channel, to plot with
    :func:`commkit.plotting.plot_error_rate`.

    Parameters
    ----------
    esn0_db : array_like
        Es/N0 of each point, in dB, ``(P,)``.
    num_blocks : array_like
        Code blocks decoded at each point, ``>= 1``.
    bit_errors : array_like
        Information bits decoded in error.
    block_errors : array_like
        Blocks with any information bit in error.
    pre_fec_bit_errors : array_like
        Transmitted bits whose channel LLR has the wrong sign.
    detected_failures : array_like
        Blocks the decoder flagged (``DecodeResult.valid`` false), whatever
        their bits.
    undetected_errors : array_like
        Blocks in error that the decoder reported valid.  Every block error
        when ``checked`` is ``False``.
    total_iterations : array_like
        Decoder iterations summed over the blocks.
    k, n : int
        Information and transmitted bits per block.
    bits_per_symbol : int
        Bits per constellation symbol, for :attr:`ebn0_db`.
    checked : bool, default True
        Whether the decoder has a success check (``DecodeResult.checked``),
        which splits failures into detected and undetected ones.
    label : str, default ""
        What the curve describes, for legends.

    Notes
    -----
    Counts are stored as read-only int64 host copies and ``esn0_db`` as
    float64.  The rates are properties; ``pre_fec_ber`` counts the bits on
    the air, ``ber`` the information bits.
    """

    esn0_db: np.ndarray
    num_blocks: np.ndarray
    bit_errors: np.ndarray
    block_errors: np.ndarray
    pre_fec_bit_errors: np.ndarray
    detected_failures: np.ndarray
    undetected_errors: np.ndarray
    total_iterations: np.ndarray
    k: int
    n: int
    bits_per_symbol: int
    checked: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        esn0 = np.array(to_device(self.esn0_db, "cpu"), dtype=np.float64)
        if esn0.ndim != 1 or esn0.size == 0 or not np.all(np.isfinite(esn0)):
            raise ValueError(
                f"ErrorRateCurve: esn0_db must be 1-D, finite and not empty, got "
                f"shape {esn0.shape}."
            )
        esn0.setflags(write=False)
        object.__setattr__(self, "esn0_db", esn0)
        for name in _COUNT_FIELDS:
            value = np.array(to_device(getattr(self, name), "cpu"))
            if value.shape != esn0.shape:
                raise ValueError(
                    f"ErrorRateCurve: {name} must have the shape of esn0_db "
                    f"{esn0.shape}, got {value.shape}."
                )
            if value.dtype.kind not in "iu" or np.any(value < 0):
                raise ValueError(f"ErrorRateCurve: {name} must be counts >= 0.")
            value = value.astype(np.int64)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.any(self.num_blocks < 1):
            raise ValueError("ErrorRateCurve: num_blocks must be >= 1 at every point.")
        for name in ("k", "n", "bits_per_symbol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | np.integer):
                raise ValueError(f"ErrorRateCurve: {name} must be an integer.")
            if value < 1:
                raise ValueError(f"ErrorRateCurve: {name} must be >= 1, got {value}.")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "checked", bool(self.checked))

    @property
    def rate(self) -> float:
        """Code rate ``k / n``."""
        return self.k / self.n

    @property
    def ebn0_db(self) -> np.ndarray:
        """Eb/N0 of each point in dB: Es/N0 per information bit,
        ``esn0_db - 10 log10(rate * bits_per_symbol)``."""
        return self.esn0_db - 10 * np.log10(self.rate * self.bits_per_symbol)

    @property
    def ber(self) -> np.ndarray:
        """Post-FEC bit error rate of the information bits."""
        return self.bit_errors / (self.num_blocks * self.k)

    @property
    def fer(self) -> np.ndarray:
        """Block (frame) error rate."""
        return self.block_errors / self.num_blocks

    @property
    def pre_fec_ber(self) -> np.ndarray:
        """Bit error rate of the channel's hard decisions, before decoding."""
        return self.pre_fec_bit_errors / (self.num_blocks * self.n)

    @property
    def mean_iterations(self) -> np.ndarray:
        """Decoder iterations per block."""
        return self.total_iterations / self.num_blocks


_COUNT_FIELDS = (
    "num_blocks",
    "bit_errors",
    "block_errors",
    "pre_fec_bit_errors",
    "detected_failures",
    "undetected_errors",
    "total_iterations",
)


def error_rate_curve(
    code: Any,
    *,
    constellation: Constellation,
    esn0_db: Any,
    method: Any = None,
    min_block_errors: int = 100,
    max_blocks: int = 10_000,
    label: str | None = None,
    rng: int | np.random.Generator | None = None,
    device: str = "cpu",
) -> ErrorRateCurve:
    """Measure a code's error rates over AWGN by Monte Carlo.

    At each Es/N0, random information bits are encoded, mapped onto the
    constellation, sent through additive white Gaussian noise at one sample
    per symbol, demapped by :func:`commkit.mapping.compute_llr` and decoded
    by :func:`commkit.coding.decode`, in batches, until ``min_block_errors``
    blocks are in error or ``max_blocks`` blocks are decoded.

    Parameters
    ----------
    code : code object
        Any code: a FEC code, a distribution matcher (:class:`CCDM`) or
        :class:`PAS`.
    constellation : Constellation
        The modulation: uniform for a FEC code, the code's own constellation
        for a matcher.  Es is its average energy (1 for the factory
        constellations).
    esn0_db : array_like
        Es/N0 of each point, in dB, ``(P,)``.
    method : decoder object, optional
        The decoder; ``None`` takes the code's default.  The curve needs no
        soft output: ``Chase(..., soft=False)`` and
        ``MaximumLikelihood(soft=False)`` decode the same bits faster.
    min_block_errors : int, default 100
        Block errors after which a point stops.  The relative standard
        error of the frame error rate is then about ``1/sqrt(min_block_errors)``.
    max_blocks : int, default 10000
        Blocks after which a point stops, whatever its errors.
    label : str, optional
        Label of the curve; ``None`` names the decoder and the code.
    rng : int or numpy.random.Generator, optional
        Seed or Generator of the information bits (drawn on the host) and
        the noise (drawn on ``device``).
    device : {"cpu", "gpu"}, default "cpu"
        Where the blocks are encoded, mapped and decoded.

    Returns
    -------
    ErrorRateCurve
        The counts at each point, with the code's ``k`` and ``n``.

    Raises
    ------
    TypeError
        If ``code`` is not a code object, ``method`` does not decode it, or
        ``constellation`` is not a :class:`~commkit.mapping.Constellation`.
    ValueError
        If ``esn0_db`` is empty or not finite, ``min_block_errors`` or
        ``max_blocks`` is below 1, ``device`` is not ``"cpu"`` or ``"gpu"``,
        or the constellation does not suit the code (shaped with a FEC code,
        or another than a matcher's own).

    See Also
    --------
    commkit.plotting.plot_error_rate : Plots the curve.
    commkit.metrics.ber, commkit.metrics.fer : The rates of one decoded record.

    Notes
    -----
    A batch holds about 2^18 transmitted bits (at least one block).  A
    point stops after the batch that reaches either limit: it may count
    more than ``min_block_errors`` block errors, but the last batch is cut
    to ``max_blocks``.  The noise is what
    :func:`~commkit.impairments.apply_awgn` adds at ``esn0_db``: circular
    complex of variance ``N0 = Es / (Es/N0)`` for a complex constellation,
    real of variance ``N0 / 2`` for a real one (PAM), so BPSK has
    ``BER = Q(sqrt(2 Es/N0))``.  The LLRs use that variance, as
    :func:`~commkit.mapping.compute_llr` does, and the constellation's
    prior.  When the record
    does not fill whole symbols, the last symbol carries random filler bits.
    """
    name = "error_rate_curve()"
    method = _resolve_method(code, method, name)
    if not isinstance(constellation, Constellation):
        raise TypeError(
            f"{name}: constellation must be a Constellation, got "
            f"{type(constellation).__name__}."
        )
    matched = getattr(code, "constellation", None)
    if matched is not None and matched != constellation:
        raise ValueError(
            f"{name}: {code!r} matches onto another constellation than "
            f"{constellation!r}."
        )
    if constellation.pmf is not None and matched is None:
        raise ValueError(
            f"{name}: a shaped constellation with a FEC code needs a distribution "
            "matcher: use code=PAS(code, constellation)."
        )
    grid = np.asarray(to_device(esn0_db, "cpu"), dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError(f"{name}: esn0_db must be 1-D, finite and not empty.")
    for arg, value in (
        ("min_block_errors", min_block_errors),
        ("max_blocks", max_blocks),
    ):
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise ValueError(f"{name}: {arg} must be an integer, got {value!r}.")
        if value < 1:
            raise ValueError(f"{name}: {arg} must be >= 1, got {value}.")
    device = _check_device(device, name)
    xp = _module_for(device)
    gen = _random.as_generator(rng)
    k, n, m = int(code.k), int(code.n), constellation.bits_per_symbol
    es = constellation.power()
    batch = max(1, -(-_BATCH_BITS // n))
    if label is None:
        label = f"{type(method).__name__} on {code!r}"
    logger.debug(
        "Measuring the error rates of %s on %r at %d point(s) on %s: up to %d "
        "blocks (batches of %d) or %d block errors per point.",
        label,
        constellation,
        grid.size,
        device,
        max_blocks,
        batch,
        min_block_errors,
    )

    counts = np.zeros((len(_COUNT_FIELDS), grid.size), dtype=np.int64)
    for i, esn0 in enumerate(grid):
        n0 = es / 10 ** (esn0 / 10)
        # As apply_awgn: real samples carry the in-phase half of the noise.
        noise_var = n0 if constellation.is_complex else n0 / 2
        while counts[0, i] < max_blocks and counts[2, i] < min_block_errors:
            b = int(min(batch, max_blocks - counts[0, i]))
            info = to_device(gen.integers(0, 2, (b, k), dtype=np.int8), device)
            bits = _encode_blocks(code, info).reshape(-1)
            filler = -bits.size % m
            if filler:
                extra = to_device(gen.integers(0, 2, filler, dtype=np.int8), device)
                bits = xp.concatenate([bits, extra])
            symbols = constellation.map(bits)
            real = symbols.real.dtype
            if xp.iscomplexobj(symbols):
                std = float(np.sqrt(noise_var / 2))
                noise = _random.complex_normal(
                    gen, symbols.shape, std=std, dtype=real, xp=xp
                )
            else:
                std = float(np.sqrt(noise_var))
                noise = std * _random.standard_normal(
                    gen, symbols.shape, dtype=real, xp=xp
                )
            llrs = constellation.llr(symbols + noise, noise_var=float(noise_var))
            llrs = llrs[: b * n].reshape(b, n)
            res = _decode_blocks(code, llrs, method)
            wrong = res.bits != info
            block_wrong = wrong.any(axis=1)
            batch_counts = xp.stack(
                [
                    xp.asarray(b, dtype=xp.int64),
                    xp.count_nonzero(wrong),
                    xp.count_nonzero(block_wrong),
                    xp.count_nonzero((llrs < 0) != bits[: b * n].reshape(b, n)),
                    xp.count_nonzero(~res.valid),
                    xp.count_nonzero(block_wrong & res.valid),
                    xp.sum(res.iterations, dtype=xp.int64),
                ]
            )
            counts[:, i] += to_device(batch_counts, "cpu").astype(np.int64)
        logger.debug(
            "  Es/N0 %.2f dB: %d block errors in %d blocks, %d bit errors.",
            esn0,
            counts[2, i],
            counts[0, i],
            counts[1, i],
        )
    curve = ErrorRateCurve(
        grid,
        **dict(zip(_COUNT_FIELDS, counts, strict=True)),
        k=k,
        n=n,
        bits_per_symbol=m,
        checked=_checked(code, method),
        label=label,
    )
    logger.info(
        "Error rates of %s: %s.",
        label,
        "; ".join(
            f"{e:.2f} dB BER {b:.2e} FER {f:.2e} ({nb} blocks)"
            for e, b, f, nb in zip(
                grid, curve.ber, curve.fer, curve.num_blocks, strict=True
            )
        ),
    )
    return curve
