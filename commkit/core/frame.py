"""Frame containers: preambles and single-carrier frames.

A :class:`SingleCarrierFrame` describes a transmitted frame (preamble,
payload, pilots, guard intervals, MIMO channels) and builds it into a
:class:`~commkit.core.Signal` with ``to_signal``; :func:`extract_payload`
cuts the payload out of a frame Signal at one sample per symbol.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .._circular import convolve_periodic
from .._sequences import barker_sequence, zadoff_chu_sequence, zc_mimo_root
from ..backend import ArrayType, dispatch, to_device
from ..filtering.pulses import Pulse
from ..mapping import Constellation
from ..math import db_to_linear, normalize
from . import generation
from ._signal_adapter import _same_fact, require_device, require_integer_sps
from .signal import Reference, Signal

__all__ = ["Preamble", "SingleCarrierFrame", "extract_payload"]


@dataclass(frozen=True, kw_only=True)
class Preamble:
    """Frame-synchronization preamble: a Barker or Zadoff-Chu sequence.

    The symbols are built from the fields on construction (host complex64,
    ``(num_symbols,)``, or ``(num_channels, num_symbols)`` for several channels); they
    cannot be set by hand.

    Parameters
    ----------
    sequence_type : {"barker", "zc"}, default "barker"
        The sequence family.
    num_symbols : int
        Length in symbols.  Barker: one of 2, 3, 4, 5, 7, 11, 13.  ZC: any
        length; a prime gives the ideal cross-correlation between roots.
    root : int, default 1
        ZC root index ``u``, in ``[1, num_symbols)`` and coprime to ``num_symbols``;
        ignored for Barker.
    num_channels : int, default 1
        Number of channels.  ZC preambles get a distinct root per channel
        (``zc_mimo_root``); a Barker preamble repeats the same sequence.
    """

    sequence_type: str = "barker"
    num_symbols: int
    # ZC root index, 1 <= root < num_symbols; ignored for Barker sequences.
    root: int = 1
    # Number of channels.  ZC preambles get a unique root per channel
    # (zc_mimo_root); Barker broadcasts the same sequence.
    num_channels: int = 1

    # Generated in __post_init__ from the fields above.
    _symbols: Any = field(default=None, init=False, repr=False, compare=False)

    # -------------------------------------------------------------------------
    # Validators and Post-Initialization Hooks
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate the fields and generate the preamble symbols.

        This ensures that standard sequences are generated correctly according
        to the requested sequence properties.

        For ``num_channels == 1`` the internal ``_symbols`` shape is ``(num_symbols,)``.
        For ``num_channels > 1`` it becomes ``(num_channels, num_symbols)``:
        - ZC: each row uses the unique root from ``zc_mimo_root``.
        - Barker: the same sequence is tiled across all channels.
        """
        if self.sequence_type not in ("barker", "zc"):
            raise ValueError(
                f"sequence_type must be 'barker' or 'zc', got {self.sequence_type!r}."
            )
        _check_int("num_symbols", self.num_symbols, 1)
        _check_int("root", self.root, 1)
        _check_int("num_channels", self.num_channels, 1)

        symbols: np.ndarray
        if self.sequence_type == "barker":
            # Barker symbols (-1, +1), the same on every channel.
            symbols = barker_sequence(self.num_symbols)
            if self.num_channels > 1:
                symbols = np.tile(symbols[None, :], (self.num_channels, 1))
        elif self.num_channels > 1:
            # One Zadoff-Chu root per channel, (num_channels, num_symbols).
            symbols = np.stack(
                [
                    zadoff_chu_sequence(
                        self.num_symbols,
                        root=zc_mimo_root(k, self.root, self.num_symbols),
                    )
                    for k in range(self.num_channels)
                ]
            )
        else:
            symbols = zadoff_chu_sequence(self.num_symbols, root=self.root)

        # Consistent internal dtype; sequences stay on the CPU (no hidden
        # device placement).
        object.__setattr__(self, "_symbols", symbols.astype("complex64"))

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def symbols(self) -> Any:
        """The preamble symbols (host complex64).

        ``(num_symbols,)``, or ``(num_channels, num_symbols)`` for several channels.
        """
        return self._symbols

    # -------------------------------------------------------------------------
    # Signal Generation
    # -------------------------------------------------------------------------

    def to_signal(
        self,
        *,
        symbol_rate: float,
        sps: int = 1,
        pulse: Pulse | ArrayType | None = None,
        periodic: bool = False,
        device: str = "cpu",
    ) -> Signal:
        """Shape the preamble into a Signal.

        Parameters
        ----------
        symbol_rate : float
            Symbol rate in Hz.
        sps : int, default 1
            Samples per symbol.
        pulse : Pulse or array_like, optional
            Pulse object or taps; ``None`` zero-stuffs without shaping (as
            :func:`commkit.generate`).
        periodic : bool, default False
            Shape the sequence as one period of a periodic sequence (circular
            convolution), so the waveform loops seamlessly.
        device : {"cpu", "gpu"}, default "cpu"
            Where the waveform is shaped and returned.

        Returns
        -------
        Signal
            The shaped preamble, normalized per channel to unit symbol power,
            without constellation or reference.
        """
        sps = require_integer_sps(sps, "Preamble.to_signal()")
        device = require_device(device, "Preamble.to_signal()")
        symbols = to_device(self.symbols, device)
        return Signal(
            samples=generation.shape_pulse(
                symbols, sps=sps, pulse=pulse, periodic=periodic
            ),
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            pulse=pulse if isinstance(pulse, Pulse) else None,
        )


def _qpsk() -> Constellation:
    return Constellation.psk(4)


@dataclass(frozen=True, kw_only=True)
class SingleCarrierFrame:
    """Single-carrier frame: preamble, payload with pilots, and guard interval.

    The fields describe the layout and the seeds of the data; payload and
    pilot symbols are drawn from them when first used, and
    :meth:`to_signal` builds the waveform.  Pilots serve channel estimation
    and pilot-aided recovery, the guard interval multipath.

    Parameters
    ----------
    payload_num_symbols : int, optional
        Number of payload symbols per channel; ``None`` takes the symbols of
        ``payload_blocks``, or 1000 without it.  With pilots it must fill
        whole pilot periods (comb: a multiple of ``pilot_period - 1``; block:
        of ``pilot_period - pilot_block_num_symbols``); otherwise construction raises.
    payload_constellation : Constellation, default ``Constellation.psk(4)``
        Payload constellation.  A shaped one
        (``Constellation.qam(64).shaped(entropy=5)``) gives a PS payload.
    payload_rng : int or numpy.random.Generator, optional
        Seed or generator of the payload data, as ``generate(rng=)``;
        ``None`` draws a fresh payload.  The frame keeps the integer seed it
        used (a generator gives one draw), so a frame, its
        ``dataclasses.replace`` copies and a saved and loaded frame
        rebuild the same payload.
    payload_code : code object, optional
        Code of the payload (:mod:`commkit.coding`), passed to
        :func:`generate` as ``code``: a FEC code with a uniform payload
        constellation, a distribution matcher (``CCDM``) or ``PAS`` with the
        shaped one it matches onto.  ``payload_num_symbols`` times the bits per
        symbol must be a multiple of ``payload_code.n`` (whole blocks),
        unless ``payload_pad`` is set.
    payload_blocks : int, optional
        Code blocks (codewords) per channel, instead of ``payload_num_symbols``: the
        payload is the symbols they fill, which must be whole symbols and
        whole pilot periods (the error names the nearest valid counts).
    payload_pad : bool, default False
        With ``payload_code`` and ``payload_num_symbols``: fill the payload symbols
        after the last whole codeword with random uncoded bits
        (``generate``'s ``pad``), so the payload length can follow an
        instrument's record length instead of the code.  The payload
        reference's ``num_blocks`` counts the codewords.  Not with a
        distribution matcher.
    preamble : Preamble, optional
        Synchronization preamble, before the body.  A multi-channel preamble
        must have ``num_channels`` channels; a one-channel preamble is repeated
        on every channel.
    pilot_pattern : {"none", "block", "comb"}, default "none"
        ``"none"``: no pilots.  ``"comb"``: one pilot every
        ``pilot_period`` symbols, the first at the start of the body.
        ``"block"``: ``pilot_block_num_symbols`` pilots at the start of every period
        of ``pilot_period`` symbols.
    pilot_period : int, default 0
        Pilot period in symbols (pilots included); ``> 1`` for ``"comb"``,
        ``> pilot_block_num_symbols`` for ``"block"``.
    pilot_block_num_symbols : int, default 0
        Pilots per period for ``"block"``.
    pilot_seed : int, default 1337
        Seed of the pilot symbols: pilots are a known sequence, the same
        in every frame by default.
    pilot_constellation : Constellation, default ``Constellation.psk(4)``
        Pilot constellation; must be unshaped.
    pilot_gain_db : float, default 0.0
        Pilot power boost in dB relative to the payload.
    guard_type : {"zero", "cp"}, default "zero"
        ``"zero"``: zeros after the frame.  ``"cp"``: a cyclic prefix, the
        last ``guard_num_symbols`` symbols of the frame repeated before it.
    guard_num_symbols : int, default 0
        Guard interval in symbols.
    num_channels : int, default 1
        Number of independent channels (MIMO streams), each with its own payload and
        pilots.

    Notes
    -----
    The layout (``get_structure_map``, the pilot mask) depends only on the
    fields and never generates data.  Payload and pilot symbols are generated
    on first access and cached.

    **Sizing a frame.**  :attr:`num_symbols` is the frame's length in
    symbols (preamble, body and guard), so ``to_signal`` gives
    ``num_symbols * sps`` samples per channel.  For a payload of whole
    codewords, give ``payload_blocks``; a ``payload_num_symbols`` that does not
    fill whole codewords raises, naming the nearest valid lengths, pilot
    periods included.  To fit an instrument's record (an
    arbitrary waveform generator's memory granularity, with
    ``to_signal(periodic=True)``), choose ``payload_num_symbols`` from the target
    length and set ``payload_pad=True``.
    """

    # None until __post_init__ resolves it (1000, or from payload_blocks).
    payload_num_symbols: int = None  # type: ignore[assignment]
    payload_constellation: Constellation = field(default_factory=_qpsk)
    payload_rng: int | np.random.Generator | None = None
    payload_code: Any = None
    payload_blocks: int | None = None
    payload_pad: bool = False

    preamble: Preamble | None = None

    pilot_pattern: str = "none"
    pilot_period: int = 0
    pilot_block_num_symbols: int = 0
    pilot_constellation: Constellation = field(default_factory=_qpsk)
    pilot_seed: int = 1337
    pilot_gain_db: float = 0.0

    guard_type: str = "zero"
    guard_num_symbols: int = 0

    num_channels: int = 1

    # Lazily generated payload and pilot data.  The frame's fields are frozen;
    # this cache is filled on first access and never changes afterwards.
    _cache: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # -------------------------------------------------------------------------
    # Validators and Post-Initialization Hooks
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate the fields and resolve the payload length.

        The payload fills whole pilot periods, and with a ``payload_code``
        whole codewords unless ``payload_pad`` is set; ``payload_blocks``
        sets ``payload_num_symbols`` to the symbols its codewords fill.  Both are
        accepted when they agree, as ``dataclasses.replace`` and
        :func:`~commkit.io.load_npz` pass both.
        """
        object.__setattr__(self, "payload_rng", _resolve_seed(self.payload_rng))
        _check_int("pilot_seed", self.pilot_seed, 0)
        if self.payload_num_symbols is not None:
            _check_int("payload_num_symbols", self.payload_num_symbols, 1)
        if self.payload_blocks is not None:
            _check_int("payload_blocks", self.payload_blocks, 1)
        _check_int("pilot_period", self.pilot_period, 0)
        _check_int("pilot_block_num_symbols", self.pilot_block_num_symbols, 0)
        _check_int("guard_num_symbols", self.guard_num_symbols, 0)
        _check_int("num_channels", self.num_channels, 1)
        for name in ("payload_constellation", "pilot_constellation"):
            if not isinstance(getattr(self, name), Constellation):
                raise ValueError(
                    f"{name} must be a Constellation, e.g. Constellation.qam(16)."
                )
        if self.pilot_constellation.pmf is not None:
            raise ValueError("pilot_constellation must not be shaped.")
        if self.pilot_pattern not in ("none", "block", "comb"):
            raise ValueError(
                "pilot_pattern must be 'none', 'block' or 'comb', "
                f"got {self.pilot_pattern!r}."
            )
        if self.guard_type not in ("zero", "cp"):
            raise ValueError(
                f"guard_type must be 'zero' or 'cp', got {self.guard_type!r}."
            )
        if self.preamble is not None and not isinstance(self.preamble, Preamble):
            raise ValueError("preamble must be a Preamble or None.")
        if self.preamble is not None and self.preamble.num_channels > 1:
            if self.preamble.num_channels != self.num_channels:
                raise ValueError(
                    f"preamble.num_channels={self.preamble.num_channels} does not "
                    f"match frame.num_channels={self.num_channels}"
                )

        per_period = 1  # payload symbols per pilot period (1: no constraint)
        if self.pilot_pattern != "none":
            block = self._pilot_block()
            if block < 1:
                raise ValueError(
                    "pilot_block_num_symbols must be >= 1 for the 'block' pattern, got "
                    f"{self.pilot_block_num_symbols}."
                )
            if self.pilot_period <= block:
                bound = (
                    "1" if self.pilot_pattern == "comb" else "pilot_block_num_symbols"
                )
                raise ValueError(
                    f"pilot_period must be > {bound} for the {self.pilot_pattern!r} "
                    f"pattern, got {self.pilot_period}."
                )
            per_period = self.pilot_period - block

        if self.payload_code is None:
            if self.payload_pad or self.payload_blocks is not None:
                raise ValueError("payload_pad and payload_blocks need a payload_code.")
        else:
            from .. import coding

            coding.base._codec(self.payload_code, "SingleCarrierFrame()")
            if self.payload_pad and getattr(self.payload_code, "constellation", None):
                raise ValueError(
                    f"payload_pad fills with uniform random bits, which would change "
                    f"the composition of {self.payload_code!r}; size the payload in "
                    "whole blocks."
                )
            if self.payload_blocks is not None and self.payload_pad:
                raise ValueError(
                    "payload_pad fills a payload_num_symbols with whole codewords and filler; "
                    "payload_blocks sizes the payload exactly, without filler.  Give "
                    "one of them."
                )
        if self.payload_num_symbols is None:
            num_symbols = (
                1000
                if self.payload_blocks is None
                else self._blocks_num_symbols(per_period)
            )
            object.__setattr__(self, "payload_num_symbols", num_symbols)
        assert self.payload_num_symbols is not None

        if self.payload_num_symbols % per_period:
            low = self.payload_num_symbols // per_period * per_period
            valid = f"{low} or {low + per_period}" if low else f"{per_period}"
            raise ValueError(
                f"payload_num_symbols={self.payload_num_symbols} does not fill whole pilot periods "
                f"({per_period} payload symbols each); use {valid}."
            )
        if self.payload_code is not None:
            self._check_codewords(per_period)

    def _blocks_num_symbols(self, per_period: int) -> int:
        """``payload_num_symbols`` of ``payload_blocks`` codewords: the symbols they fill,
        which must be whole symbols and whole pilot periods."""
        assert self.payload_code is not None and self.payload_blocks is not None
        m = self.payload_constellation.bits_per_symbol
        n = self.payload_code.n
        blocks = self.payload_blocks
        bits = blocks * n
        num_symbols = bits // m
        if bits % (m * per_period):
            unit = m * per_period  # bits of the shortest valid payload
            step = unit // math.gcd(n, unit)
            low = blocks // step * step
            valid = f"{low} or {low + step}" if low else f"{step}"
            raise ValueError(
                f"payload_blocks={blocks} codewords of {n} bits do not fill whole "
                f"payload symbols{' and pilot periods' if per_period > 1 else ''}; "
                f"use payload_blocks={valid}, or payload_num_symbols= with payload_pad=True."
            )
        return int(num_symbols)

    def _check_codewords(self, per_period: int) -> None:
        """Whole codewords in ``payload_num_symbols``, or filler with ``payload_pad``."""
        assert self.payload_code is not None and self.payload_num_symbols is not None
        m = self.payload_constellation.bits_per_symbol
        n = self.payload_code.n
        bits = self.payload_num_symbols * m
        if bits < n:
            raise ValueError(
                f"payload_num_symbols={self.payload_num_symbols} carries {bits} bits, fewer than "
                f"one codeword of payload_code.n={n}."
            )
        if self.payload_blocks is not None:
            if self.payload_blocks * n != bits:
                raise ValueError(
                    f"payload_num_symbols={self.payload_num_symbols} and payload_blocks="
                    f"{self.payload_blocks} disagree ({bits} bits against "
                    f"{self.payload_blocks * n}); give one of them."
                )
            return
        if bits % n and not self.payload_pad:
            # Whole codewords, and whole pilot periods.
            step = math.lcm(n // math.gcd(n, m), per_period)
            low = self.payload_num_symbols // step * step
            valid = f"{low} or {low + step}" if low else f"{step}"
            raise ValueError(
                f"payload_num_symbols={self.payload_num_symbols} carries {bits} bits, not a "
                f"multiple of payload_code.n={n}; use payload_num_symbols={valid}, "
                "payload_blocks= to size the payload in codewords, or "
                "payload_pad=True to fill the rest with random bits."
            )

    # -------------------------------------------------------------------------
    # Mask Generation and Internal Data Preparation Methods
    # -------------------------------------------------------------------------

    def _pilot_block(self) -> int:
        """Pilots at the start of each period: one for "comb"."""
        return 1 if self.pilot_pattern == "comb" else self.pilot_block_num_symbols

    def _generate_pilot_mask(self) -> tuple[ArrayType, int]:
        """Calculate the pilot placement mask and total frame length.

        Every period starts with its pilots; ``__post_init__`` guarantees that
        the payload fills whole periods.

        Returns
        -------
        mask : ndarray (bool)
            Boolean mask where True indicates a pilot symbol location.
        body_num_symbols : int
            Total number of symbols in the frame body (payload + pilots).
        """
        if self.pilot_pattern == "none":
            return np.zeros(
                self.payload_num_symbols, dtype=bool
            ), self.payload_num_symbols
        block = self._pilot_block()
        period = np.zeros(self.pilot_period, dtype=bool)
        period[:block] = True
        mask = np.tile(period, self.payload_num_symbols // (self.pilot_period - block))
        return mask, mask.size

    def _ensure_payload_generated(self) -> None:
        """Generate and cache the payload bits and symbols with ``generate()``."""
        if self._cache.get("payload_bits") is not None:
            return
        sig = generation.generate(
            self.payload_constellation,
            self.payload_num_symbols,
            symbol_rate=1.0,
            num_channels=self.num_channels,
            code=self.payload_code,
            pad=self.payload_pad,
            rng=self.payload_rng,
        )
        ref = sig.reference
        assert ref is not None
        self._cache["payload_bits"] = ref.bits
        self._cache["payload_info_bits"] = ref.info_bits
        # The payload_code, or the matcher generate() picked for a shaped
        # payload without one; None for uniform uncoded bits.
        self._cache["payload_reference_code"] = ref.code
        self._cache["payload_symbols"] = ref.symbols

    def _ensure_pilot_generated(self) -> None:
        """Generate and cache the pilot bits and symbols with ``generate()``.

        Pilots are always uniform: shaping would destroy the known-reference
        property required for channel estimation.
        """
        if self._cache.get("pilot_bits") is not None or self.pilot_pattern == "none":
            return
        mask, _ = self._generate_pilot_mask()
        pilot_count = int(np.sum(mask))
        sig = generation.generate(
            self.pilot_constellation,
            pilot_count,
            symbol_rate=1.0,
            num_channels=self.num_channels,
            rng=self.pilot_seed,
        )
        ref = sig.reference
        assert ref is not None
        self._cache["pilot_bits"] = ref.bits
        self._cache["pilot_symbols"] = ref.symbols

    # -------------------------------------------------------------------------
    # Properties for Accessing Payload and Pilot Data
    # -------------------------------------------------------------------------

    @property
    def payload_bits(self) -> ArrayType:
        """The payload bits (host int8).

        ``(payload_num_symbols * k,)``, or ``(num_channels, payload_num_symbols * k)`` for
        several channels.
        """
        self._ensure_payload_generated()
        return self._cache.get("payload_bits")

    @property
    def payload_info_bits(self) -> ArrayType:
        """The payload's source bits, before matching and coding (host int8).

        :attr:`payload_bits` itself (the same array) for a uniform payload
        without a code.
        """
        self._ensure_payload_generated()
        return self._cache["payload_info_bits"]

    def _payload_reference(self) -> Reference:
        """The payload's ground truth on the host (symbols and both bit streams)."""
        self._ensure_payload_generated()
        code = self._cache["payload_reference_code"]
        return Reference(
            symbols=self._cache["payload_symbols"],
            bits=self._cache["payload_bits"],
            info_bits=None if code is None else self._cache["payload_info_bits"],
            code=code,
        )

    @property
    def payload_symbols(self) -> ArrayType:
        """The payload symbols (host).

        ``(payload_num_symbols,)``, or ``(num_channels, payload_num_symbols)`` for several
        channels.
        """
        self._ensure_payload_generated()
        return self._cache.get("payload_symbols")

    @property
    def pilot_bits(self) -> ArrayType | None:
        """The pilot bits (host int8), or ``None`` without pilots."""
        if self.pilot_pattern == "none":
            return None
        self._ensure_pilot_generated()
        return self._cache.get("pilot_bits")

    @property
    def pilot_symbols(self) -> ArrayType | None:
        """The pilot symbols (host), or ``None`` without pilots.

        Without ``pilot_gain_db``; :attr:`body_symbols` carries the boost.
        """
        if self.pilot_pattern == "none":
            return None
        self._ensure_pilot_generated()
        return self._cache.get("pilot_symbols")

    @property
    def body_symbols(self) -> ArrayType:
        """The body: payload and pilot symbols interleaved (host complex64).

        The pilots carry ``pilot_gain_db``, so they differ from
        :attr:`pilot_symbols` when it is not zero.  ``(L,)`` or
        ``(num_channels, L)`` with ``L`` the body length.
        """
        mask, body_num_symbols = self._generate_pilot_mask()
        channels = (self.num_channels,) if self.num_channels > 1 else ()
        body = np.zeros((*channels, body_num_symbols), dtype="complex64")
        if self.pilot_pattern != "none":
            pilot_symbols = self.pilot_symbols
            assert pilot_symbols is not None
            if self.pilot_gain_db != 0.0:  # pilot boost, dB to linear amplitude
                pilot_symbols = pilot_symbols * db_to_linear(
                    self.pilot_gain_db, power=False
                )
            body[..., mask] = pilot_symbols
        body[..., ~mask] = self.payload_symbols
        return body

    # -------------------------------------------------------------------------
    # Frame Structure Mapping
    # -------------------------------------------------------------------------

    @property
    def num_symbols(self) -> int:
        """Frame length in symbols per channel: preamble, body and guard.

        ``to_signal(sps=sps)`` gives ``num_symbols * sps`` samples per
        channel.
        """
        preamble_num_symbols = self.preamble.num_symbols if self.preamble else 0
        return (
            preamble_num_symbols
            + self._generate_pilot_mask()[1]
            + self.guard_num_symbols
        )

    def get_structure_map(
        self,
        unit: Literal["symbols", "samples"] = "symbols",
        sps: int = 1,
        include_preamble: bool = True,
    ) -> dict[str, ArrayType]:
        """Return boolean masks of the frame's segments (host arrays).

        Parameters
        ----------
        unit : {"symbols", "samples"}, default "symbols"
            Resolution of the masks: one entry per symbol, or per sample.
        sps : int, default 1
            Samples per symbol, for ``unit="samples"``.
        include_preamble : bool, default True
            ``True``: masks over the whole frame, preamble and guard
            included.  ``False``: masks over what follows the preamble (with
            a cyclic prefix, the body alone).

        Returns
        -------
        dict of str to ndarray
            ``"pilots"`` and ``"payload"``; ``"preamble"`` with
            ``include_preamble=True``; ``"guard"`` with
            ``include_preamble=True`` or a zero guard.
        """
        xp = np
        if unit == "samples":
            sps = require_integer_sps(sps, "get_structure_map()")
        mask, body_num_symbols = self._generate_pilot_mask()

        preamble_num_symbols = self.preamble.num_symbols if self.preamble else 0

        if include_preamble:
            total_num_symbols = (
                preamble_num_symbols + body_num_symbols + self.guard_num_symbols
            )

            preamble_bool = xp.zeros(total_num_symbols, dtype=bool)
            pilot_bool = xp.zeros(total_num_symbols, dtype=bool)
            payload_bool = xp.zeros(total_num_symbols, dtype=bool)
            guard_bool = xp.zeros(total_num_symbols, dtype=bool)

            if self.guard_type == "cp":
                g_slice = slice(0, self.guard_num_symbols)
                p_slice = slice(
                    self.guard_num_symbols,
                    self.guard_num_symbols + preamble_num_symbols,
                )
                b_slice = slice(
                    self.guard_num_symbols + preamble_num_symbols, total_num_symbols
                )
            else:
                p_slice = slice(0, preamble_num_symbols)
                b_slice = slice(
                    preamble_num_symbols, preamble_num_symbols + body_num_symbols
                )
                g_slice = slice(
                    preamble_num_symbols + body_num_symbols, total_num_symbols
                )

            if preamble_num_symbols > 0:
                preamble_bool[p_slice] = True

            pilot_bool[b_slice] = mask
            payload_bool[b_slice] = ~mask

            if self.guard_num_symbols > 0:
                guard_bool[g_slice] = True

            res = {
                "preamble": preamble_bool,
                "pilots": pilot_bool,
                "payload": payload_bool,
                "guard": guard_bool,
            }
        else:
            # Preamble removed.
            # If CP, guard is at the start and is typically removed with preamble.
            # If ZERO, guard is at the end and remains part of the signal.
            if self.guard_type == "cp":
                total_num_symbols = body_num_symbols
                pilot_bool = mask
                payload_bool = ~mask
                res = {
                    "pilots": pilot_bool,
                    "payload": payload_bool,
                }
            else:
                total_num_symbols = body_num_symbols + self.guard_num_symbols
                pilot_bool = xp.zeros(total_num_symbols, dtype=bool)
                payload_bool = xp.zeros(total_num_symbols, dtype=bool)
                guard_bool = xp.zeros(total_num_symbols, dtype=bool)

                b_slice = slice(0, body_num_symbols)
                g_slice = slice(body_num_symbols, total_num_symbols)

                pilot_bool[b_slice] = mask
                payload_bool[b_slice] = ~mask
                guard_bool[g_slice] = True

                res = {
                    "pilots": pilot_bool,
                    "payload": payload_bool,
                    "guard": guard_bool,
                }

        if unit == "samples":
            for k in res:
                res[k] = xp.repeat(res[k], sps)

        return res

    # -------------------------------------------------------------------------
    # Signal Generation
    # -------------------------------------------------------------------------

    def to_signal(
        self,
        *,
        symbol_rate: float,
        sps: int = 1,
        pulse: Pulse | ArrayType | None = None,
        periodic: bool = False,
        device: str = "cpu",
    ) -> Signal:
        """Build the frame's waveform: shaped, oversampled, with its guard interval.

        Parameters
        ----------
        symbol_rate : float
            Symbol rate in Hz.
        sps : int, default 1
            Samples per symbol (oversampling factor).
        pulse : Pulse or array_like, optional
            Pulse object or taps for both preamble and body; ``None``
            zero-stuffs without shaping (as :func:`commkit.generate`).
        periodic : bool, default False
            Shape the whole frame (preamble, body and guard) at once as one
            period of a periodic sequence (circular convolution): the frame
            loops seamlessly, e.g. on a streaming AWG, and the preamble-body
            junction carries no filter transients.  The guard is inserted in
            symbols before shaping instead of in samples after it.
        device : {"cpu", "gpu"}, default "cpu"
            Where the waveform is shaped and returned.  The frame's symbols
            and bits stay on the host either way.

        Returns
        -------
        Signal
            complex64 samples, ``(N,)`` or ``(num_channels, N)``, with
            ``frame`` set to this frame.  ``constellation`` and ``reference``
            are not set: they belong to the payload, which
            :func:`extract_payload` returns as a plain Signal.

        Notes
        -----
        The preamble and the body are each scaled so that the larger of their
        I and Q peaks is 1 (``normalize(mode="dac_peak")``), so both use the
        full DAC range whatever their modulation.  The assembled frame is
        then normalized per channel to unit symbol power (average sample power
        ``1/sps``), as ``generate`` does; the pilot-to-payload ratio of
        ``pilot_gain_db`` is kept.
        """
        sps = require_integer_sps(sps, "SingleCarrierFrame.to_signal()")
        device = require_device(device, "SingleCarrierFrame.to_signal()")
        if periodic:
            return Signal(
                samples=self._periodic_samples(sps, pulse, device),
                sampling_rate=symbol_rate * sps,
                symbol_rate=symbol_rate,
                pulse=pulse if isinstance(pulse, Pulse) else None,
                frame=self,
            )

        # 1. Shape Body (Payload + Pilots)
        body, xp, _ = dispatch(to_device(self.body_symbols, device))
        body_samples = generation.shape_pulse(body, sps=sps, pulse=pulse)

        # Normalize body per-channel via normalize's "dac_peak" mode:
        # max(peak_|I|, peak_|Q|) - a single scale factor that brings the
        # dominant component to 1.0 while preserving the I/Q ratio.
        # Complex-envelope peak normalization (used in the DSP chain) divides
        # by max(|sample|) instead, leaving components at ≤ 1/√2 ≈ 0.707 for
        # square QAM/PSK whose envelope peak sits at 45°.  Applied
        # per-section (body and preamble separately) so each segment uses
        # the full DAC range regardless of modulation type or constellation
        # phase geometry.
        body_samples = normalize(body_samples, mode="dac_peak", axis=-1)

        # 2. Shape Preamble (if present)
        if self.preamble is not None:
            # Use Preamble's to_signal for shaping to reuse logic,
            # but we only need the samples.
            # CRITICAL: Must use EXACT same shaping parameters as body.
            preamble_signal = self.preamble.to_signal(
                sps=sps, symbol_rate=symbol_rate, pulse=pulse, device=device
            )
            preamble_samples = preamble_signal.samples
            # (L*sps,) for SISO  or  (num_channels, L*sps) for MIMO - shape driven by preamble.num_channels

            # I/Q peak normalization - axis=-1 works for both 1-D and 2-D
            preamble_samples = normalize(preamble_samples, mode="dac_peak", axis=-1)
            if self.num_channels > 1 and preamble_samples.ndim == 1:
                # A one-channel preamble is broadcast to every channel.
                preamble_samples = xp.tile(preamble_samples, (self.num_channels, 1))

            # Concatenate Preamble + Body
            samples = xp.concatenate([preamble_samples, body_samples], axis=-1)
        else:
            samples = body_samples

        # 3. Apply Guard Interval at sample level
        if self.guard_num_symbols > 0:
            guard_num_samples = int(self.guard_num_symbols * sps)
            if self.guard_type == "zero":
                zeros: ArrayType
                if self.num_channels > 1:
                    zeros = xp.zeros(
                        (self.num_channels, guard_num_samples), dtype="complex64"
                    )
                else:
                    zeros = xp.zeros(guard_num_samples, dtype="complex64")
                samples = xp.concatenate([samples, zeros], axis=-1)
            elif self.guard_type == "cp":
                cp_slice = samples[..., -guard_num_samples:]
                samples = xp.concatenate([cp_slice, samples], axis=-1)

        # 4. Normalize assembled frame to unit average power.
        # Each section (preamble, body) was independently I/Q peak-normalized so
        # that both use the full DAC range irrespective of their modulation format.
        # After concatenation the sections may differ in average power, so a final
        # global normalization brings the frame to unit symbol power (Es = 1),
        # i.e. average sample power = 1/sps.  Pilot/payload power ratios within
        # the body are preserved because every section's samples are scaled by the
        # same factor.  Guard zeros remain zero after scaling.
        samples = normalize(samples, mode="symbol_power", sps=sps, axis=-1)

        # The payload modulation (and its pmf) lives on the frame:
        # sig.frame.payload_constellation.
        return Signal(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            pulse=pulse if isinstance(pulse, Pulse) else None,
            frame=self,
        )

    def _periodic_samples(
        self, sps: int, pulse: Pulse | ArrayType | None, device: str
    ) -> ArrayType:
        """The frame shaped once, circularly, at unit symbol power.

        Each section gets the gain that brings its own circularly shaped
        waveform to a ``dac_peak`` of 1, as in the linear path; the scaled
        sections and the guard then form one symbol stream that is shaped
        as a whole.
        """
        taps = pulse.taps(sps) if isinstance(pulse, Pulse) else pulse
        body, xp, _ = dispatch(to_device(self.body_symbols, device))

        def dac_peak_gain(symbols: ArrayType) -> ArrayType:
            """Per channel, the gain that scales the larger quadrature peak to 1."""
            x = generation.expand(symbols, factor=sps)
            if taps is not None:
                x = convolve_periodic(x, taps)
            peak = xp.maximum(
                xp.max(xp.abs(x.real), axis=-1, keepdims=True),
                xp.max(xp.abs(x.imag), axis=-1, keepdims=True),
            )
            return 1.0 / peak

        stream = body * dac_peak_gain(body)
        if self.preamble is not None:
            preamble = to_device(self.preamble.symbols, device)
            preamble = preamble * dac_peak_gain(preamble)
            if self.num_channels > 1 and preamble.ndim == 1:
                # A one-channel preamble is broadcast to every channel.
                preamble = xp.tile(preamble, (self.num_channels, 1))
            stream = xp.concatenate([preamble, stream], axis=-1)

        if self.guard_num_symbols > 0:
            if self.guard_type == "zero":
                zeros = xp.zeros(
                    (*stream.shape[:-1], self.guard_num_symbols), stream.dtype
                )
                stream = xp.concatenate([stream, zeros], axis=-1)
            elif self.guard_type == "cp":
                stream = xp.concatenate(
                    [stream[..., -self.guard_num_symbols :], stream], axis=-1
                )

        stream = stream.astype("complex64", copy=False)
        return generation.shape_pulse(stream, sps=sps, pulse=taps, periodic=True)


def extract_payload(signal: Signal) -> Signal:
    """Extract the payload of a frame Signal as a plain Signal at the symbol rate.

    The payload's position is known only in the full frame, so the Signal
    must hold exactly the frame's symbols, one sample per symbol, aligned
    with its start (preamble, pilots and guard included).

    Parameters
    ----------
    signal : Signal
        A Signal with a ``frame``, at one sample per symbol, ``(N,)`` or
        ``(num_channels, N)`` with ``N`` the frame length in symbols.

    Returns
    -------
    Signal
        The payload symbols at one sample per symbol.  ``constellation`` is
        the payload constellation, ``reference`` holds the payload symbols,
        bits and source bits (on the samples' device) and ``frame`` is
        ``None``.

    Raises
    ------
    ValueError
        If the Signal has no frame, is not at one sample per symbol, or its
        length or channel count is not the frame's.
    """
    if not isinstance(signal, Signal):
        raise TypeError(
            f"extract_payload() takes a Signal, got {type(signal).__name__}."
        )
    frame = signal.frame
    if frame is None:
        raise ValueError("extract_payload(): the Signal has no frame.")
    if not _same_fact(signal.sps, 1):
        raise ValueError(
            f"extract_payload() needs one sample per symbol, got sps={signal.sps}; "
            "decimate to the symbol rate first."
        )
    payload = frame.get_structure_map()["payload"]
    x = signal.samples
    if x.shape[-1] != payload.size:
        raise ValueError(
            f"extract_payload(): the Signal has {x.shape[-1]} symbols, the frame "
            f"{payload.size}. The payload position is known only in the full "
            "frame."
        )
    num_channels = 1 if x.ndim == 1 else x.shape[0]
    if num_channels != frame.num_channels:
        raise ValueError(
            f"extract_payload(): the Signal has {num_channels} channel(s), the "
            f"frame {frame.num_channels} channel(s)."
        )
    device = "cpu" if signal.xp is np else "gpu"
    reference = frame._payload_reference().to(device)
    return Signal(
        samples=x[..., signal.xp.asarray(payload)],
        sampling_rate=signal.sampling_rate,
        symbol_rate=signal.symbol_rate,
        constellation=frame.payload_constellation,
        pulse=signal.pulse,
        reference=reference,
        center_frequency=signal.center_frequency,
        domain=signal.domain,
    )


def _frame_pilots(
    frame: SingleCarrierFrame, include_preamble: bool, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Symbol indices and symbols (host) of a frame's pilots, for the
    ``from_frame`` constructors of the pilot-aided methods."""
    if frame.pilot_symbols is None:
        raise ValueError(f"{name}.from_frame(): the frame has no pilots.")
    mask = frame.get_structure_map(include_preamble=include_preamble)["pilots"]
    return np.flatnonzero(mask), np.asarray(frame.pilot_symbols)


def _resolve_seed(rng: Any) -> int:
    """The integer seed a frame keeps: ``rng`` itself, one draw from a
    Generator, or fresh entropy for ``None``."""
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(2**63))
    if rng is None:
        return int(np.random.SeedSequence().generate_state(1, np.uint64)[0] >> 1)
    _check_int("payload_rng", rng, 0)
    return int(rng)


def _check_int(name: str, value: Any, low: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value < low:
        raise ValueError(f"{name} must be >= {low}, got {value}.")
