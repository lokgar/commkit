"""Frame containers: preambles and single-carrier frames.

A :class:`SingleCarrierFrame` describes a transmitted frame (preamble,
payload, pilots, guard intervals, MIMO streams) and builds it into a
:class:`~commkit.core.Signal` with ``to_signal``; :func:`extract_payload`
cuts the payload out of a frame Signal at one sample per symbol.
"""

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
    ``(length,)``, or ``(num_streams, length)`` for several streams); they
    cannot be set by hand.

    Parameters
    ----------
    sequence_type : {"barker", "zc"}, default "barker"
        The sequence family.
    length : int
        Length in symbols.  Barker: one of 2, 3, 4, 5, 7, 11, 13.  ZC: any
        length; a prime gives the ideal cross-correlation between roots.
    root : int, default 1
        ZC root index ``u``, in ``[1, length)`` and coprime to ``length``;
        ignored for Barker.
    num_streams : int, default 1
        Number of TX streams.  ZC preambles get a distinct root per stream
        (``zc_mimo_root``); a Barker preamble repeats the same sequence.
    """

    sequence_type: str = "barker"
    length: int
    # ZC root index, 1 <= root < length; ignored for Barker sequences.
    root: int = 1
    # Number of TX streams.  ZC preambles get a unique root per stream
    # (zc_mimo_root); Barker broadcasts the same sequence.
    num_streams: int = 1

    # Generated in __post_init__ from the fields above.
    _symbols: Any = field(default=None, init=False, repr=False, compare=False)

    # -------------------------------------------------------------------------
    # Validators and Post-Initialization Hooks
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate the fields and generate the preamble symbols.

        This ensures that standard sequences are generated correctly according
        to the requested sequence properties.

        For ``num_streams == 1`` the internal ``_symbols`` shape is ``(length,)``.
        For ``num_streams > 1`` it becomes ``(num_streams, length)``:
        - ZC: each row uses the unique root from ``zc_mimo_root``.
        - Barker: the same sequence is tiled across all streams.
        """
        if self.sequence_type not in ("barker", "zc"):
            raise ValueError(
                f"sequence_type must be 'barker' or 'zc', got {self.sequence_type!r}."
            )
        _check_int("length", self.length, 1)
        _check_int("root", self.root, 1)
        _check_int("num_streams", self.num_streams, 1)

        symbols: np.ndarray
        if self.sequence_type == "barker":
            # Barker symbols (-1, +1), the same on every stream.
            symbols = barker_sequence(self.length)
            if self.num_streams > 1:
                symbols = np.tile(symbols[None, :], (self.num_streams, 1))
        elif self.num_streams > 1:
            # One Zadoff-Chu root per stream, (num_streams, length).
            symbols = np.stack(
                [
                    zadoff_chu_sequence(
                        self.length, root=zc_mimo_root(k, self.root, self.length)
                    )
                    for k in range(self.num_streams)
                ]
            )
        else:
            symbols = zadoff_chu_sequence(self.length, root=self.root)

        # Consistent internal dtype; sequences stay on the CPU (no hidden
        # device placement).
        object.__setattr__(self, "_symbols", symbols.astype("complex64"))

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def symbols(self) -> Any:
        """The preamble symbols (host complex64).

        ``(length,)``, or ``(num_streams, length)`` for several streams.
        """
        return self._symbols

    @property
    def num_symbols(self) -> int:
        """Number of symbols per stream, ``length``."""
        return self.length

    # -------------------------------------------------------------------------
    # Signal Generation
    # -------------------------------------------------------------------------

    def to_signal(
        self,
        sps: int,
        symbol_rate: float,
        *,
        pulse: Pulse | ArrayType | None = None,
        periodic: bool = False,
        device: str = "cpu",
    ) -> Signal:
        """Shape the preamble into a Signal.

        Parameters
        ----------
        sps : int
            Samples per symbol.
        symbol_rate : float
            Symbol rate in Hz.
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
            The shaped preamble, normalized per stream to unit symbol power,
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
    payload_len : int, default 1000
        Number of payload symbols per stream.  With pilots it must fill whole
        pilot periods (comb: a multiple of ``pilot_period - 1``; block: of
        ``pilot_period - pilot_block_len``); otherwise construction raises.
    payload_constellation : Constellation, default ``Constellation.psk(4)``
        Payload constellation.  A shaped one
        (``Constellation.qam(64).shaped(entropy=5)``) gives a PS payload.
    payload_seed : int, default 42
        Seed of the payload data.
    payload_code : code object, optional
        Channel code of the payload (:mod:`commkit.coding`), passed to
        :func:`generate` as ``code``: ``payload_len`` times the bits per
        symbol must be a multiple of ``payload_code.n``.  Not with a shaped
        payload constellation.
    preamble : Preamble, optional
        Synchronization preamble, before the body.  A multi-stream preamble
        must have ``num_streams`` streams; a one-stream preamble is repeated
        on every stream.
    pilot_pattern : {"none", "block", "comb"}, default "none"
        ``"none"``: no pilots.  ``"comb"``: one pilot every
        ``pilot_period`` symbols, the first at the start of the body.
        ``"block"``: ``pilot_block_len`` pilots at the start of every period
        of ``pilot_period`` symbols.
    pilot_period : int, default 0
        Pilot period in symbols (pilots included); ``> 1`` for ``"comb"``,
        ``> pilot_block_len`` for ``"block"``.
    pilot_block_len : int, default 0
        Pilots per period for ``"block"``.
    pilot_seed : int, default 1337
        Seed of the pilot symbols.
    pilot_constellation : Constellation, default ``Constellation.psk(4)``
        Pilot constellation; must be unshaped.
    pilot_gain_db : float, default 0.0
        Pilot power boost in dB relative to the payload.
    guard_type : {"zero", "cp"}, default "zero"
        ``"zero"``: zeros after the frame.  ``"cp"``: a cyclic prefix, the
        last ``guard_len`` symbols of the frame repeated before it.
    guard_len : int, default 0
        Guard interval in symbols.
    num_streams : int, default 1
        Number of independent streams (MIMO), each with its own payload and
        pilots.

    Notes
    -----
    The layout (``get_structure_map``, the pilot mask) depends only on the
    fields and never generates data.  Payload and pilot symbols are generated
    on first access and cached.
    """

    payload_len: int = 1000
    payload_constellation: Constellation = field(default_factory=_qpsk)
    payload_seed: int = 42
    payload_code: Any = None

    preamble: Preamble | None = None

    pilot_pattern: str = "none"
    pilot_period: int = 0
    pilot_block_len: int = 0
    pilot_constellation: Constellation = field(default_factory=_qpsk)
    pilot_seed: int = 1337
    pilot_gain_db: float = 0.0

    guard_type: str = "zero"
    guard_len: int = 0

    num_streams: int = 1

    # Lazily generated payload and pilot data.  The frame's fields are frozen;
    # this cache is filled on first access and never changes afterwards.
    _cache: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # -------------------------------------------------------------------------
    # Validators and Post-Initialization Hooks
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate the fields, including that the payload fills whole periods."""
        _check_int("payload_len", self.payload_len, 1)
        _check_int("pilot_period", self.pilot_period, 0)
        _check_int("pilot_block_len", self.pilot_block_len, 0)
        _check_int("guard_len", self.guard_len, 0)
        _check_int("num_streams", self.num_streams, 1)
        for name in ("payload_constellation", "pilot_constellation"):
            if not isinstance(getattr(self, name), Constellation):
                raise ValueError(
                    f"{name} must be a Constellation, e.g. Constellation.qam(16)."
                )
        if self.pilot_constellation.pmf is not None:
            raise ValueError("pilot_constellation must not be shaped.")
        if self.payload_code is not None:
            from .. import coding

            coding.base._codec(self.payload_code, "SingleCarrierFrame()")
            bits = self.payload_len * self.payload_constellation.bits_per_symbol
            if bits % self.payload_code.n:
                raise ValueError(
                    f"payload_len={self.payload_len} carries {bits} bits, not a "
                    f"multiple of payload_code.n={self.payload_code.n}."
                )
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
        if self.preamble is not None and self.preamble.num_streams > 1:
            if self.preamble.num_streams != self.num_streams:
                raise ValueError(
                    f"preamble.num_streams={self.preamble.num_streams} does not "
                    f"match frame.num_streams={self.num_streams}"
                )

        if self.pilot_pattern == "none":
            return
        block = self._pilot_block()
        if block < 1:
            raise ValueError(
                "pilot_block_len must be >= 1 for the 'block' pattern, got "
                f"{self.pilot_block_len}."
            )
        if self.pilot_period <= block:
            bound = "1" if self.pilot_pattern == "comb" else "pilot_block_len"
            raise ValueError(
                f"pilot_period must be > {bound} for the {self.pilot_pattern!r} "
                f"pattern, got {self.pilot_period}."
            )
        per_period = self.pilot_period - block
        if self.payload_len % per_period:
            low = self.payload_len // per_period * per_period
            valid = f"{low} or {low + per_period}" if low else f"{per_period}"
            raise ValueError(
                f"payload_len={self.payload_len} does not fill whole pilot periods "
                f"({per_period} payload symbols each); use {valid}."
            )

    # -------------------------------------------------------------------------
    # Mask Generation and Internal Data Preparation Methods
    # -------------------------------------------------------------------------

    def _pilot_block(self) -> int:
        """Pilots at the start of each period: one for "comb"."""
        return 1 if self.pilot_pattern == "comb" else self.pilot_block_len

    def _generate_pilot_mask(self) -> tuple[ArrayType, int]:
        """Calculate the pilot placement mask and total frame length.

        Every period starts with its pilots; ``__post_init__`` guarantees that
        the payload fills whole periods.

        Returns
        -------
        mask : ndarray (bool)
            Boolean mask where True indicates a pilot symbol location.
        body_length : int
            Total number of symbols in the frame body (payload + pilots).
        """
        if self.pilot_pattern == "none":
            return np.zeros(self.payload_len, dtype=bool), self.payload_len
        block = self._pilot_block()
        period = np.zeros(self.pilot_period, dtype=bool)
        period[:block] = True
        mask = np.tile(period, self.payload_len // (self.pilot_period - block))
        return mask, mask.size

    def _ensure_payload_generated(self) -> None:
        """Generate and cache the payload bits and symbols with ``generate()``."""
        if self._cache.get("payload_bits") is not None:
            return
        sig = generation.generate(
            self.payload_constellation,
            self.payload_len,
            symbol_rate=1.0,
            num_channels=self.num_streams,
            code=self.payload_code,
            rng=self.payload_seed,
        )
        ref = sig.reference
        assert ref is not None
        self._cache["payload_bits"] = ref.bits
        self._cache["payload_info_bits"] = ref.info_bits
        self._cache["payload_code_block"] = ref.code_block
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
            num_channels=self.num_streams,
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

        ``(payload_len * k,)``, or ``(num_streams, payload_len * k)`` for
        several streams.
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
        code_block = self._cache["payload_code_block"]
        coded = code_block != (1, 1)
        return Reference(
            symbols=self._cache["payload_symbols"],
            bits=self._cache["payload_bits"],
            info_bits=self._cache["payload_info_bits"] if coded else None,
            code_block=code_block if coded else None,
        )

    @property
    def payload_symbols(self) -> ArrayType:
        """The payload symbols (host).

        ``(payload_len,)``, or ``(num_streams, payload_len)`` for several
        streams.
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
        ``(num_streams, L)`` with ``L`` the body length.
        """
        mask, body_length = self._generate_pilot_mask()
        streams = (self.num_streams,) if self.num_streams > 1 else ()
        body = np.zeros((*streams, body_length), dtype="complex64")
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
        mask, body_length = self._generate_pilot_mask()

        preamble_len = self.preamble.num_symbols if self.preamble else 0

        if include_preamble:
            total_len = preamble_len + body_length + self.guard_len

            preamble_bool = xp.zeros(total_len, dtype=bool)
            pilot_bool = xp.zeros(total_len, dtype=bool)
            payload_bool = xp.zeros(total_len, dtype=bool)
            guard_bool = xp.zeros(total_len, dtype=bool)

            if self.guard_type == "cp":
                g_slice = slice(0, self.guard_len)
                p_slice = slice(self.guard_len, self.guard_len + preamble_len)
                b_slice = slice(self.guard_len + preamble_len, total_len)
            else:
                p_slice = slice(0, preamble_len)
                b_slice = slice(preamble_len, preamble_len + body_length)
                g_slice = slice(preamble_len + body_length, total_len)

            if preamble_len > 0:
                preamble_bool[p_slice] = True

            pilot_bool[b_slice] = mask
            payload_bool[b_slice] = ~mask

            if self.guard_len > 0:
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
                total_len = body_length
                pilot_bool = mask
                payload_bool = ~mask
                res = {
                    "pilots": pilot_bool,
                    "payload": payload_bool,
                }
            else:
                total_len = body_length + self.guard_len
                pilot_bool = xp.zeros(total_len, dtype=bool)
                payload_bool = xp.zeros(total_len, dtype=bool)
                guard_bool = xp.zeros(total_len, dtype=bool)

                b_slice = slice(0, body_length)
                g_slice = slice(body_length, total_len)

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
        sps: int = 4,
        symbol_rate: float = 1e6,
        *,
        pulse: Pulse | ArrayType | None = None,
        periodic: bool = False,
        device: str = "cpu",
    ) -> Signal:
        """Build the frame's waveform: shaped, oversampled, with its guard interval.

        Parameters
        ----------
        sps : int, default 4
            Samples per symbol (oversampling factor).
        symbol_rate : float, default 1e6
            Symbol rate in Hz.
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
            complex64 samples, ``(N,)`` or ``(num_streams, N)``, with
            ``frame`` set to this frame.  ``constellation`` and ``reference``
            are not set: they belong to the payload, which
            :func:`extract_payload` returns as a plain Signal.

        Notes
        -----
        The preamble and the body are each scaled so that the larger of their
        I and Q peaks is 1 (``normalize(mode="dac_peak")``), so both use the
        full DAC range whatever their modulation.  The assembled frame is
        then normalized per stream to unit symbol power (average sample power
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
            # (L*sps,) for SISO  or  (num_streams, L*sps) for MIMO - shape driven by preamble.num_streams

            # I/Q peak normalization - axis=-1 works for both 1-D and 2-D
            preamble_samples = normalize(preamble_samples, mode="dac_peak", axis=-1)
            if self.num_streams > 1 and preamble_samples.ndim == 1:
                # A one-stream preamble is broadcast to every stream.
                preamble_samples = xp.tile(preamble_samples, (self.num_streams, 1))

            # Concatenate Preamble + Body
            samples = xp.concatenate([preamble_samples, body_samples], axis=-1)
        else:
            samples = body_samples

        # 3. Apply Guard Interval at sample level
        if self.guard_len > 0:
            guard_len_samples = int(self.guard_len * sps)
            if self.guard_type == "zero":
                zeros: ArrayType
                if self.num_streams > 1:
                    zeros = xp.zeros(
                        (self.num_streams, guard_len_samples), dtype="complex64"
                    )
                else:
                    zeros = xp.zeros(guard_len_samples, dtype="complex64")
                samples = xp.concatenate([samples, zeros], axis=-1)
            elif self.guard_type == "cp":
                cp_slice = samples[..., -guard_len_samples:]
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
            if self.num_streams > 1 and preamble.ndim == 1:
                # A one-stream preamble is broadcast to every stream.
                preamble = xp.tile(preamble, (self.num_streams, 1))
            stream = xp.concatenate([preamble, stream], axis=-1)

        if self.guard_len > 0:
            if self.guard_type == "zero":
                zeros = xp.zeros((*stream.shape[:-1], self.guard_len), stream.dtype)
                stream = xp.concatenate([stream, zeros], axis=-1)
            elif self.guard_type == "cp":
                stream = xp.concatenate(
                    [stream[..., -self.guard_len :], stream], axis=-1
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
        ``(num_streams, N)`` with ``N`` the frame length in symbols.

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
    if num_channels != frame.num_streams:
        raise ValueError(
            f"extract_payload(): the Signal has {num_channels} channel(s), the "
            f"frame {frame.num_streams} stream(s)."
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
    )


def _check_int(name: str, value: Any, low: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value < low:
        raise ValueError(f"{name} must be >= {low}, got {value}.")
