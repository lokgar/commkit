"""
Frame containers: structured preamble and single-carrier frame models.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .._sequences import barker_sequence, zadoff_chu_sequence, zc_mimo_root
from ..backend import ArrayType
from ..filtering import Pulse
from ..mapping import Constellation
from ..math import db_to_linear, normalize
from . import generation
from ._signal_adapter import _same_fact, require_integer_sps
from .signal import Reference, Signal


@dataclass(frozen=True, kw_only=True)
class Preamble:
    """
    Structured container for frame synchronization sequences (preambles).

    Preambles are automatically generated based on the specified sequence type
    and length. Manual overrides for bits or symbols are not supported to
    ensure consistency within the processing pipeline.

    Attributes
    ----------
    sequence_type : {"barker", "zc"}, default "barker"
        The synchronization sequence algorithm.
    length : int
        Total length of the preamble in symbols.
        For "barker": length must be from the set {2, 3, 4, 5, 7, 11, 13}.
        For "zc": length must be a prime number.
    root : int, default 1
        ZC root index (only meaningful for ``sequence_type='zc'``).
        Must satisfy ``1 <= root < length``.
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
        """
        Validate the fields and generate the preamble symbols.

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

        stype = self.sequence_type.lower()
        symbols: Any

        if stype == "barker":
            # Barker symbols (-1, +1)
            base = barker_sequence(self.length)
        elif stype in ("zc", "zadoff_chu"):
            # ZC complex symbols - use the named 'root' field directly.
            base = zadoff_chu_sequence(self.length, root=self.root)
        else:
            base = None

        if base is not None and self.num_streams > 1:
            if stype in ("zc", "zadoff_chu"):
                rows = [
                    zadoff_chu_sequence(
                        self.length,
                        root=zc_mimo_root(k, self.root, self.length),
                    )
                    for k in range(self.num_streams)
                ]
                symbols = np.stack(rows, axis=0)  # (num_streams, length)
            else:
                symbols = np.tile(base[None, :], (self.num_streams, 1))
        else:
            symbols = base

        # Consistent internal dtype; sequences stay on the CPU (no hidden
        # device placement).
        if symbols is not None:
            symbols = symbols.astype("complex64")
        object.__setattr__(self, "_symbols", symbols)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def symbols(self) -> Any:
        """The IQ symbols of the preamble."""
        return self._symbols

    @property
    def num_symbols(self) -> int:
        """Total number of symbols in the preamble."""
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
    ) -> Signal:
        """
        Pulse-shaped waveform of the preamble sequence.

        Parameters
        ----------
        sps : int
            Samples per symbol.
        symbol_rate : float
            Symbol rate in Hz.
        pulse : Pulse or array_like, optional
            Pulse object or taps; ``None`` zero-stuffs without shaping (as
            :func:`commkit.generate`).

        Returns
        -------
        Signal
            The shaped preamble at unit symbol power.
        """
        sps = require_integer_sps(sps, "Preamble.to_signal()")
        return Signal(
            samples=generation.shape_pulse(self.symbols, sps=sps, pulse=pulse),
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            pulse=pulse if isinstance(pulse, Pulse) else None,
        )


def _qpsk() -> Constellation:
    return Constellation.psk(4)


@dataclass(frozen=True, kw_only=True)
class SingleCarrierFrame:
    """
    Represents a structured single-carrier frame with Preamble, Pilots, Payload, and Guard Interval.

    This class provides a high-level abstraction for constructing frames
    used in digital communication systems (1/10/100 GbE, 5G, etc.).
    It supports various pilot patterns for channel estimation and guard
    intervals for multi-path mitigation.

    Attributes
    ----------
    payload_len : int, default 1000
        Number of payload symbols per stream.  With pilots it must fill whole
        pilot periods (comb: a multiple of ``pilot_period - 1``; block: of
        ``pilot_period - pilot_block_len``); otherwise construction raises.
    payload_constellation : Constellation, default ``Constellation.psk(4)``
        Payload constellation.  A shaped one
        (``Constellation.qam(64).shaped(entropy=5)``) gives a PS payload.
    payload_seed : int, default 42
        Seed for reproducible payload data generation.
    preamble : Preamble, optional
        Structured preamble for synchronization.  For MIMO with ZC sequences,
        each TX stream automatically receives a unique root via
        ``zc_mimo_root``.
    pilot_pattern : {"none", "block", "comb"}, default "none"
        "none": No pilots.
        "block": A block of symbols at the start of the frame body.
        "comb": Single pilot symbols interleaved every `pilot_period`.
    pilot_period : int, default 0
        The period of pilot insertion in symbols.
    pilot_block_len : int, default 0
        Length of the pilot block (mode="block") in symbols.
    pilot_seed : int, default 1337
        Seed for pilot symbol generation.
    pilot_constellation : Constellation, default ``Constellation.psk(4)``
        Pilot constellation; must be unshaped.
    pilot_gain_db : float, default 0.0
        Pilot boosting in dB relative to the payload power.
    guard_type : {"zero", "cp"}, default "zero"
        "zero": Zero-padding at the end of the frame.
        "cp": Cyclic Prefix prepended to the frame.
    guard_len : int, default 0
        Length of the guard interval in symbols.
    num_streams : int, default 1
        Number of independent spatial streams (MIMO).

    Notes
    -----
    The layout (``get_structure_map``, the pilot mask) depends only on the
    fields and never generates data.  Payload and pilot symbols are generated
    on first access and cached.
    """

    payload_len: int = 1000
    payload_constellation: Constellation = field(default_factory=_qpsk)
    payload_seed: int = 42

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
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

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

        per_period = None
        if self.pilot_pattern == "comb" and self.pilot_period > 1:
            per_period = self.pilot_period - 1
        elif (
            self.pilot_pattern == "block"
            and self.pilot_period > self.pilot_block_len > 0
        ):
            per_period = self.pilot_period - self.pilot_block_len
        if per_period is not None and self.payload_len % per_period:
            low = self.payload_len // per_period * per_period
            valid = f"{low} or {low + per_period}" if low else f"{per_period}"
            raise ValueError(
                f"payload_len={self.payload_len} does not fill whole pilot periods "
                f"({per_period} payload symbols each); use {valid}."
            )

    # -------------------------------------------------------------------------
    # Mask Generation and Internal Data Preparation Methods
    # -------------------------------------------------------------------------

    def _generate_pilot_mask(self) -> tuple[ArrayType, int]:
        """
        Calculates the pilot placement mask and total frame length.

        Returns
        -------
        mask : array_like (bool)
            Boolean mask where True indicates a pilot symbol location.
        body_length : int
            Total number of symbols in the frame body (payload + pilots).
        """
        xp = np

        # No pilots: simple payload mapping
        if self.pilot_pattern == "none":
            body_length = self.payload_len
            mask = xp.zeros(body_length, dtype=bool)
            return mask, body_length

        # Comb pattern: single pilot every N symbols
        if self.pilot_pattern == "comb":
            if self.pilot_period <= 1:
                raise ValueError("pilot_period must be > 1 for 'comb' pattern.")
            data_per_period = self.pilot_period - 1
            num_full_periods = self.payload_len // data_per_period
            remainder = self.payload_len % data_per_period

            total_length = num_full_periods * self.pilot_period + remainder
            # If we have a remainder, we need one more pilot at the start of the partial period
            if remainder > 0:
                total_length += 1

            mask = xp.zeros(total_length, dtype=bool)
            mask[:: self.pilot_period] = True
            return mask, total_length

        # Block pattern: block of pilots followed by block of data
        if self.pilot_pattern == "block":
            if self.pilot_period <= self.pilot_block_len:
                raise ValueError(
                    "pilot_period must be > pilot_block_len for 'block' pattern."
                )
            data_per_block = self.pilot_period - self.pilot_block_len
            num_blocks = int(xp.ceil(self.payload_len / data_per_block))

            # Create a single block pattern [P P ... P D D ... D]
            block_pattern = xp.zeros(self.pilot_period, dtype=bool)
            block_pattern[: self.pilot_block_len] = True

            # Repeat the pattern for all blocks
            mask = xp.tile(block_pattern, num_blocks)

            # Truncation: Find the exact index where the required payload ends
            false_indices = xp.where(~mask)[0]
            last_idx = false_indices[self.payload_len - 1]
            mask = mask[: last_idx + 1]
            return mask, len(mask)

        return xp.zeros(self.payload_len, dtype=bool), self.payload_len

    def _ensure_payload_generated(self) -> None:
        """Generate and cache the payload bits and symbols with ``generate()``."""
        if self._cache.get("payload_bits") is not None:
            return
        sig = generation.generate(
            self.payload_constellation,
            self.payload_len,
            symbol_rate=1.0,
            num_channels=self.num_streams,
            rng=self.payload_seed,
        )
        ref = sig.reference
        assert ref is not None
        self._cache["payload_bits"] = ref.bits
        self._cache["payload_symbols"] = ref.symbols

    def _ensure_pilot_generated(self) -> None:
        """
        Generate and cache the pilot bits and symbols with ``generate()``.

        Pilots are always uniform: shaping would destroy the known-reference
        property required for channel estimation.
        """
        if self._cache.get("pilot_bits") is not None or self.pilot_pattern == "none":
            return
        mask, _ = self._generate_pilot_mask()
        pilot_count = int(np.sum(mask))
        if pilot_count == 0:
            return
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
        """
        Returns the raw payload bits.

        Returns
        -------
        array_like
            Binary bits (0s and 1s).
        """
        self._ensure_payload_generated()
        return self._cache.get("payload_bits")

    @property
    def payload_symbols(self) -> ArrayType:
        """
        Returns the modulated payload symbols.

        Returns
        -------
        array_like
            IQ symbols.
        """
        self._ensure_payload_generated()
        return self._cache.get("payload_symbols")

    @property
    def pilot_bits(self) -> ArrayType | None:
        """
        Returns the raw pilot bits, if pilots are enabled.

        Returns
        -------
        array_like or None
            Binary bits if `pilot_pattern` is not "none".
        """
        if self.pilot_pattern == "none":
            return None
        self._ensure_pilot_generated()
        return self._cache.get("pilot_bits")

    @property
    def pilot_symbols(self) -> ArrayType | None:
        """
        Returns the modulated pilot symbols.

        Returns
        -------
        array_like or None
            IQ symbols if `pilot_pattern` is not "none".
        """
        if self.pilot_pattern == "none":
            return None
        self._ensure_pilot_generated()
        return self._cache.get("pilot_symbols")

    @property
    def body_symbols(self) -> ArrayType:
        """
        Returns the interleaved payload and pilot symbols.

        WARNING: Pilot gain is applied if `pilot_gain_db` is not zero,
        so these are not "clean" symbols but scaled relatively.

        Returns
        -------
        array_like
            Determined by `pilot_pattern` and `pilot_period`.
        """
        xp = np
        mask, body_length = self._generate_pilot_mask()

        body: np.ndarray
        if self.num_streams > 1:
            # Shape: (Channels, Time)
            body = xp.zeros((self.num_streams, body_length), dtype="complex64")

            if self.pilot_pattern != "none":
                pilot_symbols = self.pilot_symbols
                assert pilot_symbols is not None
                # Apply pilot boosting/gain (dB to linear)
                if self.pilot_gain_db != 0.0:
                    pilot_symbols = pilot_symbols * db_to_linear(
                        self.pilot_gain_db, power=False
                    )

                body[:, mask] = pilot_symbols

            body[:, ~mask] = self.payload_symbols
        else:
            body = xp.zeros(body_length, dtype="complex64")
            if self.pilot_pattern != "none":
                pilot_symbols = self.pilot_symbols
                assert pilot_symbols is not None
                # Apply pilot boosting/gain (dB to linear)
                if self.pilot_gain_db != 0.0:
                    pilot_symbols = pilot_symbols * db_to_linear(
                        self.pilot_gain_db, power=False
                    )

                body[mask] = pilot_symbols
            body[~mask] = self.payload_symbols

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
        """
        Generates boolean masks identifying the segments of the frame.

        Parameters
        ----------
        unit : {"symbols", "samples"}, default "symbols"
            The scale of the returned masks.
        sps : int, default 1
            Samples per symbol (required if unit="samples").
        include_preamble : bool, default True
            If True, returns masks for the full frame including preamble and
            guard intervals. If False, returns masks only for the segments
            after the preamble (and after CP removal if guard_type='cp').

        Returns
        -------
        dict
            Dictionary containing boolean masks for:
            - 'preamble' (only if include_preamble=True)
            - 'pilots'
            - 'payload'
            - 'guard' (only if include_preamble=True OR guard_type='zero')
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
    ) -> Signal:
        """
        Generates a shaped, oversampled waveform from the frame description.

        This is the primary method for moving from a logical frame to
        physical IQ samples. It handles upsampling, pulse shaping,
        guard interval insertion, and metadata population.

        Parameters
        ----------
        sps : int, default 4
            Samples per symbol (oversampling factor).
        symbol_rate : float, default 1e6
            Symbol rate in Hz.
        pulse : Pulse or array_like, optional
            Pulse object or taps for both preamble and body; ``None``
            zero-stuffs without shaping (as :func:`commkit.generate`).

        Returns
        -------
        Signal
            A `Signal` object containing the IQ samples and metadata.

        Notes
        -----
        Each section (preamble and body) is independently I/Q component peak-normalised
        so both occupy the full DAC range regardless of their modulation format.
        After concatenation the full frame is normalised to **unit symbol power
        (Es = 1)**, meaning average sample power = 1/sps.  This matches the
        convention used by ``shape_pulse`` and ``apply_awgn``.
        Pilot/payload power ratios set by `pilot_gain_db` are preserved throughout.
        """
        xp = np
        sps = require_integer_sps(sps, "SingleCarrierFrame.to_signal()")

        # 1. Shape Body (Payload + Pilots)
        body_samples = generation.shape_pulse(self.body_symbols, sps=sps, pulse=pulse)

        # Normalise body per-channel via normalize's "dac_peak" mode:
        # max(peak_|I|, peak_|Q|) - a single scale factor that brings the
        # dominant component to 1.0 while preserving the I/Q ratio.
        # Complex-envelope peak normalisation (used in the DSP chain) divides
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
                sps=sps, symbol_rate=symbol_rate, pulse=pulse
            )
            preamble_samples = xp.asarray(preamble_signal.samples)
            # (L*sps,) for SISO  or  (num_streams, L*sps) for MIMO - shape driven by preamble.num_streams

            # I/Q peak normalisation - axis=-1 works for both 1-D and 2-D
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
                zeros: np.ndarray
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
        # Each section (preamble, body) was independently I/Q peak-normalised so
        # that both use the full DAC range irrespective of their modulation format.
        # After concatenation the sections may differ in average power, so a final
        # global normalization brings the frame to unit symbol power (Es = 1),
        # i.e. average sample power = 1/sps.  Pilot/payload power ratios within
        # the body are preserved because every section's samples are scaled by the
        # same factor.  Guard zeros remain zero after scaling.
        samples = normalize(samples, mode="symbol_power", sps=sps, axis=-1)

        # The payload modulation lives on the frame; the PS pmf is reachable
        # through the Signal's ps_pmf bridge property (frame.payload_ps_pmf).
        return Signal(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            pulse=pulse if isinstance(pulse, Pulse) else None,
            frame=self,
        )


def extract_payload(signal: Signal) -> Signal:
    """The payload of a frame Signal, as a plain Signal at the symbol rate.

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
        the payload constellation, ``reference`` holds the payload symbols
        and bits (on the samples' device) and ``frame`` is ``None``.

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
    reference = Reference(symbols=frame.payload_symbols, bits=frame.payload_bits).to(
        device
    )
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
