"""
Frame containers: structured preamble and single-carrier frame models.
"""

from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

from ._signal_adapter import require_integer_sps

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

from .. import helpers
from ..backend import ArrayType, is_cupy_available, to_device
from ..logger import logger
from . import generation
from .signal import Signal


class Preamble(BaseModel):
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

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    sequence_type: Literal["barker", "zc"] = "barker"
    length: int
    root: int = Field(
        default=1,
        ge=1,
        description="ZC root index.  Only meaningful for ``sequence_type='zc'``; "
        "ignored for Barker sequences.  Must satisfy ``1 <= root < length``; "
        "for prime ``length`` every root in this range yields a valid CAZAC sequence.",
    )
    num_streams: int = Field(
        default=1,
        ge=1,
        description="Number of TX streams.  For ZC preambles each stream gets a "
        "unique root derived via ``helpers.zc_mimo_root``.  "
        "For Barker the same sequence is broadcast to all streams.",
    )

    # Internal state managed during post-init
    _symbols: Any = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Validators and Post-Initialization Hooks
    # -------------------------------------------------------------------------

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook to automate symbol generation and device placement.

        This ensures that standard sequences are generated correctly according
        to the requested sequence properties.

        For ``num_streams == 1`` the internal ``_symbols`` shape is ``(length,)``.
        For ``num_streams > 1`` it becomes ``(num_streams, length)``:
        - ZC: each row uses the unique root from ``helpers.zc_mimo_root``.
        - Barker: the same sequence is tiled across all streams.
        """
        from .. import timing

        stype = self.sequence_type.lower()

        if stype == "barker":
            # Barker symbols (-1, +1)
            base = timing.barker_sequence(self.length)
        elif stype in ("zc", "zadoff_chu"):
            # ZC complex symbols - use the named 'root' field directly.
            base = timing.zadoff_chu_sequence(self.length, root=self.root)
        else:
            base = None

        if base is not None and self.num_streams > 1:
            if stype in ("zc", "zadoff_chu"):
                rows = [
                    timing.zadoff_chu_sequence(
                        self.length,
                        root=helpers.zc_mimo_root(k, self.root, self.length),
                    )
                    for k in range(self.num_streams)
                ]
                self._symbols = np.stack(rows, axis=0)  # (num_streams, length)
            else:
                self._symbols = np.tile(base[None, :], (self.num_streams, 1))
        else:
            self._symbols = base

        # Move to GPU if available
        if is_cupy_available():
            if self._symbols is not None:
                self._symbols = to_device(self._symbols, "gpu")

            # Ensure consistent internal dtype (complex64)
            if self._symbols is not None:
                self._symbols = self._symbols.astype("complex64")

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
        pulse_shape: str = "rrc",
        filter_span: int = 10,
        rrc_rolloff: float = 0.35,
        rc_rolloff: float = 0.35,
        rise_time: float = 0.0,
        duty_cycle: float = 1.0,
    ) -> Signal:
        """
        Generates a shaped waveform from the preamble sequence.

        Parameters
        ----------
        sps : int
            Samples per symbol.
        symbol_rate : float
            Symbol rate in Hz.
        pulse_shape : str, default "rrc"
            The pulse shaping type to apply.
        filter_span : int, default 10
            Filter span in symbols.
        rrc_rolloff : float, default 0.35
            Roll-off factor for RRC filter.
        rc_rolloff : float, default 0.35
            Roll-off factor for RC filter.
        rise_time : float, default 0.22
            10%-90% edge transition duration in symbol periods (smoothrect only).
        duty_cycle : float, default 1.0
            FWHM of the Gaussian pulse in symbol periods (gaussian only).
        duty_cycle : float, default 1.0
            Fraction of the symbol period occupied by the pulse (rect/smoothrect).

        Returns
        -------
        Signal
            A `Signal` object with the shaped preamble.
        """
        from .generation import shape_pulse

        sps = require_integer_sps(sps, "Preamble.to_signal()")

        samples = shape_pulse(
            self.symbols,
            sps=sps,
            pulse_shape=pulse_shape,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            rise_time=rise_time,
            duty_cycle=duty_cycle,
        )

        return Signal(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            mod_scheme=None,
            mod_order=None,
            source_symbols=None,
            pulse_shape=pulse_shape,
            duty_cycle=duty_cycle,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            rise_time=rise_time,
            signal_type="Preamble",
        )


class SingleCarrierFrame(BaseModel):
    """
    Represents a structured single-carrier frame with Preamble, Pilots, Payload, and Guard Interval.

    This class provides a high-level abstraction for constructing frames
    used in digital communication systems (1/10/100 GbE, 5G, etc.).
    It supports various pilot patterns for channel estimation and guard
    intervals for multi-path mitigation.

    Attributes
    ----------
    payload_len : int, default 1000
        Number of data symbols per spatial stream.
    payload_mod_scheme : str, default "PSK"
        Modulation for payload data (e.g., 'QAM').
    payload_mod_order : int, default 4
        Modulation order for payload (e.g., 16 for 16-QAM).
    payload_seed : int, default 42
        Seed for reproducible payload data generation.
    payload_nu : float, optional
        Maxwell-Boltzmann shaping parameter nu >= 0 for a
        probabilistically shaped QAM payload.  Mutually exclusive with
        ``payload_entropy``.  Requires ``payload_mod_scheme`` to contain
        ``"qam"`` (case-insensitive).  nu = 0 -> uniform QAM.
    payload_entropy : float, optional
        Target entropy in bits per symbol for a PS-QAM payload.  The
        optimal nu is solved numerically via ``mapping.optimal_nu``.
        Mutually exclusive with ``payload_nu``.  Same QAM-only constraint.
    preamble : Preamble, optional
        Structured preamble for synchronization.  For MIMO with ZC sequences,
        each TX stream automatically receives a unique root via
        ``helpers.zc_mimo_root``.
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
    pilot_mod_scheme : str, default "PSK"
        Modulation for pilots.
    pilot_mod_order : int, default 4
        Modulation order for pilots.
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
    **PS-QAM payload**: set either ``payload_nu`` or ``payload_entropy`` (not
    both) together with a QAM ``payload_mod_scheme``.  The MB distribution is
    solved once and cached; access the resulting PMF via the read-only
    ``payload_ps_pmf`` property after the frame has been generated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    payload_len: int = Field(default=1000, gt=0)
    payload_seed: int = 42
    payload_mod_scheme: str = "PSK"
    payload_mod_order: int = Field(default=4, ge=1)
    payload_mod_unipolar: bool = False
    payload_nu: float | None = Field(default=None, ge=0)
    payload_entropy: float | None = Field(default=None, gt=0)

    preamble: Preamble | None = None

    pilot_pattern: Literal["none", "block", "comb"] = "none"
    pilot_period: int = Field(default=0, ge=0)
    pilot_block_len: int = Field(default=0, ge=0)
    pilot_seed: int = 1337
    pilot_mod_scheme: str = "PSK"
    pilot_mod_order: int = Field(default=4, ge=1)
    pilot_mod_unipolar: bool = False
    pilot_gain_db: float = 0.0

    guard_type: Literal["zero", "cp"] = "zero"
    guard_len: int = Field(default=0, ge=0)

    num_streams: int = Field(default=1, ge=1)

    # Internal cache
    _payload_bits: Any | None = PrivateAttr(default=None)
    _payload_symbols: Any | None = PrivateAttr(default=None)
    _payload_ps_pmf: Any | None = PrivateAttr(default=None)
    _pilot_bits: Any | None = PrivateAttr(default=None)
    _pilot_symbols: Any | None = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Validators and Post-Initialization Hooks
    # -------------------------------------------------------------------------

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook.

        Validates that payload_len is evenly divisible by the per-period or
        per-block data count implied by the pilot parameters.  If not, snaps
        payload_len up to the next valid multiple and emits a warning so the
        frame structure always satisfies:
            num_pilot_periods == num_data_periods  (comb)
            num_pilot_blocks  == num_data_blocks   (block)
        """
        import math

        if self.pilot_pattern == "comb" and self.pilot_period > 1:
            data_per_period = self.pilot_period - 1
            if self.payload_len % data_per_period != 0:
                snapped = (
                    math.ceil(self.payload_len / data_per_period) * data_per_period
                )
                logger.warning(
                    "SingleCarrierFrame (comb): payload_len=%s is not "
                    "divisible by data_per_period=%s (pilot_period=%s). "
                    "Snapping payload_len %s -> %s so that "
                    "num_pilot_periods == num_data_periods == %s.",
                    self.payload_len,
                    data_per_period,
                    self.pilot_period,
                    self.payload_len,
                    snapped,
                    snapped // data_per_period,
                )
                self.payload_len = snapped

        elif (
            self.pilot_pattern == "block"
            and self.pilot_period > self.pilot_block_len > 0
        ):
            data_per_block = self.pilot_period - self.pilot_block_len
            if self.payload_len % data_per_block != 0:
                snapped = math.ceil(self.payload_len / data_per_block) * data_per_block
                logger.warning(
                    "SingleCarrierFrame (block): payload_len=%s is not "
                    "divisible by data_per_block=%s (pilot_period=%s, "
                    "pilot_block_len=%s). Snapping payload_len %s -> %s so "
                    "that num_pilot_blocks == num_data_blocks == %s.",
                    self.payload_len,
                    data_per_block,
                    self.pilot_period,
                    self.pilot_block_len,
                    self.payload_len,
                    snapped,
                    snapped // data_per_block,
                )
                self.payload_len = snapped

    @model_validator(mode="after")
    def _check_psqam_fields(self) -> "SingleCarrierFrame":
        if self.payload_nu is not None and self.payload_entropy is not None:
            raise ValueError(
                "payload_nu and payload_entropy are mutually exclusive - specify one or neither."
            )
        if self.payload_nu is not None or self.payload_entropy is not None:
            if "qam" not in self.payload_mod_scheme.lower():
                raise ValueError(
                    f"payload_nu / payload_entropy require a QAM payload modulation, "
                    f"got payload_mod_scheme='{self.payload_mod_scheme}'."
                )
        return self

    @model_validator(mode="after")
    def _check_preamble_streams(self) -> "SingleCarrierFrame":
        if self.preamble is not None and self.preamble.num_streams > 1:
            if self.preamble.num_streams != self.num_streams:
                raise ValueError(
                    f"preamble.num_streams={self.preamble.num_streams} does not match "
                    f"frame.num_streams={self.num_streams}"
                )
        return self

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
        xp = cp if is_cupy_available() else np

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
        """
        Generates and caches payload bits and symbols via the appropriate Signal factory.

        Dispatches to ``generate_psqam``, ``generate_qam``, ``generate_psk``,
        or ``generate_pam`` based on ``payload_mod_scheme`` and the PS
        parameters.  Using the factories as the single source of generation
        logic avoids duplicating bit/symbol generation code here.
        """
        if self._payload_bits is not None:
            return

        scheme = self.payload_mod_scheme.lower()
        is_ps = self.payload_nu is not None or self.payload_entropy is not None

        common: dict[str, Any] = dict(
            num_symbols=self.payload_len,
            sps=1,
            symbol_rate=1.0,
            pulse_shape="none",
            num_streams=self.num_streams,
            seed=self.payload_seed,
        )

        if is_ps:
            sig = generation.generate_psqam(
                order=self.payload_mod_order,
                nu=self.payload_nu,
                entropy=self.payload_entropy,
                **common,
            )
            self._payload_ps_pmf = sig.ps_pmf
        elif "qam" in scheme:
            sig = generation.generate_qam(
                order=self.payload_mod_order,
                unipolar=self.payload_mod_unipolar,
                **common,
            )
        elif "psk" in scheme:
            sig = generation.generate_psk(
                order=self.payload_mod_order,
                **common,
            )
        elif "pam" in scheme or "ask" in scheme:
            sig = generation.generate_pam(
                order=self.payload_mod_order,
                unipolar=self.payload_mod_unipolar,
                **common,
            )
        else:
            sig = generation.generate(
                modulation=self.payload_mod_scheme,
                order=self.payload_mod_order,
                unipolar=self.payload_mod_unipolar,
                **common,
            )

        self._payload_bits = sig.source_bits
        self._payload_symbols = sig.source_symbols

    def _ensure_pilot_generated(self) -> None:
        """
        Generates and caches pilot bits and symbols via the appropriate Signal factory.

        Pilots are always generated with a uniform distribution - PS on pilots
        would destroy the known-reference property required for channel estimation.
        """
        if self._pilot_bits is not None or self.pilot_pattern == "none":
            return

        xp = cp if is_cupy_available() else np
        mask, _ = self._generate_pilot_mask()
        pilot_count = int(xp.sum(mask))
        if pilot_count == 0:
            return

        scheme = self.pilot_mod_scheme.lower()

        common: dict[str, Any] = dict(
            num_symbols=pilot_count,
            sps=1,
            symbol_rate=1.0,
            pulse_shape="none",
            num_streams=self.num_streams,
            seed=self.pilot_seed,
        )

        if "qam" in scheme:
            sig = generation.generate_qam(
                order=self.pilot_mod_order,
                unipolar=self.pilot_mod_unipolar,
                **common,
            )
        elif "psk" in scheme:
            sig = generation.generate_psk(
                order=self.pilot_mod_order,
                **common,
            )
        elif "pam" in scheme or "ask" in scheme:
            sig = generation.generate_pam(
                order=self.pilot_mod_order,
                unipolar=self.pilot_mod_unipolar,
                **common,
            )
        else:
            sig = generation.generate(
                modulation=self.pilot_mod_scheme,
                order=self.pilot_mod_order,
                unipolar=self.pilot_mod_unipolar,
                **common,
            )

        self._pilot_bits = sig.source_bits
        self._pilot_symbols = sig.source_symbols

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
        return self._payload_bits

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
        return self._payload_symbols

    @property
    def payload_ps_pmf(self) -> Any | None:
        """
        Returns the Maxwell-Boltzmann PMF used for PS-QAM payload generation.

        ``None`` for uniform (non-PS) payloads.  Pass this to
        ``metrics.mi`` and ``compute_llr`` after frame equalization
        to compute PS-aware capacity and soft-decision metrics.

        Returns
        -------
        np.ndarray or None
            PMF array of shape ``(payload_mod_order,)`` summing to 1, or ``None``.
        """
        self._ensure_payload_generated()
        return self._payload_ps_pmf

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
        return self._pilot_bits

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
        return self._pilot_symbols

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
        xp = cp if is_cupy_available() else np
        mask, body_length = self._generate_pilot_mask()

        if self.num_streams > 1:
            # Shape: (Channels, Time)
            body = xp.zeros((self.num_streams, body_length), dtype="complex64")

            if self.pilot_pattern != "none":
                pilot_symbols = self.pilot_symbols
                assert pilot_symbols is not None
                # Apply pilot boosting/gain (dB to linear)
                if self.pilot_gain_db != 0.0:
                    pilot_symbols = pilot_symbols * helpers.db_to_linear(
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
                    pilot_symbols = pilot_symbols * helpers.db_to_linear(
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
        xp = cp if is_cupy_available() else np
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
        pulse_shape: str = "rrc",
        filter_span: int = 10,
        rrc_rolloff: float = 0.35,
        rc_rolloff: float = 0.35,
        rise_time: float = 0.0,
        duty_cycle: float = 1.0,
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
        pulse_shape : str, default "rrc"
            Pulse shaping filter type.
        filter_span : int, default 10
            Filter span in symbols.
        rrc_rolloff : float, default 0.35
            Roll-off factor for RRC filter.
        rc_rolloff : float, default 0.35
            Roll-off factor for RC filter.
        rise_time : float, default 0.22
            10%-90% edge transition duration in symbol periods (smoothrect only).
        duty_cycle : float, default 1.0
            FWHM of the Gaussian pulse in symbol periods (gaussian only).
        duty_cycle : float, default 1.0
            Fraction of the symbol period occupied by the pulse (rect/smoothrect).

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
        xp = cp if is_cupy_available() else np
        from .. import mapping
        from .generation import shape_pulse

        sps = require_integer_sps(sps, "SingleCarrierFrame.to_signal()")

        # 1. Shape Body (Payload + Pilots)
        body_symbols = self.body_symbols
        body_samples = shape_pulse(
            symbols=body_symbols,
            sps=sps,
            pulse_shape=pulse_shape,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            rise_time=rise_time,
            duty_cycle=duty_cycle,
        )

        # Normalise body per-channel via helpers.normalize's "dac_peak" mode:
        # max(peak_|I|, peak_|Q|) - a single scale factor that brings the
        # dominant component to 1.0 while preserving the I/Q ratio.
        # Complex-envelope peak normalisation (used in the DSP chain) divides
        # by max(|sample|) instead, leaving components at ≤ 1/√2 ≈ 0.707 for
        # square QAM/PSK whose envelope peak sits at 45°.  Applied
        # per-section (body and preamble separately) so each segment uses
        # the full DAC range regardless of modulation type or constellation
        # phase geometry.
        body_samples = helpers.normalize(body_samples, "dac_peak", axis=-1)

        # 2. Shape Preamble (if present)
        if self.preamble is not None:
            # Use Preamble's to_signal for shaping to reuse logic,
            # but we only need the samples.
            # CRITICAL: Must use EXACT same shaping parameters as body.
            preamble_signal = self.preamble.to_signal(
                sps=sps,
                symbol_rate=symbol_rate,
                pulse_shape=pulse_shape,
                filter_span=filter_span,
                rrc_rolloff=rrc_rolloff,
                rc_rolloff=rc_rolloff,
                rise_time=rise_time,
                duty_cycle=duty_cycle,
            )
            preamble_samples = xp.asarray(preamble_signal.samples)
            # (L*sps,) for SISO  or  (num_streams, L*sps) for MIMO - shape driven by preamble.num_streams

            # I/Q peak normalisation - axis=-1 works for both 1-D and 2-D
            preamble_samples = helpers.normalize(preamble_samples, "dac_peak", axis=-1)

            # Concatenate Preamble + Body
            samples = xp.concatenate([preamble_samples, body_samples], axis=-1)
        else:
            samples = body_samples

        # 3. Apply Guard Interval at sample level
        if self.guard_len > 0:
            guard_len_samples = int(self.guard_len * sps)
            if self.guard_type == "zero":
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
        samples = helpers.normalize(samples, "symbol_power", sps=sps, axis=-1)

        # Resolve ν: payload_nu is set directly; for entropy-specified frames call optimal_nu.
        # payload_ps_pmf is already computed above (body_symbols triggers _ensure_payload_generated).
        if self.payload_nu is not None:
            ps_nu_val: float | None = self.payload_nu
        elif self.payload_entropy is not None:
            ps_nu_val, _ = mapping.optimal_nu(
                self.payload_mod_order, self.payload_entropy
            )
        else:
            ps_nu_val = None

        return Signal(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            mod_scheme=None,
            mod_order=None,
            mod_unipolar=None,
            mod_rz=None,
            source_bits=None,  # extract via frame.get_structure_map() after equalization
            source_symbols=None,  # samples include full frame (preamble + body);
            # extract payload segment via frame.get_structure_map() explicitly.
            ps_pmf=self.payload_ps_pmf,
            ps_nu=ps_nu_val,
            pulse_shape=pulse_shape,
            duty_cycle=duty_cycle,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            rise_time=rise_time,
            signal_type="Single-Carrier Frame",
            frame=self,
        )
