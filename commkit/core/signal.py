"""
Core signal processing abstractions and data containers.

This module defines the primary data structures used throughout the library.
It provides high-level abstractions for handling raw IQ samples, physical
layer metadata, and complex frame structures.

All core classes are built on Pydantic for robust validation and support
transparent backend switching between CPU (NumPy) and GPU (CuPy).
"""

import types
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


from .. import helpers
from ..backend import (
    ArrayType,
    from_jax,
    get_array_module,
    get_scipy_module,
    is_cupy_available,
    to_device,
    to_jax,
)
from ..logger import logger


class Signal(BaseModel):
    """
    Primary container for digital baseband or RF signals.

    The `Signal` class encapsulates complex-valued IQ samples along with the
    physical layer metadata (sampling rate, modulation, etc.) required for
    comprehensive Digital Signal Processing (DSP) pipelines. It supports
    seamless switching between CPU (NumPy) and GPU (CuPy) backends.

    Attributes
    ----------
    samples : array_like
        The complex IQ samples.
        Shape: (N_samples,) for SISO or (N_channels, N_samples) for MIMO.
        The last dimension is always assumed to be Time.
    sampling_rate : float
        Sampling frequency in Hertz (Hz). Must be > 0.
    symbol_rate : float
        Symbol frequency (Baud rate) in Hertz (Hz). Must be > 0.
    mod_scheme : str, optional
        Identifier for the modulation format (e.g., 'QPSK', '16QAM').
        For frame-generated signals this is ``None``; modulation is carried by
        ``frame.payload_mod_scheme`` instead.
    mod_order : int, optional
        The modulation order. Similar to `mod_scheme`, it might be `None`
        if multiple modes are present within a frame. See
        ``frame.payload_mod_order``.
    mod_unipolar : bool, optional
        If True, uses a unipolar constellation (e.g., 0 to M-1).
    mod_rz : bool, optional
        If True, uses Return-to-Zero (RZ) signaling.
    source_bits : array_like, optional
        The original binary data that generated the signal (full wire order).
        Populated by ``Signal.generate`` and the factory methods.
        For frame-generated signals this is ``None``; extract the payload
        segment via ``frame.get_structure_map()`` and construct a plain
        ``Signal`` with the relevant ``source_bits`` for per-segment metrics.
    source_symbols : array_like, optional
        The mapped constellation symbols before pulse shaping (full wire
        order). Same scoping note as ``source_bits``.
    ps_pmf : array_like of float, optional
        Maxwell-Boltzmann PMF of shape ``(M,)`` for PS-QAM signals.
        Set automatically by ``generate_psqam``.  When present, the
        normalization of ``source_symbols`` is skipped (PS symbols have
        intentionally lower average energy than uniform QAM), and
        ``mi``, ``gmi``, and ``plot_constellation`` use the
        non-uniform prior automatically.  ``None`` for all other modulations.
    ps_nu : float, optional
        Maxwell-Boltzmann shaping parameter ν ≥ 0.  Set automatically
        alongside ``ps_pmf`` by ``generate_psqam`` and
        ``SingleCarrierFrame.to_signal``.  ν = 0 is uniform QAM (never
        stored; ``ps_nu`` is ``None`` for non-PS signals).  When the
        signal was specified via ``entropy``, ν is the numerically solved
        value returned by ``mapping.optimal_nu``.
    pulse_shape : str, optional
        Name of the pulse shaping filter (e.g., ``'rrc'``, ``'rect'``,
        ``'gaussian'``).
    filter_span : int
        Span of the pulse-shaping filter in symbols.
    rrc_rolloff : float
        Roll-off factor for the Root-Raised Cosine (RRC) filter.
    rc_rolloff : float
        Roll-off factor for the Raised Cosine (RC) filter.
    duty_cycle : float
        Pulse width in symbol periods. Meaning depends on pulse shape:
        ``rect``/``smoothrect`` - on-time fraction (incl. ramps);
        ``gaussian`` - FWHM. For NRZ signals this is always 1.0 internally;
        only meaningful when ``mod_rz=True``. Stored so
        ``generate_shaping_taps()`` can reconstruct the correct taps.
    rise_time : float
        Edge transition duration in symbol periods for ``rect`` and
        ``smoothrect``. For ``rect``: linear ramp duration (flat top =
        ``duty_cycle - 2 * rise_time``). For ``smoothrect``: 10%-90%
        erf-edge duration. Ignored for all other pulse shapes.
    spectral_domain : {"BASEBAND", "PASSBAND", "INTERMEDIATE"}
        The signal's current placement in the frequency spectrum.
    physical_domain : {"DIG", "RF", "OPT"}
        The physical transmission domain: ``'DIG'`` (Digital), ``'RF'``
        (Radio), ``'OPT'`` (Optical).
    center_frequency : float
        The carrier or center frequency in Hz.
    digital_frequency_offset : float
        Cumulative digital frequency shift applied to the signal in Hz.
    pilot_tone_frequency : numpy.ndarray
        Per-channel pilot-tone frequencies in Hz, as a 1-D ``float64`` array
        (one entry per channel; length 1 for a SISO / shared tone).  Any scalar
        or sequence assigned is coerced to this array form, so the field is
        handled uniformly like the other array fields.  Set by
        ``add_pilot_tone``; distinct per-channel tones enable e.g. tone-based
        polarization demultiplexing.
    pilot_tone_power_ratio_db : numpy.ndarray
        Per-channel pilot-to-signal power ratio (PSR) in dB of the added
        tone(s), in the same 1-D ``float64`` array form as
        ``pilot_tone_frequency``.  Set by ``add_pilot_tone``; travels with the
        signal through save/load.
    signal_type : {"Single-Carrier Frame", "OFDM Frame", "Preamble"}, optional
        Human-readable label for the signal structure. Informational only.
    frame : Frame, optional
        The frame that generated the signal.
    resolved_symbols : array_like, optional
        Symbols at 1 SPS, normalised to unit average power.
        Populated by ``resolve_symbols()``.  Call only on plain signals
        (non-frame); frame signals contain mixed preamble/pilot/payload that
        may have different modulations or gains - resolve after splitting.
    resolved_bits : array_like, optional
        Hard-decision bits demapped from ``resolved_symbols``.
        Populated by ``demap_symbols_hard()``.

    Notes
    -----
    **Frame-generated signals**: ``SingleCarrierFrame.to_signal`` sets
    ``self.frame`` but leaves ``source_symbols`` and
    ``source_bits`` as ``None``.  The receive workflow is:

    1. Run timing / FOE / CPR / equalization on the frame signal.
    2. Use ``frame.get_structure_map()`` to slice sample/symbol indices.
    3. Extract each segment and build a plain ``Signal`` with the appropriate
       ``source_symbols``/``source_bits`` before calling ``resolve_symbols()``,
       ``evm()``, ``ber()``, etc.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    samples: Any
    sampling_rate: float = Field(..., gt=0)
    symbol_rate: float = Field(..., gt=0)

    mod_scheme: str | None = None
    mod_order: int | None = None
    mod_unipolar: bool | None = None
    mod_rz: bool | None = None

    source_bits: Any | None = None
    source_symbols: Any | None = None
    ps_pmf: Any | None = None  # (M,) PMF over constellation; set only for PS-QAM
    ps_nu: float | None = None  # MB shaping parameter ν; set only for PS-QAM

    pulse_shape: str | None = None
    filter_span: int = Field(default=10, ge=1)
    rrc_rolloff: float = Field(default=0.35, ge=0, le=1)
    rc_rolloff: float = Field(default=0.35, ge=0, le=1)
    duty_cycle: float = Field(default=1.0, gt=0, le=1)
    rise_time: float = Field(default=0.0, ge=0)

    spectral_domain: Literal["BASEBAND", "PASSBAND", "INTERMEDIATE"] = "BASEBAND"
    physical_domain: Literal["DIG", "RF", "OPT"] = "DIG"

    center_frequency: float = Field(default=0, ge=0)
    digital_frequency_offset: float | None = None
    pilot_tone_frequency: Any | None = None
    pilot_tone_power_ratio_db: Any | None = None

    # Human-readable label for the signal structure
    signal_type: Literal["Single-Carrier Frame", "OFDM Frame", "Preamble"] | None = None

    # Back-reference to the SingleCarrierFrame that generated this signal (set by
    # SingleCarrierFrame.to_signal()). Enables frame-aware convenience methods
    # (correct_timing, frame-aware equalizers) without requiring the caller to re-supply
    # the frame object.  Excluded from serialisation (numpy arrays inside frame
    # duplicate samples data and are not JSON-serialisable).
    frame: Any | None = Field(default=None, exclude=True, repr=False)

    # Resolved data from processing (1 SPS, normalized - populated by resolve_symbols())
    resolved_symbols: Any | None = Field(default=None, repr=False)
    resolved_bits: Any | None = Field(default=None, repr=False)

    # -------------------------------------------------------------------------
    # Validators and Post-Initialization Hooks
    # -------------------------------------------------------------------------

    @field_validator("pilot_tone_frequency", "pilot_tone_power_ratio_db", mode="before")
    @classmethod
    def _coerce_pilot_field(cls, v: Any) -> Any:
        """Coerce pilot metadata to a 1-D per-channel ``float64`` array.

        ``None`` passes through; anything else (scalar, sequence, or array)
        becomes a 1-D ``float64`` ``np.ndarray`` - one value per channel - so the
        field is handled uniformly like the other array fields (always an array,
        never a scalar or list).  A scalar becomes a length-1 array.
        """
        if v is None:
            return None
        return np.asarray(v, dtype=np.float64).reshape(-1)

    @field_validator("samples", mode="before")
    @classmethod
    def validate_samples(cls, v: Any) -> Any:
        """
        Validates and coerces the samples input into a backend-compatible array.

        This validator ensures that the input is converted to a NumPy or CuPy array
        and enforces a (Channels, Time) shape convention for multidimensional inputs.

        Parameters
        ----------
        v : array_like
            Input samples (list, tuple, NumPy array, CuPy array, or JAX array).

        Returns
        -------
        array_like
            The validated samples as a NumPy or CuPy array.

        Raises
        ------
        ValueError
            If the input cannot be converted to a supported array type or has
            unsupported dimensions (> 2).

        Notes
        -----
        The library enforces a **Time-Last** convention: (N_channels, N_samples)
        or simply (N_samples,) for 1D signals. This aligns with C-contiguous
        memory layout which is generally more performant for time-axis operations.
        """
        arr = helpers.validate_array(v, name="samples")

        # Check shape conventions
        # We enforce Time-Last convention: (Channels, Time) or (Time,) for 1D.
        # This aligns better with C-contiguous operations on the time axis (last axis)
        # which is critical for CuPy performance/stability.

        if arr.ndim > 2:
            raise ValueError(
                f"Samples array has {arr.ndim} dimensions. "
                "Only 1D (SISO) or 2D (MIMO/Dual-Pol) arrays are supported."
            )

        if arr.ndim == 2:
            # Check dimensions to guess orientation
            s0, s1 = arr.shape
            # If dim0 (rows) > dim1 (cols) and dim0 >> 10, it's likely (Time, Channels)
            # We want (Channels, Time).
            if s0 > s1 and s0 > 32:  # Heuristic: Time dim usually > 32
                logger.warning(
                    "Samples shape is %s. Converting to Time-Last convention "
                    "(N_channels=%s, N_samples=%s). Please provide input as "
                    "(N_channels, N_samples) for MIMO signals.",
                    arr.shape,
                    s1,
                    s0,
                )
                arr = arr.T  # Transpose to (Channels, Time)

            # If shape is (2, 2), ambiguous but assumes (Channels, Time)
            # If s1 > s0, likely already correct.

        return arr

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook to handle metadata derivation and device placement.

        This method automatically derives `source_symbols` from `source_bits` if
        modulation parameters are present, and moves the signal samples to
        the GPU if a compatible device is available.
        """
        # Bit-first: derive symbols from bits if not provided
        if self.source_bits is not None and self.source_symbols is None:
            if self.mod_scheme and self.mod_order:
                from .. import mapping

                self.source_symbols = mapping.map_bits(
                    self.source_bits,
                    self.mod_scheme,
                    self.mod_order,
                    unipolar=self.mod_unipolar or False,
                )

        # Ensure source_symbols are normalized to unit average power for consistent metrics.
        # For MIMO (multichannel), we normalize per-stream (axis=-1) to ensure each stream
        # independently adheres to E_s=1, facilitating per-stream metric calculation.
        # Skip for PS-QAM: symbols are exact constellation points whose sample average
        # power is intentionally < 1 (MB weights inner points more). Scaling them would
        # break the correspondence with ps_pmf.
        if self.source_symbols is not None and self.ps_pmf is None:
            self.source_symbols = helpers.normalize(
                self.source_symbols, mode="average_power", axis=-1
            )

        # Default to GPU if available and supported
        if is_cupy_available():
            self.to("gpu")

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def print_info(self) -> None:
        """
        Prints a formatted summary of the signal's physical and digital properties.

        In Jupyter/IPython environments, this renders as an HTML table. In standard
        shells, it outputs a plain-text table via the logger.

        Sections
        --------
        **Signal** - always shown: type, waveform, rate, shape, backend.
        **Frame structure** - shown when ``frame`` carries frame metadata
        (preamble, payload, pilots, guard).
        **Reference data** - shows which symbol/bit arrays and frame object are
        attached (determines which of ``ber()``, ``evm()``
        can be called without extra arguments).
        """

        # -- helpers ----------------------------------------------------------
        def _yn(v) -> str:
            return "yes" if v is not None else "no"

        # Modulation: frame signals store it on the frame.
        frame = getattr(self, "frame", None)
        mod_scheme = self.mod_scheme or (getattr(frame, "payload_mod_scheme", None))
        mod_order = self.mod_order or (getattr(frame, "payload_mod_order", None))
        mod_unipolar = self.mod_unipolar or (
            getattr(frame, "payload_mod_unipolar", False)
        )
        mod_str = (
            f"{mod_scheme or 'None'} / {mod_order or 'None'}"
            f"{' (UNIPOL)' if mod_unipolar else ''}"
            f"{' (RZ)' if self.mod_rz else ''}"
        )
        bit_rate = (
            helpers.format_si(self.symbol_rate * np.log2(mod_order), "bps")
            if mod_order
            else "None"
        )

        # -- Section 1: Signal ---------------------------------------------
        sig_type_label = self.signal_type or "Signal"

        rows: list[tuple[str, str]] = [
            ("Signal type", sig_type_label),
            ("Spectral domain", self.spectral_domain),
            ("Physical domain", self.physical_domain),
            ("Modulation", mod_str),
            ("Symbol rate", helpers.format_si(self.symbol_rate, "Baud")),
            ("Bit rate", bit_rate),
            ("Sampling rate", helpers.format_si(self.sampling_rate, "Hz")),
            ("Samples per symbol", f"{self.sps:.2f}"),
            ("Pulse shape", self.pulse_shape.upper() if self.pulse_shape else "None"),
        ]

        if self.ps_pmf is not None and mod_order:
            rows.append(
                (
                    "PS shaping (ν)",
                    f"{self.ps_nu:.4f}" if self.ps_nu is not None else "unknown",
                )
            )

        rows += [
            ("Duration", helpers.format_si(self.duration, "s")),
            ("Center frequency", helpers.format_si(self.center_frequency, "Hz")),
        ]

        if self.digital_frequency_offset is not None:
            rows.append(
                (
                    "Frequency offset",
                    helpers.format_si(self.digital_frequency_offset, "Hz"),
                )
            )

        def _pilot_row(value, fmt) -> str:
            # value is a 1-D per-channel array (validator-coerced).
            return ", ".join(fmt(float(x)) for x in value)

        if self.pilot_tone_frequency is not None:
            rows.append(
                (
                    "Pilot tone frequency",
                    _pilot_row(
                        self.pilot_tone_frequency,
                        lambda x: helpers.format_si(x, "Hz"),
                    ),
                )
            )

        if self.pilot_tone_power_ratio_db is not None:
            rows.append(
                (
                    "Pilot tone power",
                    _pilot_row(self.pilot_tone_power_ratio_db, lambda x: f"{x:.1f} dB"),
                )
            )

        rows += [
            ("Backend", self.backend.upper()),
            (
                "Configuration",
                "SISO" if self.num_streams == 1 else f"MIMO ({self.num_streams}x)",
            ),
            ("Samples shape", str(self.samples.shape)),
        ]

        # -- Section 2: Structure info (content varies by signal_type) -------
        if self.signal_type == "Preamble" and self.frame is not None:
            preamble = getattr(self.frame, "preamble", None)
            if preamble is not None:
                rows.append(("--- Preamble info", ""))
                preamble_str = (
                    f"{preamble.sequence_type.upper()}  len={preamble.length}"
                )
                if preamble.sequence_type == "zc":
                    preamble_str += f"  root={preamble.root}"
                rows.append(("Sequence", preamble_str))

        elif self.signal_type == "Single-Carrier Frame" and self.frame is not None:
            frame = self.frame
            rows.append(("--- Frame structure", ""))

            if hasattr(frame, "preamble") and frame.preamble is not None:
                p = frame.preamble
                preamble_str = f"{p.sequence_type.upper()}  len={p.length}"
                if p.sequence_type == "zc":
                    preamble_str += f"  root={p.root}"
                rows.append(("Preamble", preamble_str))
            else:
                rows.append(("Preamble", "none"))

            if hasattr(frame, "payload_len") and frame.payload_len is not None:
                rows.append(("Payload length", f"{frame.payload_len} symbols"))

            pilot_pattern = getattr(frame, "pilot_pattern", "none")
            if pilot_pattern != "none":
                mask, _ = frame._generate_pilot_mask()
                pilot_count = (
                    int(np.sum(mask))
                    if hasattr(frame, "_generate_pilot_mask")
                    else None
                )
                pilot_period = getattr(frame, "pilot_period", None)
                pilot_gain_db = getattr(frame, "pilot_gain_db", None)

                pilot_str = (
                    f"{pilot_pattern}"
                    + (f"  count={pilot_count}" if pilot_count else "")
                    + (f"  period={pilot_period}" if pilot_period else "")
                    + (
                        f"  gain={pilot_gain_db} dB"
                        if pilot_gain_db is not None
                        else ""
                    )
                )
                rows.append(("Pilots", pilot_str))
            else:
                rows.append(("Pilots", "none"))

            if hasattr(frame, "guard_len") and frame.guard_len:
                rows.append(("Guard", f"{frame.guard_type}  len={frame.guard_len}"))
            else:
                rows.append(("Guard", "none"))

        # -- Section 3: Reference data -------------------------------------
        rows.append(("--- Reference data", ""))
        rows.append(("source_symbols", _yn(self.source_symbols)))
        rows.append(("source_bits", _yn(self.source_bits)))
        rows.append(("ps_pmf", _yn(self.ps_pmf)))
        rows.append(("frame attached", _yn(self.frame)))

        # -- Section 4: Resolved data --------------------------------------
        rows.append(("--- Resolved data", ""))
        rows.append(("resolved_symbols", _yn(self.resolved_symbols)))
        rows.append(("resolved_bits", _yn(self.resolved_bits)))

        # -- Render --------------------------------------------------------
        # Rich HTML table in Jupyter; plain-text table everywhere else. IPython
        # is an optional dependency (the ``notebook`` extra) - degrade
        # gracefully to the plain-text path when it is not installed.
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display

            ipy = get_ipython()
            in_kernel = ipy is not None and "IPKernelApp" in ipy.config
        except ImportError:
            in_kernel = False

        if in_kernel:
            body = "".join(
                f"<tr><td><b>{prop}</b></td><td>{val}</td></tr>" for prop, val in rows
            )
            display(HTML(f"<table>{body}</table>"))
        else:
            width = max(len(prop) for prop, _ in rows)
            text = "\n".join(f"{prop.ljust(width)}  {val}" for prop, val in rows)
            logger.info("\n%s", text)

    def clone(self) -> "Signal":
        """
        Create a fully independent copy of the `Signal` instance.

        Samples, provenance arrays, resolved caches, and the attached frame
        are all deep-copied. Ordinary DSP transforms should use
        :meth:`replace_samples` instead.

        Returns
        -------
        Signal
            A new signal object with identical data and metadata.
        """
        return self.model_copy(deep=True)

    def _shallow_clone(self) -> "Signal":
        """Return a new container sharing all array and frame references.

        This is an internal building block for operations that replace a
        non-waveform field. Public callers should normally choose between
        :meth:`clone` and :meth:`replace_samples`.
        """
        return self.model_copy(deep=False)

    def replace_samples(
        self,
        samples: Any,
        *,
        _preserve_resolved: bool = False,
        **metadata: Any,
    ) -> "Signal":
        """Return a shallow metadata copy with a replacement sample buffer.

        This is the functional update operation for waveform transforms. The
        old sample buffer is never copied: provenance arrays and the attached
        frame remain shared with the input unless explicitly replaced through
        ``metadata``. Assignment validation is applied to both the replacement
        samples and every metadata override.

        Shared provenance arrays and frame objects are not guaranteed immutable:
        later mutation through either container can affect the other. A supplied
        sample view may also share memory with the old waveform. Use `clone()`
        when independent backing data is required.

        Resolved symbols and bits are derived caches and are invalidated by
        default because changing waveform samples can make them stale. Internal
        transforms that can prove the caches remain valid may pass
        ``_preserve_resolved=True``.

        Parameters
        ----------
        samples : array_like
            Replacement waveform samples.
        _preserve_resolved : bool, default False
            Internal opt-in to retain ``resolved_symbols`` and ``resolved_bits``.
        **metadata
            Explicit Signal field updates such as ``sampling_rate``.

        Returns
        -------
        Signal
            A new Signal sharing unchanged metadata with this instance.
        """
        new = self._shallow_clone()
        new.samples = samples
        if not _preserve_resolved:
            new.resolved_symbols = None
            new.resolved_bits = None
        for key, value in metadata.items():
            setattr(new, key, value)
        return new

    def to(self, device: str) -> "Signal":
        """
        Transfers signal data to the target device (CPU or GPU).

        Parameters
        ----------
        device : {"CPU", "GPU"}
            The target device. Case-insensitive.

        Returns
        -------
        Signal
            Returns self for method chaining.

        Raises
        ------
        ImportError
            If GPU is requested but CuPy is not installed/functional.
        """
        self.samples = to_device(self.samples, device)
        return self

    def export_samples_to_jax(self, device: str | None = None) -> Any:
        """
        Exports the signal samples to a JAX array.

        Ensures zero-copy transfer to JAX when possible, preserving the
        device affinity of the underlying samples unless otherwise specified.

        Parameters
        ----------
        device : {"CPU", "GPU", "TPU"}, optional
            Target JAX device. If None, it targets the device matching the
            signal's current backend (CPU or GPU).

        Returns
        -------
        jax.Array
            JAX array containing signal samples.
            Shape: (N_channels, N_samples) or (N_samples,).
        """
        # If device is not explicitly requested, use the signal's backend
        target_device = device if device is not None else self.backend
        return to_jax(self.samples, device=target_device)

    def update_samples_from_jax(self, jax_array: Any) -> "Signal":
        """
        Updates signal samples from a JAX array.

        Converts the JAX array back to the signal's original backend (NumPy
        or CuPy) to maintain consistent state. Invalidates resolved symbols and
        bits because they were derived from the previous waveform.

        Parameters
        ----------
        jax_array : jax.Array
            Input JAX array. Shape must match signal's expected shape.

        Returns
        -------
        Signal
            Returns self for method chaining.
        """
        original_backend = self.backend
        # Convert JAX array to backend-compatible array
        # from_jax will return NumPy (for CPU/TPU) or CuPy (for GPU)
        new_samples = from_jax(jax_array)

        # Ensure we move the data back to the original backend if it differs
        # (e.g., if signal was GPU but jax_array was on CPU/TPU)
        self.samples = to_device(new_samples, original_backend)
        self.resolved_symbols = None
        self.resolved_bits = None

        return self

    def time_axis(self) -> ArrayType:
        """
        Generates the time vector associated with signal samples.

        Returns
        -------
        array_like
            Time axis in seconds, starting at 0.
            Shape: (N_samples,).
        """
        n_samples = self.samples.shape[-1]
        return self.xp.arange(0, n_samples) / self.sampling_rate

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def xp(self) -> types.ModuleType:
        """
        Access the active array backend (NumPy or CuPy).

        This property allows for backend-agnostic code by returning the
        appropriate module based on where the samples currently reside.

        Returns
        -------
        module
            `numpy` if data is on CPU, `cupy` if on GPU.
        """
        return get_array_module(self.samples)

    @property
    def sp(self) -> types.ModuleType:
        """
        Access the signal processing module (`scipy` or `cupyx.scipy`).

        Returns
        -------
        module
            Appropriate signal processing library for the current backend.
        """
        return get_scipy_module(self.xp)

    @property
    def backend(self) -> str:
        """
        Returns the current computational backend name.

        Returns
        -------
        {"CPU", "GPU"}
            A string indicating the device location of samples.
        """
        return "GPU" if self.xp == cp else "CPU"

    @property
    def num_streams(self) -> int:
        """
        Returns the number of spatial or polarization streams.

        Returns
        -------
        int
            1 for SISO signals, N for MIMO/Dual-Pol signals.
        """
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[0]

    @property
    def duration(self) -> float:
        """
        Returns the total duration of the signal.

        Returns
        -------
        float
            Duration in seconds.
        """
        if self.samples.ndim == 1:
            return self.samples.shape[0] / self.sampling_rate
        return self.samples.shape[-1] / self.sampling_rate

    @property
    def sps(self) -> float:
        """
        Samples per symbol.

        Returns
        -------
        float
            Ratio of sampling rate to symbol rate.
        """
        return self.sampling_rate / self.symbol_rate

    @property
    def bits_per_symbol(self) -> int | None:
        """
        Bits per symbol for the active modulation scheme.

        Returns
        -------
        int or None
            Calculated as ``log2(modulation_order)``.
        """
        if self.mod_order:
            return int(np.log2(self.mod_order))
        return None
