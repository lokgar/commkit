"""Synchronization preambles: :class:`Preamble`.

A preamble is a known Barker or Zadoff-Chu sequence placed before a frame's
body (see :class:`~commkit.core.SingleCarrierFrame`) and found again by
:func:`commkit.timing.estimate_timing`.  Multi-channel Zadoff-Chu preambles
take a distinct root per channel.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._sequences import barker_sequence, zadoff_chu_sequence, zc_mimo_root
from ..backend import ArrayType, to_device
from ..filtering.pulses import Pulse
from . import generation
from ._signal_adapter import require_device, require_integer_sps
from .signal import Signal

__all__ = ["Preamble"]


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


def _check_int(name: str, value: Any, low: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value < low:
        raise ValueError(f"{name} must be >= {low}, got {value}.")
