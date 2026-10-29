"""Known pilots of the pilot-aided blind equalizers: :class:`EqualizerPilots`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._array import broadcast_channels
from ..backend import to_device
from ..core.frame import _frame_pilots

__all__ = ["EqualizerPilots"]


@dataclass(frozen=True)
class EqualizerPilots:
    """Known pilot symbols at known positions, for ``cma``, ``rde``,
    ``block_cma`` and ``block_rde``.

    At the pilot positions the blind equalizers take the LMS error towards
    the known symbol, which resolves their phase ambiguity there; the data
    positions stay blind.

    Parameters
    ----------
    values : array_like
        The pilot symbols in transmission order, ``(K,)`` (the same on every
        channel) or ``(C, K)``, on the scale of the unit-power
        constellation (without the boost).
    mask : array_like of bool
        ``(N_sym,)``, True at the ``K`` pilot positions among the
        equalizer's output symbols.
    gain_db : float, default 0.0
        Pilot boost in dB relative to the payload.  The received samples at
        the pilot positions are scaled down by it before the input's RMS
        normalization, so boosted pilots neither inflate the RMS nor bias
        the Godard target or ring radii of the data positions.

    Notes
    -----
    :meth:`from_frame` takes all three from a ``SingleCarrierFrame``, so the
    gain cannot disagree with the frame's.
    """

    values: np.ndarray
    mask: np.ndarray
    gain_db: float = 0.0

    def __post_init__(self) -> None:
        values = np.array(to_device(self.values, "cpu"), dtype=np.complex64)
        mask = np.array(to_device(self.mask, "cpu"))
        if mask.ndim != 1 or mask.dtype.kind not in "bui":
            raise ValueError(f"mask must be a 1-D boolean array, got {mask.shape}.")
        mask = mask.astype(bool)
        count = int(mask.sum())
        if values.ndim not in (1, 2) or values.shape[-1] != count:
            raise ValueError(
                f"values must have shape (K,) or (C, K) with K = {count} pilots "
                f"(the True entries of mask), got {values.shape}."
            )
        for name, arr in (("values", values), ("mask", mask)):
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        object.__setattr__(self, "gain_db", float(self.gain_db))

    @classmethod
    def from_frame(
        cls, frame: Any, *, include_preamble: bool = True
    ) -> EqualizerPilots:
        """The pilots of a ``SingleCarrierFrame``: positions, symbols, gain.

        Parameters
        ----------
        frame : SingleCarrierFrame
            A frame with pilots.
        include_preamble : bool, default True
            Count positions from the frame start (the preamble's first
            symbol, as after ``correct_timing(mode="slice")``), the mask
            spanning the whole frame; ``False`` counts from the end of the
            preamble.

        Returns
        -------
        EqualizerPilots

        Raises
        ------
        ValueError
            If the frame has no pilots.
        """
        _, values = _frame_pilots(frame, include_preamble, "EqualizerPilots")
        mask = frame.get_structure_map(include_preamble=include_preamble)["pilots"]
        return cls(values, mask, frame.pilot_gain_db)

    def _dense(self, num_channels: int, num_symbols: int, function_name: str) -> Any:
        """The dense ``(C, N_sym)`` complex64 reference (zeros at the data)
        and the uint8 mask the kernels take."""
        if self.mask.size != num_symbols:
            raise ValueError(
                f"{function_name}: pilots.mask has {self.mask.size} entries for "
                f"{num_symbols} output symbols."
            )
        values = self.values
        if values.ndim == 1:
            values = broadcast_channels(values, num_channels, np, name="pilots.values")
        elif values.shape[0] != num_channels:
            raise ValueError(
                f"{function_name}: pilots.values has {values.shape[0]} channels "
                f"for {num_channels}."
            )
        reference = np.zeros((num_channels, num_symbols), dtype=np.complex64)
        reference[:, self.mask] = values
        return reference, self.mask.astype(np.uint8)
