"""Containers and factories: :class:`Signal`, frames and generation.

- :class:`Signal` and :class:`Reference` (:mod:`commkit.core.signal`): a
  waveform with its properties, description and ground truth.
- :class:`SingleCarrierFrame`, its :class:`FrameStructure` masks and
  :func:`extract_payload` (:mod:`commkit.core.frame`): structured frames with
  pilots, guard intervals and MIMO streams.
- :class:`Preamble` (:mod:`commkit.core.preamble`): the synchronization
  preamble before a frame's body.
- :func:`generate` (:mod:`commkit.core.generation`): random symbols,
  pulse-shaped into a Signal; also re-exported as ``commkit.generate``.
- :class:`SignalInput`: the array-or-Signal argument of a Signal-aware
  function, for writing your own with the library's conventions.
"""

from ._signal_input import SignalInput
from .frame import FrameStructure, SingleCarrierFrame, extract_payload
from .generation import generate
from .preamble import Preamble
from .signal import Reference, Signal

__all__ = [
    "FrameStructure",
    "Preamble",
    "Reference",
    "Signal",
    "SignalInput",
    "SingleCarrierFrame",
    "extract_payload",
    "generate",
]
