"""Containers and factories: :class:`Signal`, frames and generation.

- :class:`Signal` and :class:`Reference` (:mod:`commkit.core.signal`): a
  waveform with its facts, description and ground truth.
- :class:`Preamble`, :class:`SingleCarrierFrame` and :func:`extract_payload`
  (:mod:`commkit.core.frame`): structured frames with pilots, guard intervals
  and MIMO streams.
- :func:`generate` (:mod:`commkit.core.generation`): random symbols,
  pulse-shaped into a Signal; also re-exported as ``commkit.generate``.
"""

from .frame import Preamble, SingleCarrierFrame, extract_payload
from .generation import generate
from .signal import Reference, Signal

__all__ = [
    "Preamble",
    "Reference",
    "Signal",
    "SingleCarrierFrame",
    "extract_payload",
    "generate",
]
