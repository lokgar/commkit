"""Tests for the synchronization preamble (commkit.core.preamble).

Preambles inside frames are tested in test_frame.py.
"""

from typing import Any

from commkit.core import Preamble
from commkit.filtering import RRC


def test_preamble_to_signal() -> None:
    """Preamble.to_signal() builds a standalone Signal without a reference."""
    preamble = Preamble(sequence_type="barker", num_symbols=13)
    sig = preamble.to_signal(sps=4, symbol_rate=1e6, pulse=RRC(0.35))

    assert len(sig.samples) == 13 * 4
    assert sig.constellation is None
    assert sig.reference is None


def test_preamble_device(backend_device: str, xp: Any) -> None:
    sig = Preamble(sequence_type="barker", num_symbols=13).to_signal(
        sps=4, symbol_rate=1e6, device=backend_device
    )
    assert isinstance(sig.samples, xp.ndarray)
