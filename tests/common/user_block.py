"""A Signal-aware transform written as user code, on the public API only.

The contract registry checks it like a library transform, so the
conventions :class:`commkit.core.SignalInput` promises hold for code
outside the package: array in, array out on the same device; Signal in, a
new Signal out; a conflicting sampling rate raises.
"""

from typing import Any

from commkit.backend import dispatch
from commkit.core import SignalInput

__all__ = ["apply_gain_ramp"]


def apply_gain_ramp(
    samples: Any, *, ramp_db_per_s: float, sampling_rate: float | None = None
) -> Any:
    """Apply a gain that grows by ``ramp_db_per_s`` dB per second."""
    inp = SignalInput(samples, "apply_gain_ramp()")
    fs = inp.sampling_rate(sampling_rate)
    x, xp, _ = dispatch(inp.samples)
    t = xp.arange(x.shape[-1]) / fs
    gain = (10 ** (ramp_db_per_s * t / 20)).astype(x.real.dtype)
    return inp.wrap(x * gain)
