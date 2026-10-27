r"""Laser and carrier linewidth: :func:`estimate_linewidth` and its methods.

The method object chooses the estimator and the kind of record it reads:

* phase trajectories (radians, real-valued) - ``IncrementSlope``,
  ``IncrementSubtract`` and ``BetaSeparation``, e.g. the output of
  ``carrier_phase_trajectory`` or ``separate_drift_phase_noise``;
* delayed self-heterodyne / self-homodyne beat records (real photocurrent or
  complex IQ, array or Signal) - ``DshFmPsd``, ``DshIncrement`` and
  ``DshLorentzian`` (see :mod:`commkit.analysis.interferometry`).

Every method returns a :class:`LinewidthEstimate`: the linewidth in Hz plus
that method's diagnostics, ready for the ``plotting.analysis`` plots.  The
values are host floats (SISO) or ``(C,)`` arrays (MIMO), like the metrics:
the fits run host-side on plot-sized reductions.
"""

from typing import Any

from ...backend import ArrayType, dispatch
from ...core._signal_adapter import adapt_signal
from ...core.signal import Signal
from .dsh import (
    DshFmPsd,
    DshIncrement,
    DshLorentzian,
    _dsh_fm_psd,
    _dsh_increment,
    _dsh_lorentzian,
    _warn_if_phase_trajectory,
)
from .phase import (
    BetaSeparation,
    IncrementSlope,
    IncrementSubtract,
    _beta_separation,
    _increment_slope,
    _increment_subtract,
)
from .result import LinewidthEstimate

__all__ = [
    "BetaSeparation",
    "DshFmPsd",
    "DshIncrement",
    "DshLorentzian",
    "IncrementSlope",
    "IncrementSubtract",
    "LinewidthEstimate",
    "estimate_linewidth",
]


# -----------------------------------------------------------------------------
# Verb
# -----------------------------------------------------------------------------

_PHASE_METHODS: dict[type, Any] = {
    IncrementSlope: _increment_slope,
    IncrementSubtract: _increment_subtract,
    BetaSeparation: _beta_separation,
}
_BEAT_METHODS: dict[type, Any] = {
    DshFmPsd: _dsh_fm_psd,
    DshIncrement: _dsh_increment,
    DshLorentzian: _dsh_lorentzian,
}


def estimate_linewidth(
    samples: ArrayType | Signal,
    method: Any,
    *,
    sampling_rate: float | None = None,
) -> LinewidthEstimate:
    r"""Estimate a laser or carrier linewidth with the given method.

    Parameters
    ----------
    samples : array_like or Signal
        The record the method reads, ``(N,)`` or ``(C, N)`` (channels are
        independent records):

        * ``IncrementSlope``, ``IncrementSubtract``, ``BetaSeparation``: a
          real phase trajectory in radians (``carrier_phase_trajectory``,
          ``separate_drift_phase_noise``, ``dsh_phase``).  Complex input
          raises.
        * ``DshFmPsd``, ``DshIncrement``, ``DshLorentzian``: a delayed
          self-heterodyne beat record, real (single photodetector) or complex
          (IQ front end), as an array or a Signal.  A phase trajectory is
          real too, so it cannot be rejected by type: real input logs a
          warning when its power sits near DC (below half the carrier, or
          below ``sampling_rate/400`` without ``f_shift``) or when it has no
          beat line, a peak 10 dB above the spectrum at half and 1.5 times
          its frequency (a drift-removed residual, or a beat too weak or too
          wide for its carrier to measure).
    method : IncrementSlope, IncrementSubtract, BetaSeparation, DshFmPsd, DshIncrement or DshLorentzian
        The estimator and its parameters.
    sampling_rate : float, optional
        Sampling rate of the record in Hz (a fact: the symbol rate for a
        per-symbol phase trajectory).  Taken from a Signal; required for
        arrays.  A value that disagrees with the Signal raises.

    Returns
    -------
    LinewidthEstimate
        ``value`` in Hz and the method's diagnostics, on the host.

    Raises
    ------
    TypeError
        If ``method`` is not one of the methods above.
    ValueError
        On complex input to a phase method, a missing ``sampling_rate`` or a
        delay below one sample.
    """
    name = "estimate_linewidth()"
    adapter = adapt_signal(samples, function_name=name)
    fs = float(adapter.resolve_fact("sampling_rate", sampling_rate))
    x, xp, _ = dispatch(adapter.array)
    kind = type(method)
    if kind in _PHASE_METHODS:
        if xp.iscomplexobj(x):
            raise ValueError(
                f"{name}: {kind.__name__} reads a real phase trajectory in "
                "radians, got complex samples (extract it with "
                "carrier_phase_trajectory or dsh_phase, or use a Dsh* method "
                "for a beat record)."
            )
        return _PHASE_METHODS[kind](x, method, fs)
    if kind in _BEAT_METHODS:
        _warn_if_phase_trajectory(x, getattr(method, "f_shift", None), fs)
        return _BEAT_METHODS[kind](x, method, fs)
    raise TypeError(
        f"{name}: method must be IncrementSlope, IncrementSubtract, "
        "BetaSeparation, DshFmPsd, DshIncrement or DshLorentzian, got "
        f"{kind.__name__}."
    )
