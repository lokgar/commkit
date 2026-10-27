"""Fiber compensation: chromatic dispersion and digital backpropagation.

:func:`correct_chromatic_dispersion` is the inverse of
:func:`commkit.impairments.apply_chromatic_dispersion`, and
:func:`correct_fiber_propagation` (digital backpropagation) the inverse of
:func:`commkit.impairments.apply_fiber_propagation`.
"""

import numpy as np

from .._array import restore_1d
from .._dispersion import apply_dispersion
from .._fiber import Fiber, check_link, propagate, to_watts
from ..backend import dispatch
from ..core._signal_adapter import S, adapt_signal
from ..logger import logger

__all__ = [
    "correct_chromatic_dispersion",
    "correct_fiber_propagation",
]

# -----------------------------------------------------------------------------
# CHROMATIC DISPERSION (Signal-aware)
# -----------------------------------------------------------------------------


def correct_chromatic_dispersion(
    samples: S,
    *,
    dispersion_ps_nm_km: float,
    fiber_length_km: float,
    center_wavelength_nm: float,
    sampling_rate: float | None = None,
) -> S:
    """Compensate chromatic dispersion electronically: the inverse fiber response.

    Multiplies the spectrum by ``H(ω) = exp(+j β₂ L ω² / 2)`` with
    ``β₂ = -D λ² / (2π c)``, which undoes
    :func:`commkit.impairments.apply_chromatic_dispersion` exactly.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    dispersion_ps_nm_km : float
        Fiber dispersion parameter D in ps / (nm km) (SMF-28: 17 at 1550 nm).
    fiber_length_km : float
        Fiber length in km.
    center_wavelength_nm : float
        Carrier wavelength in nm.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.

    Returns
    -------
    ndarray or Signal
        Compensated samples, same shape, dtype and device as the input.

    Examples
    --------
    >>> sig = correct_chromatic_dispersion(
    ...     sig, dispersion_ps_nm_km=17.0, fiber_length_km=80.0,
    ...     center_wavelength_nm=1550.0)
    """
    signal_adapter = adapt_signal(
        samples, function_name="correct_chromatic_dispersion()"
    )
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    logger.info(
        "Correcting CD (D=%s ps/nm/km, L=%s km, λ=%s nm).",
        dispersion_ps_nm_km,
        fiber_length_km,
        center_wavelength_nm,
    )
    result = apply_dispersion(
        signal_adapter.array,
        sampling_rate=sampling_rate,
        dispersion_ps_nm_km=dispersion_ps_nm_km,
        fiber_length_km=fiber_length_km,
        center_wavelength_nm=center_wavelength_nm,
        inverse=True,
    )
    return signal_adapter.wrap_samples(result)


def correct_fiber_propagation(
    samples: S,
    *,
    fiber: Fiber,
    span_length_km: float,
    num_spans: int,
    launch_power_dbm: float,
    center_wavelength_nm: float = 1550.0,
    steps_per_span: int = 100,
    nonlinear_scale: float = 1.0,
    sampling_rate: float | None = None,
) -> S:
    """Undo a nonlinear fiber link by digital backpropagation (DBP).

    Runs the split-step solver of
    :func:`commkit.impairments.apply_fiber_propagation` backward: the spans
    in reverse order and every step inverted (negated dispersion and Kerr
    phase).  The solver works on the loss-normalized field ``u = A e^{αz/2}``,
    on which an amplifier that restores the span loss is the identity, so
    neither the loss nor the gain is applied as a factor: the loss enters
    only through the Kerr phase of each step, ``γ (e^{-αz_{k-1}} -
    e^{-αz_k}) / α`` times the power, which DBP negates.  With the link's own
    ``steps_per_span`` and ``nonlinear_scale=1`` this is the exact inverse
    of a noiseless, PMD-free link, up to rounding: the symmetric split step
    is time-reversible.  That is the default.

    Fewer steps (a common low-complexity choice is one per span) make DBP an
    approximation, and ``nonlinear_scale`` must then be tuned below 1 for
    the best signal quality.  Measured on 4 x 80 km of standard fiber at
    32 GBd, 6 dBm, NF 5 dB (distortion -16 dB after dispersion compensation
    alone): 10 steps per span with ξ = 1 reach the ASE floor (-30 dB); one
    step per span gains 2 dB with ξ = 0.5 but nothing with ξ = 1.

    Parameters
    ----------
    samples : array_like or Signal
        Received field, ``(N,)`` or ``(2, N)``, in any units: its average
        power (summed over the polarizations) is taken as the launch power,
        so the output of :func:`apply_fiber_propagation` or a receiver
        normalized to unit power both work.
    fiber : Fiber
        The link's fiber.  Its PMD is not undone (a realization is random
        and unknown to the receiver); leave it to the adaptive MIMO
        equalizer.
    span_length_km : float
        Length of every span in km.
    num_spans : int
        Number of spans to backpropagate.
    launch_power_dbm : float
        The link's launch power in dBm, summed over the polarizations.
    center_wavelength_nm : float, default 1550.0
        Carrier wavelength in nm.
    steps_per_span : int, default 100
        Backward steps per span (the forward model's default).
    nonlinear_scale : float, default 1.0
        Factor ξ on the backward Kerr phase; 1 inverts the Kerr effect.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.

    Returns
    -------
    ndarray or Signal
        The backpropagated field in the input's units, complex, on the
        input's device.  A :class:`Signal` returns a new :class:`Signal`.

    See Also
    --------
    correct_chromatic_dispersion : the linear (``γ = 0``) special case.

    Notes
    -----
    DBP cannot undo the nonlinear interaction of the signal with ASE added
    along the link, nor random PMD: both bound its gain on a real link.

    Examples
    --------
    >>> from commkit.impairments import Fiber
    >>> rx = correct_fiber_propagation(
    ...     rx, fiber=Fiber(), span_length_km=80.0, num_spans=10,
    ...     launch_power_dbm=2.0)
    """
    name = "correct_fiber_propagation()"
    signal_adapter = adapt_signal(samples, function_name=name)
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    link = check_link(
        fiber=fiber,
        span_length_km=span_length_km,
        num_spans=num_spans,
        steps_per_span=steps_per_span,
        center_wavelength_nm=center_wavelength_nm,
        function_name=name,
    )
    if not np.isfinite(nonlinear_scale):
        raise ValueError(
            f"{name}: nonlinear_scale must be finite, got {nonlinear_scale!r}."
        )
    x, xp, _ = dispatch(signal_adapter.array)
    field, was_1d, scale = to_watts(x, xp, launch_power_dbm, name)
    logger.info(
        "DBP: %d x %.4g km, %.3g dBm, %d steps/span, xi=%.3g.",
        link.num_spans,
        link.span_length_km,
        launch_power_dbm,
        link.steps_per_span,
        nonlinear_scale,
    )
    out = propagate(
        field,
        xp,
        link=link,
        sampling_rate=sampling_rate,
        direction=-1,
        nonlinear_scale=float(nonlinear_scale),
    )
    out *= out.real.dtype.type(1.0 / scale)
    return signal_adapter.wrap_samples(restore_1d(was_1d, out))
