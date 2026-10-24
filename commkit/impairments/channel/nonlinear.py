"""Nonlinear fiber propagation: the Kerr effect with dispersion, loss and PMD.

:func:`apply_fiber_propagation` propagates a field over amplified spans of a
:class:`Fiber` with the symmetric split-step Fourier method: self-phase
modulation (and, for a dual-polarization field, the Manakov cross-phase
modulation between the polarizations) together with chromatic dispersion,
its slope, attenuation, polarization-mode dispersion and amplifier noise.
Its compensation counterpart, digital backpropagation, is
:func:`commkit.filtering.correct_fiber_propagation`.
"""

import math

from ..._array import restore_1d
from ..._fiber import Fiber, check_link, propagate, to_watts
from ..._random import RNG
from ...backend import dispatch
from ...core._signal_adapter import S, adapt_signal
from ...logger import logger

__all__ = ["Fiber", "apply_fiber_propagation"]


def apply_fiber_propagation(
    samples: S,
    *,
    fiber: Fiber,
    span_length_km: float,
    num_spans: int,
    launch_power_dbm: float,
    center_wavelength_nm: float = 1550.0,
    noise_figure_db: float | None = None,
    steps_per_span: int = 100,
    sampling_rate: float | None = None,
    rng: RNG = None,
) -> S:
    """Propagate a field over ``num_spans`` amplified spans of nonlinear fiber.

    Solves the nonlinear Schrödinger equation for ``(N,)`` input (one
    polarization) or the Manakov equation for ``(2, N)`` input (two
    polarizations, Kerr coefficient ``8/9 · γ``) with the symmetric
    split-step Fourier method.  Each span is followed by a lumped amplifier
    whose gain restores the span loss exactly and which, with
    ``noise_figure_db``, adds amplified spontaneous emission (ASE).

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband field, ``(N,)`` or ``(2, N)``, in any units: its
        average power (summed over the polarizations) is taken as the launch
        power.  Real input is propagated as a complex field.
    fiber : Fiber
        Loss, dispersion (and slope), Kerr coefficient and PMD of the fiber.
        PMD needs ``(2, N)`` input.
    span_length_km : float
        Length of every span in km.
    num_spans : int
        Number of spans, each followed by an amplifier.
    launch_power_dbm : float
        Average launch power of the field in dBm, summed over the
        polarizations.  It sets the strength of the Kerr effect and of the
        ASE relative to the signal.
    center_wavelength_nm : float, default 1550.0
        Carrier wavelength in nm, at which the fiber's dispersion is given
        and which sets the photon energy of the ASE.
    noise_figure_db : float, optional
        Amplifier noise figure in dB (``>= 0``).  ``None`` (default) gives
        noiseless amplifiers.  Each amplifier of gain ``G`` adds white ASE
        of spectral density ``(F G - 1) hν / 2`` per polarization over the
        simulated bandwidth (the sampling rate).
    steps_per_span : int, default 100
        Equal split-step steps per span.  The error falls as the square of
        the step length; also the number of PMD sections per span.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.
    rng : int, numpy.random.Generator or None
        Random source of the ASE (drawn on the field's device) and of the
        PMD waveplates (drawn on the host).  Unused without either.

    Returns
    -------
    ndarray or Signal
        The field at the output of the last amplifier, in the input's units:
        the input's power stands for the launch power, so a link whose
        amplifiers restore every span loss returns the input's power plus
        the ASE.  Complex, on the input's device; ``(N,)`` or ``(2, N)`` as
        the input.  A :class:`Signal` returns a new :class:`Signal`.

    Raises
    ------
    ValueError
        On input that is not ``(N,)`` or ``(2, N)``, PMD with a single
        polarization, a negative noise figure, or input without power.

    See Also
    --------
    commkit.filtering.correct_fiber_propagation :
        Digital backpropagation, the inverse of this model.
    apply_chromatic_dispersion : the linear, single-span special case.

    Notes
    -----
    The field obeys (CommKit's engineering sign convention, the complex
    conjugate of the textbook form)

    ``dA/dz = -α/2 A - j β₂/2 ∂²A/∂t² + β₃/6 ∂³A/∂t³ - j γ |A|² A``,

    so a spectrum is multiplied by ``exp(-j β₂ ω² L / 2)``, as in
    :func:`apply_chromatic_dispersion`, and the Kerr phase is
    ``exp(-j γ |A|² L)``.  PMD is a waveplate model with one Haar-random
    section per step, of differential group delay
    ``pmd · sqrt(3π h / 8)``, which gives a Maxwellian DGD of mean
    ``pmd_ps_sqrt_km · sqrt(L)``.

    Precision follows the input.  complex64 (the default storage) loses
    about 6e-8 per FFT pair: 1e-6 relative after 2 x 80 km at 7 steps per
    span, 6e-4 (about -64 dB) after 100 x 80 km at 100 steps, far below the
    SNR of a real link.  For more dynamic range (nonlinear interference at
    low launch power, GN-model checks, step convergence or DBP-inverse
    studies) pass complex128, e.g. ``sig.replace(samples=sig.samples.astype(
    "complex128"))``: about 1e-12 over the same 100 spans, at 1.6x the CPU
    time (the GPU then uses its plain CuPy stages).  Equalizers and LLRs
    downstream work in complex64/float32 either way.

    Nonlinear spectral broadening must fit in the sampling rate; a warning
    is logged when the field after the first span has more than 0.1 % of
    its power beyond 0.4 f_s.

    Examples
    --------
    >>> rx = apply_fiber_propagation(
    ...     sig, fiber=Fiber(), span_length_km=80.0, num_spans=10,
    ...     launch_power_dbm=2.0, noise_figure_db=5.0, rng=1)
    """
    name = "apply_fiber_propagation()"
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
    if noise_figure_db is not None and not (
        math.isfinite(noise_figure_db) and noise_figure_db >= 0
    ):
        raise ValueError(
            f"{name}: noise_figure_db must be >= 0 (or None), got {noise_figure_db!r}."
        )
    x, xp, _ = dispatch(signal_adapter.array)
    field, was_1d, scale = to_watts(x, xp, launch_power_dbm, name)
    if fiber.pmd_ps_sqrt_km > 0 and field.shape[0] != 2:
        raise ValueError(
            f"{name}: PMD (fiber.pmd_ps_sqrt_km > 0) couples two polarizations; "
            "pass a (2, N) field."
        )
    logger.info(
        "Fiber propagation: %d x %.4g km, %.3g dBm, %d steps/span, NF=%s.",
        link.num_spans,
        link.span_length_km,
        launch_power_dbm,
        link.steps_per_span,
        noise_figure_db,
    )
    out = propagate(
        field,
        xp,
        link=link,
        sampling_rate=sampling_rate,
        direction=1,
        noise_figure_db=noise_figure_db,
        rng=rng,
    )
    out *= out.real.dtype.type(1.0 / scale)
    return signal_adapter.wrap_samples(restore_1d(was_1d, out))
