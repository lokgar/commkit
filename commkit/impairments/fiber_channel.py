"""The fiber channel: linear and nonlinear impairments of optical fiber.

Linear: :func:`apply_chromatic_dispersion`, :func:`apply_pmd` (a single
differential-group-delay section) and :func:`apply_polarization_mixing` (a
rotation of the state of polarization).

Nonlinear: :func:`apply_fiber_propagation` propagates a field over amplified
spans of a :class:`Fiber` with the symmetric split-step Fourier method:
self-phase modulation (and, for a dual-polarization field, the Manakov
cross-phase modulation between the polarizations) together with chromatic
dispersion, its slope, attenuation, polarization-mode dispersion and
amplifier noise.

**Records.**  Every function here works in the frequency domain on the
whole record, which it treats as one period of a periodic sequence: the
dispersed tails of the last symbols wrap onto the first, as when an AWG
plays the record in a loop.  The receiver undoes such a record exactly with
``periodic=True``; a capture's receiver uses the default.

The compensation counterparts are
:func:`commkit.filtering.correct_chromatic_dispersion` and digital
backpropagation, :func:`commkit.filtering.correct_fiber_propagation`.  Import
the public names from :mod:`commkit.impairments`, their documented home.
"""

import math
from types import ModuleType
from typing import Any

import numpy as np

from .. import _fft
from .._array import require_channels, restore_1d
from .._dispersion import apply_dispersion
from .._fiber import Fiber, check_link, jones_section, propagate, to_watts
from .._random import RNG
from ..backend import ArrayType, dispatch
from ..core._signal_adapter import S, adapt_signal
from ..logger import logger
from ..math import _phasor

__all__ = [
    "Fiber",
    "apply_chromatic_dispersion",
    "apply_fiber_propagation",
    "apply_pmd",
    "apply_polarization_mixing",
]


def _jones_rotation(theta: float, xp: ModuleType, dtype: Any) -> ArrayType:
    """Static 2x2 Jones rotation matrix ``R(theta) = [[cos, -sin], [sin, cos]]``.

    Shared builder for ``apply_pmd``'s rotation into the PSP frame,
    ``R(-theta)`` (the section of ``_fiber.jones_section``), and
    ``apply_polarization_mixing``'s static-``theta`` path.  The time-varying
    path of ``apply_polarization_mixing`` builds no matrix (one 2x2 matrix
    per sample is not worth building); it applies the same rotation by
    broadcasting per sample.
    """
    c = math.cos(theta)
    s = math.sin(theta)
    return xp.array([[c, -s], [s, c]], dtype=dtype)


def apply_pmd(
    samples: S,
    *,
    dgd: float,
    theta: float = 0.0,
    sampling_rate: float | None = None,
) -> S:
    """Apply first-order polarization-mode dispersion (PMD) to a dual-pol field.

    One section with the frequency-dependent Jones matrix

        H(f) = R(θ) · diag(exp(-jπfτ), exp(+jπfτ)) · R(-θ)

    with the differential group delay (DGD) ``τ``, the orientation ``θ`` of
    the principal states of polarization (PSPs) and the rotation
    ``R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]``.  ``R(-θ)`` projects the
    field onto the PSPs, which are delayed by ``±τ/2`` (a differential delay
    of ``τ``), and ``R(θ)`` rotates back.  ``θ`` is part of the PMD model,
    not a separate rotation: with ``τ = 0`` the section is the identity.
    For a frequency-independent rotation use
    :func:`apply_polarization_mixing`; for PMD distributed along a fiber,
    :func:`apply_fiber_propagation` with ``Fiber(pmd_ps_sqrt_km=...)``.

    Parameters
    ----------
    samples : array_like or Signal
        Dual-polarization field, ``(2, N)``.
    dgd : float
        Differential group delay ``τ`` in seconds.
    theta : float, default 0.0
        PSP orientation ``θ`` in radians: 0 aligns the PSPs with the X and Y
        axes (no coupling between them), ``π/4`` couples them most.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.

    Returns
    -------
    ndarray or Signal
        The distorted field, same shape, dtype and device as the input.

    Raises
    ------
    ValueError
        If the input is not ``(2, N)``.

    Examples
    --------
    >>> x = sig.samples  # shape (2, N), dual-pol
    >>> distorted = apply_pmd(x, dgd=5e-12, theta=np.pi / 5, sampling_rate=fs)
    >>> distorted = apply_pmd(sig, dgd=5e-12, theta=np.pi / 5)  # Signal input
    """
    signal_adapter = adapt_signal(samples, function_name="apply_pmd()")
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)

    logger.info("Applying PMD (DGD=%.2e s, theta=%.3f rad).", dgd, theta)

    x, xp, _ = dispatch(signal_adapter.array)

    require_channels(
        x, 2, name="samples", description="dual-pol input with shape (2, N)"
    )

    N = x.shape[1]
    freqs = xp.fft.fftfreq(N, d=1.0 / sampling_rate)

    # H(f) = R(+θ) · diag(e^{-jπfτ}, e^{+jπfτ}) · R(-θ): one PMD section with
    # the real rotation U = R(-θ) into the PSP frame (U^H = R(+θ)).
    S_F = _fft.fft(x, axis=-1)  # (2, N)
    phase = xp.pi * freqs * dgd  # float64; responses in the spectrum's precision
    h_neg = _phasor(-phase, S_F.dtype, xp)
    out_F = jones_section(
        S_F, _jones_rotation(-theta, np, np.float64), h_neg, xp.conj(h_neg)
    )
    result = _fft.ifft(out_F, axis=-1)

    # Preserve input dtype (ifft may produce complex128 from complex64 input)
    if result.dtype != x.dtype:
        result = result.astype(x.dtype)

    return signal_adapter.wrap_samples(result)


def apply_polarization_mixing(
    samples: S,
    *,
    theta: float | ArrayType,
    drift_rate: float = 0.0,
    sampling_rate: float | None = None,
) -> S:
    """Rotate the state of polarization, statically or over time.

    A frequency-independent real rotation,
    ``[x'[n], y'[n]] = R(θ[n]) · [x[n], y[n]]`` with
    ``R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]``: unlike :func:`apply_pmd`,
    there is no differential group delay.  For testing polarization-diverse
    receivers and modeling slow SOP drift.

    Parameters
    ----------
    samples : array_like or Signal
        Dual-polarization field, ``(2, N)``.
    theta : float or array_like
        Rotation angle in radians: a scalar for one rotation of every sample
        (a ramp with ``drift_rate``), or ``(N,)`` angles, one per sample.
    drift_rate : float, default 0.0
        Linear SOP drift for a scalar ``theta``, in rad/s:
        ``θ(t) = theta + drift_rate · t``, so the drift is a property of the
        channel whatever the sampling rate.  Not with an array ``theta``.
    sampling_rate : float, optional
        Sampling rate in Hz, for ``drift_rate``.  Taken from the Signal;
        required for array input with a drift.  A value that disagrees with
        the Signal raises.

    Returns
    -------
    ndarray or Signal
        The rotated field, same shape, dtype and device as the input.

    Raises
    ------
    ValueError
        If the input is not ``(2, N)``, an array ``theta`` is not ``(N,)``
        or comes with a ``drift_rate``.

    Examples
    --------
    >>> # Static 45° rotation
    >>> rotated = apply_polarization_mixing(samples, theta=np.pi / 4)

    >>> # Slow linear SOP drift: 10 krad/s
    >>> drifted = apply_polarization_mixing(signal, theta=0.0, drift_rate=1e4)
    """
    signal_adapter = adapt_signal(samples, function_name="apply_polarization_mixing()")

    logger.info(
        "Applying polarization mixing (theta=%s, drift=%.3g rad/s).",
        theta if np.ndim(theta) == 0 else "array",
        drift_rate,
    )
    if drift_rate != 0.0:
        if np.ndim(theta) != 0:
            raise ValueError(
                "apply_polarization_mixing(): drift_rate needs a scalar theta; "
                "an array theta gives every angle."
            )
        sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)

    x, xp, _ = dispatch(signal_adapter.array)

    require_channels(
        x, 2, name="samples", description="dual-pol input with shape (2, N)"
    )

    N = x.shape[1]

    # Build angle trajectory
    if np.ndim(theta) == 0:
        scalar_theta = float(theta)
        if drift_rate != 0.0:
            assert sampling_rate is not None
            step = drift_rate / sampling_rate  # rad per sample
            angles = xp.arange(N, dtype=xp.float64) * step + scalar_theta
        else:
            # Static: scalar path - avoid building (N,) array
            R = _jones_rotation(scalar_theta, xp, x.dtype)
            result = R @ x
            if result.dtype != x.dtype:
                result = result.astype(x.dtype)
            return signal_adapter.wrap_samples(result)
    else:
        angles = xp.asarray(theta, dtype=xp.float64)
        if angles.shape != (N,):
            raise ValueError(
                f"theta array must have shape (N,)={(N,)}, got {angles.shape}."
            )

    # Time-varying: vectorized per-sample rotation via broadcasting
    # R(θ[n]) applied to each column of x
    # Angles in float64, the rotation in the samples' precision.
    cos_t = xp.cos(angles).astype(x.real.dtype)  # (N,)
    sin_t = xp.sin(angles).astype(x.real.dtype)  # (N,)

    Ex, Ey = x[0], x[1]
    Ex_out = cos_t * Ex - sin_t * Ey
    Ey_out = sin_t * Ex + cos_t * Ey

    result = xp.stack([Ex_out, Ey_out])  # (2, N)

    if result.dtype != x.dtype:
        result = result.astype(x.dtype)

    return signal_adapter.wrap_samples(result)


def apply_chromatic_dispersion(
    samples: S,
    *,
    dispersion_ps_nm_km: float,
    fiber_length_km: float,
    center_wavelength_nm: float | None = None,
    sampling_rate: float | None = None,
) -> S:
    """Apply chromatic dispersion (CD) in the frequency domain.

    Multiplies the spectrum by ``H(ω) = exp(-j β₂ L ω² / 2)`` with
    ``β₂ = -D λ² / (2π c)``: the dispersion parameter ``D``, the center
    wavelength ``λ``, the speed of light ``c`` and the fiber length ``L``
    (second-order dispersion only, without loss or nonlinearity; see
    :func:`apply_fiber_propagation` for those).

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    dispersion_ps_nm_km : float
        Dispersion parameter ``D`` in ps/(nm km); standard single-mode fiber
        (G.652) has 17 at 1550 nm.  Positive is anomalous (``β₂ < 0``).
    fiber_length_km : float
        Fiber length in km.
    center_wavelength_nm : float, optional
        Carrier wavelength in nm (e.g. 1550 for the C band).  Taken from an optical Signal's
        carrier (``domain="OPT"``, ``c / center_frequency``); required for
        array input and Signals of another domain.  A value that disagrees
        with the optical Signal raises.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.

    Returns
    -------
    ndarray or Signal
        The dispersed samples, same shape, dtype and device as the input.

    See Also
    --------
    commkit.filtering.correct_chromatic_dispersion :
        Remove CD in the receiver (electronic dispersion compensation).

    Examples
    --------
    >>> distorted = apply_chromatic_dispersion(
    ...     x, dispersion_ps_nm_km=17.0, fiber_length_km=80.0,
    ...     center_wavelength_nm=1550.0, sampling_rate=fs)
    >>> distorted = apply_chromatic_dispersion(  # optical Signal: its carrier
    ...     sig, dispersion_ps_nm_km=17.0, fiber_length_km=80.0)
    """
    signal_adapter = adapt_signal(samples, function_name="apply_chromatic_dispersion()")
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    center_wavelength_nm = signal_adapter.resolve_wavelength_nm(center_wavelength_nm)

    logger.info(
        "Applying CD (D=%s ps/nm/km, L=%s km, λ=%s nm).",
        dispersion_ps_nm_km,
        fiber_length_km,
        center_wavelength_nm,
    )

    result = apply_dispersion(
        signal_adapter.array,
        sampling_rate=sampling_rate,
        dispersion_ps_nm=dispersion_ps_nm_km * fiber_length_km,
        center_wavelength_nm=center_wavelength_nm,
        inverse=False,
    )
    return signal_adapter.wrap_samples(result)


def apply_fiber_propagation(
    samples: S,
    *,
    fiber: Fiber,
    span_length_km: float,
    num_spans: int,
    launch_power_dbm: float,
    center_wavelength_nm: float | None = None,
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
    center_wavelength_nm : float, optional
        Carrier wavelength in nm, at which the fiber's dispersion is given
        and which sets the photon energy of the ASE.  Taken from an optical
        Signal's carrier (``domain="OPT"``, ``c / center_frequency``); required for
        array input and Signals of another domain.  A value that disagrees
        with the optical Signal raises.
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
    center_wavelength_nm = signal_adapter.resolve_wavelength_nm(center_wavelength_nm)
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
