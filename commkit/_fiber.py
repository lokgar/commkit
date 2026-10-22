"""Split-step Fourier fiber propagation (private).

Single source of the nonlinear fiber model used by
``impairments.apply_fiber_propagation`` (the channel) and
``filtering.correct_fiber_propagation`` (digital backpropagation, DBP):

- :class:`Fiber`: the fiber's description (loss, dispersion and its slope,
  Kerr coefficient, PMD).
- :func:`propagate`: the symmetric split-step solver over amplified spans,
  forward or backward.

Model.  ``(N,)`` input is one polarization and follows the scalar NLSE; a
``(2, N)`` field follows the Manakov equation, whose Kerr term averages the
fast birefringence of the fiber into ``8/9 · γ · (|x|² + |y|²)``.  In
CommKit's sign convention (``_dispersion``: propagation multiplies a spectrum
by ``exp(-j β₂ ω² L / 2)``) the field obeys

    dA/dz = -α/2 A - j β₂/2 ∂²A/∂t² + β₃/6 ∂³A/∂t³ - j γ |A|² A,

the complex conjugate of the textbook (Agrawal) form, so the Kerr phase is
``exp(-j γ |A|² L)``.  Matching the dispersion sign is what keeps solitons
in anomalous fiber and makes DBP an inverse.

Solver.  Each span of length ``L`` is cut into ``K`` equal steps.  A step
is ``D(h/2) N(h) D(h/2)``: the nonlinear phase uses the power at the step's
midpoint and the effective length ``2 sinh(αh/2)/α``, so the scheme is
second order (and exact for a continuous wave).  Equal steps measured about
3x more accurate than logarithmic ones (equal nonlinear phase per step) on a
32 GBd, 80 km span at 4-10 dBm for every step count from 10 to 200: the
long last steps of the logarithmic rule cost more dispersion accuracy than
its balanced Kerr phase gains.  Neighbouring half steps are merged, so a
step costs one FFT pair.  A lumped amplifier after every
span restores the span loss exactly and, with a noise figure, adds ASE.

PMD is the waveplate (coarse-step) model: every step carries one section, a
Haar-random Jones rotation and a differential group delay
``τ_k = D_PMD · sqrt(3π h_k / 8)``, which gives a Maxwellian DGD of mean
``D_PMD · sqrt(L)`` over many sections.  The waveplates are drawn on the
host, like the other parameters of a realization.

DBP runs the same solver backward: spans in reverse, each step's operators
inverted (negated lengths and Kerr phase, scaled by ``nonlinear_scale``).
With the forward steps and ``nonlinear_scale = 1`` it inverts a noiseless,
PMD-free link to rounding error, because the symmetric step is
time-reversible.

Devices.  The CPU runs NumPy stages on ``scipy.fft`` (``_fft``), with the
PMD stage in one Numba pass.  CuPy input runs three fused elementwise
kernels (linear stage, linear stage with a PMD section, Kerr stage) on
in-place, unscaled cuFFT transforms, so each stage reads and writes the
field once and a span allocates nothing; a PMD-free span is captured once
into a CUDA graph and replayed.  Any failure falls back to the plain CuPy
stages, which complex128 input also uses.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

from . import _fft
from ._array import as_2d
from ._dispersion import _SPEED_OF_LIGHT, beta2_length, beta3_length
from ._random import as_generator, complex_normal
from .backend import ArrayType
from .logger import logger

_PLANCK = 6.62607015e-34  # J s

# A first span whose spectrum has more than this fraction of its power
# beyond 0.4 f_s is likely to alias its nonlinear broadening.
_ALIASING_FRACTION = 1e-3


@dataclass(frozen=True)
class Fiber:
    """A single-mode fiber: loss, dispersion, Kerr nonlinearity and PMD.

    All values are at the signal's center wavelength.  The defaults describe
    standard single-mode fiber (ITU-T G.652) at 1550 nm, with third-order
    dispersion and PMD switched off.

    Parameters
    ----------
    attenuation_db_km : float, default 0.2
        Power attenuation in dB/km, ``>= 0``.
    dispersion_ps_nm_km : float, default 17.0
        Dispersion parameter D in ps/(nm km); positive is anomalous
        (``β₂ < 0``).
    dispersion_slope_ps_nm2_km : float, optional
        Dispersion slope S = dD/dλ in ps/(nm² km), about 0.057 for standard
        fiber; it matters for wideband (multi-channel) fields.  A value gives
        the exact third-order term ``β₃ = (S + 2D/λ) (λ²/2πc)²`` (note that
        ``S = 0``, a D constant in wavelength, still has ``β₃ ≠ 0``).
        ``None`` (default) sets ``β₃ = 0``: second-order dispersion only, as
        :func:`commkit.impairments.apply_chromatic_dispersion`.
    nonlinearity_per_w_km : float, default 1.3
        Kerr coefficient γ in 1/(W km), ``>= 0``.
    pmd_ps_sqrt_km : float, default 0.0
        PMD coefficient in ps/sqrt(km), ``>= 0``: the mean differential group
        delay after a length L is ``pmd_ps_sqrt_km · sqrt(L)``.  Typical
        modern fiber is below 0.1.  Needs a ``(2, N)`` field.
    """

    attenuation_db_km: float = 0.2
    dispersion_ps_nm_km: float = 17.0
    dispersion_slope_ps_nm2_km: float | None = None
    nonlinearity_per_w_km: float = 1.3
    pmd_ps_sqrt_km: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "attenuation_db_km",
            "dispersion_ps_nm_km",
            "dispersion_slope_ps_nm2_km",
            "nonlinearity_per_w_km",
            "pmd_ps_sqrt_km",
        ):
            value = getattr(self, name)
            if value is None and name == "dispersion_slope_ps_nm2_km":
                continue
            if not isinstance(value, int | float | np.integer | np.floating) or (
                not math.isfinite(value)
            ):
                raise ValueError(
                    f"Fiber: {name} must be a finite number, got {value!r}."
                )
            object.__setattr__(self, name, float(value))
        for name in ("attenuation_db_km", "nonlinearity_per_w_km", "pmd_ps_sqrt_km"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"Fiber: {name} must be >= 0, got {getattr(self, name)}."
                )

    @property
    def alpha_per_km(self) -> float:
        """Power attenuation coefficient α in 1/km (``e^{-αL}`` loss)."""
        return self.attenuation_db_km * math.log(10.0) / 10.0


@dataclass(frozen=True)
class _Link:
    """Validated link parameters shared by the forward model and DBP."""

    fiber: Fiber
    span_length_km: float
    num_spans: int
    steps_per_span: int
    center_wavelength_nm: float


def check_link(
    *,
    fiber: Any,
    span_length_km: float,
    num_spans: int,
    steps_per_span: int,
    center_wavelength_nm: float,
    function_name: str,
) -> _Link:
    """Validate the link arguments of the public functions."""
    if not isinstance(fiber, Fiber):
        raise TypeError(
            f"{function_name}: fiber must be a Fiber, got {type(fiber).__name__}; "
            "use e.g. Fiber(attenuation_db_km=0.2)."
        )
    for name, value in (
        ("span_length_km", span_length_km),
        ("center_wavelength_nm", center_wavelength_nm),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{function_name}: {name} must be > 0, got {value!r}.")
    for name, value in (("num_spans", num_spans), ("steps_per_span", steps_per_span)):
        if not isinstance(value, int | np.integer) or value < 1:
            raise ValueError(
                f"{function_name}: {name} must be a positive integer, got {value!r}."
            )
    return _Link(
        fiber,
        float(span_length_km),
        int(num_spans),
        int(steps_per_span),
        float(center_wavelength_nm),
    )


def to_watts(
    x: ArrayType, xp: ModuleType, launch_power_dbm: float, function_name: str
) -> tuple[ArrayType, bool, float]:
    """``(field, was_1d, scale)``: a complex ``(C, N)`` working copy of ``x``
    scaled to sqrt(W), with its average power (summed over the
    polarizations) at ``launch_power_dbm``.  Divide the result by ``scale``
    to return to the input's units."""
    if not math.isfinite(launch_power_dbm):
        raise ValueError(
            f"{function_name}: launch_power_dbm must be finite, got {launch_power_dbm!r}."
        )
    x2, was_1d = as_2d(x, name="samples")
    if x2.shape[0] not in (1, 2):
        raise ValueError(
            f"{function_name}: samples must be one polarization (N,) or two "
            f"(2, N), got shape {x.shape}."
        )
    dtype = np.result_type(x2.dtype, np.complex64)
    field = x2.astype(dtype, copy=True)
    power = float(xp.sum(xp.mean(field.real**2 + field.imag**2, axis=-1)))
    if not power > 0:
        raise ValueError(f"{function_name}: samples carry no power to launch.")
    scale = math.sqrt(1e-3 * 10.0 ** (launch_power_dbm / 10.0) / power)
    field *= field.real.dtype.type(scale)
    return field, was_1d, scale


def _nonlinear_lengths(h: np.ndarray, alpha_per_km: float) -> np.ndarray:
    """Effective nonlinear length of each step relative to its midpoint,
    ``2 sinh(αh/2) / α`` (km); ``h`` for a lossless fiber."""
    if alpha_per_km == 0.0:
        return h.copy()
    return 2.0 * np.sinh(alpha_per_km * h / 2.0) / alpha_per_km


def _haar_unitaries(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Haar-random SU(2) matrices ``[[a, b], [-b*, a*]]``, ``(*shape, 2, 2)``."""
    q = rng.standard_normal((*shape, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    a = q[..., 0] + 1j * q[..., 1]
    b = q[..., 2] + 1j * q[..., 3]
    return np.stack(
        [np.stack([a, b], axis=-1), np.stack([-b.conj(), a.conj()], axis=-1)], axis=-2
    )


# -----------------------------------------------------------------------------
# Stage operations
# -----------------------------------------------------------------------------


class _Ops:
    """Stage operations on NumPy or (unfused) CuPy arrays.

    ``phi`` is the dispersion phase per km on the FFT grid,
    ``-(β₂ω²/2 + β₃ω³/6) · 1 km``, and ``omega`` the angular frequency
    grid, both in the field's real precision on its device.
    """

    fused = False

    def __init__(self, xp: ModuleType, phi: ArrayType, omega: ArrayType) -> None:
        self.xp = xp
        self.phi = phi
        self.omega = omega

    def fft(self, x: ArrayType) -> ArrayType:
        return _fft.fft(x, axis=-1, overwrite_x=True)

    def ifft(self, x: ArrayType) -> ArrayType:
        return _fft.ifft(x, axis=-1, overwrite_x=True)

    def _rotation(self, phase: ArrayType, dtype: Any) -> ArrayType:
        """``exp(j·phase)`` from float32 sin/cos (much faster than complex exp)."""
        xp = self.xp
        r = xp.empty(phase.shape, dtype=dtype)
        r.real = xp.cos(phase)
        r.imag = xp.sin(phase)
        return r

    def linear(
        self,
        X: ArrayType,
        length_km: float,
        gain: float,
        section: tuple[np.ndarray, float] | None,
    ) -> ArrayType:
        """``X ← gain · e^{j length·phi} · X`` with an optional PMD section
        ``(U, τ/2)``: ``X ← U^H diag(e^{-jωτ/2}, e^{jωτ/2}) U X``."""
        h = self._rotation(self.phi * self.phi.dtype.type(length_km), X.dtype)
        h *= X.real.dtype.type(gain)
        if section is None:
            X *= h
            return X
        u, half_tau = section
        u = u.astype(X.dtype)
        d = self._rotation(self.omega * self.omega.dtype.type(half_tau), X.dtype)
        y0 = (u[0, 0] * X[0] + u[0, 1] * X[1]) * h * d.conj()
        y1 = (u[1, 0] * X[0] + u[1, 1] * X[1]) * h * d
        X[0] = u[0, 0].conj() * y0 + u[1, 0].conj() * y1
        X[1] = u[0, 1].conj() * y0 + u[1, 1].conj() * y1
        return X

    def nonlinear(self, x: ArrayType, k: float) -> ArrayType:
        """``x ← x · exp(-j k Σ_c |x_c|²)`` (the power summed over channels)."""
        power = (x.real * x.real + x.imag * x.imag).sum(axis=0)
        x *= self._rotation(power * power.dtype.type(-k), x.dtype)
        return x


_NUMBA: dict[str, Callable[..., None]] = {}


def _numba_pmd_kernel() -> Callable[..., None]:
    """Fused, parallel CPU linear stage with a PMD section (in place).

    NumPy needs about ten passes and temporaries for the 2x2 Jones product;
    this is one (2x2^20 complex64: 52 ms -> 7.5 ms per stage, 4 cores; the
    whole PMD propagation 0.61x).  The PMD-free stages stay NumPy: fused
    Numba versions measured only 0.84-0.94x at 2^20 samples and 1.04-1.30x
    (slower) at 2^16 on 4 cores, as their thread pool competes with
    scipy.fft's workers.
    """
    if "pmd" not in _NUMBA:
        import numba

        # Real arithmetic: a complex literal (1j) would promote complex64
        # work to complex128.
        @numba.njit(cache=True, fastmath=True, nogil=True, parallel=True)
        def linear_pmd(X, phi, omega, length, gain, half_tau, u):  # type: ignore[no-untyped-def]
            for i in numba.prange(X.shape[1]):  # type: ignore[attr-defined]
                a = length * phi[i]
                b = omega[i] * half_tau
                h = complex(gain * np.cos(a), gain * np.sin(a))
                d = complex(np.cos(b), np.sin(b))
                x0 = X[0, i]
                x1 = X[1, i]
                y0 = (u[0, 0] * x0 + u[0, 1] * x1) * h * d.conjugate()
                y1 = (u[1, 0] * x0 + u[1, 1] * x1) * h * d
                X[0, i] = u[0, 0].conjugate() * y0 + u[1, 0].conjugate() * y1
                X[1, i] = u[0, 1].conjugate() * y0 + u[1, 1].conjugate() * y1

        _NUMBA["pmd"] = linear_pmd
    return _NUMBA["pmd"]


class _CpuOps(_Ops):
    """NumPy stages, with the PMD section in one Numba pass."""

    def linear(
        self,
        X: ArrayType,
        length_km: float,
        gain: float,
        section: tuple[np.ndarray, float] | None,
    ) -> ArrayType:
        if section is None:
            return super().linear(X, length_km, gain, None)
        X = np.ascontiguousarray(X)
        real = X.real.dtype.type
        u, half_tau = section
        _numba_pmd_kernel()(
            X,
            self.phi,
            self.omega,
            real(length_km),
            real(gain),
            real(half_tau),
            u.astype(X.dtype),
        )
        return X


_KERNELS: dict[str, Any] = {}


def _cupy_kernels() -> dict[str, Any]:
    """Fused CuPy elementwise kernels of the three stages (complex64)."""
    if not _KERNELS:
        import cupy as cp

        _KERNELS["linear"] = cp.ElementwiseKernel(
            "float32 phi, float32 length, float32 gain",
            "complex64 x",
            """
            float s, c;
            sincosf(length * phi, &s, &c);
            x = x * complex<float>(gain * c, gain * s);
            """,
            "commkit_fiber_linear",
        )
        _KERNELS["linear_pmd"] = cp.ElementwiseKernel(
            "float32 phi, float32 omega, float32 length, float32 gain, "
            "float32 half_tau, complex64 u00, complex64 u01, complex64 u10, "
            "complex64 u11",
            "complex64 x0, complex64 x1",
            """
            float s, c, sd, cd;
            sincosf(length * phi, &s, &c);
            sincosf(omega * half_tau, &sd, &cd);
            const complex<float> h(gain * c, gain * s);
            const complex<float> y0 =
                (u00 * x0 + u01 * x1) * h * complex<float>(cd, -sd);
            const complex<float> y1 =
                (u10 * x0 + u11 * x1) * h * complex<float>(cd, sd);
            x0 = conj(u00) * y0 + conj(u10) * y1;
            x1 = conj(u01) * y0 + conj(u11) * y1;
            """,
            "commkit_fiber_linear_pmd",
        )
        _KERNELS["kerr1"] = cp.ElementwiseKernel(
            "float32 k",
            "complex64 x",
            """
            float s, c;
            sincosf(-k * norm(x), &s, &c);
            x = x * complex<float>(c, s);
            """,
            "commkit_fiber_kerr1",
        )
        _KERNELS["kerr2"] = cp.ElementwiseKernel(
            "float32 k",
            "complex64 x0, complex64 x1",
            """
            float s, c;
            sincosf(-k * (norm(x0) + norm(x1)), &s, &c);
            const complex<float> r(c, s);
            x0 = x0 * r;
            x1 = x1 * r;
            """,
            "commkit_fiber_kerr2",
        )
    return _KERNELS


class _FusedOps(_Ops):
    """CuPy complex64 stages: one fused kernel per stage, in-place cuFFT.

    Every operation works in place on the field's buffer and allocates
    nothing, which is what lets a span be captured into a CUDA graph.
    """

    fused = True

    def __init__(self, xp: ModuleType, phi: ArrayType, omega: ArrayType) -> None:
        super().__init__(xp, phi, omega)
        self.k = _cupy_kernels()
        # Both transforms run unscaled and every linear stage (one between
        # each forward and inverse FFT) carries the 1/N: no separate scaling
        # pass, so nothing in a span allocates.
        self.inv_n = 1.0 / phi.shape[-1]

    def fft(self, x: ArrayType) -> ArrayType:
        return _fft.fft(x, axis=-1, overwrite_x=True)

    def ifft(self, x: ArrayType) -> ArrayType:
        return _fft.ifft(x, axis=-1, overwrite_x=True, norm="forward")

    def linear(
        self,
        X: ArrayType,
        length_km: float,
        gain: float,
        section: tuple[np.ndarray, float] | None,
    ) -> ArrayType:
        f32 = np.float32
        gain = gain * self.inv_n
        if section is None:
            self.k["linear"](self.phi, f32(length_km), f32(gain), X)
            return X
        u, half_tau = section
        u = u.astype(np.complex64)
        self.k["linear_pmd"](
            self.phi,
            self.omega,
            f32(length_km),
            f32(gain),
            f32(half_tau),
            u[0, 0],
            u[0, 1],
            u[1, 0],
            u[1, 1],
            X[0],
            X[1],
        )
        return X

    def nonlinear(self, x: ArrayType, k: float) -> ArrayType:
        if x.shape[0] == 1:
            self.k["kerr1"](np.float32(k), x[0])
        else:
            self.k["kerr2"](np.float32(k), x[0], x[1])
        return x


def _make_ops(x: ArrayType, xp: ModuleType, phi: ArrayType, omega: ArrayType) -> _Ops:
    if xp is np:
        return _CpuOps(xp, phi, omega)
    if x.dtype != xp.complex64:
        return _Ops(xp, phi, omega)
    try:
        return _FusedOps(xp, phi, omega)
    except Exception as exc:  # pragma: no cover - GPU toolchain dependent
        logger.warning(
            "Fiber propagation: fused CUDA kernels unavailable (%s); using the "
            "plain CuPy path.",
            exc,
        )
        return _Ops(xp, phi, omega)


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _SpanPlan:
    """The operator sequence of one span: ``K + 1`` linear stages around
    ``K`` Kerr stages (lengths in km, Kerr coefficients in rad/W)."""

    linear_km: np.ndarray  # (K + 1,)
    kerr: np.ndarray  # (K,) γ_eff · effective length
    pmd_half_tau_s: np.ndarray | None  # (K,) half DGD per section


def _span_plan(link: _Link, num_channels: int) -> _SpanPlan:
    fiber = link.fiber
    alpha = fiber.alpha_per_km
    h = np.full(link.steps_per_span, link.span_length_km / link.steps_per_span)
    linear = np.concatenate([[h[0] / 2], (h[:-1] + h[1:]) / 2, [h[-1] / 2]])
    gamma = fiber.nonlinearity_per_w_km * (8.0 / 9.0 if num_channels == 2 else 1.0)
    kerr = gamma * _nonlinear_lengths(h, alpha)
    half_tau = None
    if fiber.pmd_ps_sqrt_km > 0:
        half_tau = 0.5 * fiber.pmd_ps_sqrt_km * 1e-12 * np.sqrt(3.0 * np.pi * h / 8.0)
    return _SpanPlan(linear, kerr, half_tau)


def _dispersion_phase(
    num_samples: int, sampling_rate: float, link: _Link, xp: ModuleType, dtype: Any
) -> tuple[ArrayType, ArrayType]:
    """``(phi, omega)``: dispersion phase per km and the angular frequency
    grid, computed in float64 and stored as the field's real ``dtype`` on
    ``xp``'s device."""
    fiber = link.fiber
    kw = {
        "dispersion_ps_nm_km": fiber.dispersion_ps_nm_km,
        "fiber_length_km": 1.0,
        "center_wavelength_nm": link.center_wavelength_nm,
    }
    b2 = beta2_length(**kw)
    slope = fiber.dispersion_slope_ps_nm2_km
    b3 = 0.0 if slope is None else beta3_length(dispersion_slope_ps_nm2_km=slope, **kw)
    omega = 2.0 * np.pi * np.fft.fftfreq(num_samples, d=1.0 / sampling_rate)
    phi = -(b2 / 2.0 * omega**2 + b3 / 6.0 * omega**3)
    return xp.asarray(phi.astype(dtype)), xp.asarray(omega.astype(dtype))


def _ase_std(link: _Link, noise_figure_db: float, sampling_rate: float) -> float:
    """Per-component std (sqrt(W)) of one amplifier's ASE in one polarization.

    Gain ``G = e^{αL}`` and noise figure ``F`` give the ASE spectral density
    ``n_sp hν (G - 1) = (F G - 1) hν / 2`` per polarization (exact
    ``n_sp = (F G - 1) / (2 (G - 1))``), over the simulated bandwidth f_s.
    """
    gain = math.exp(link.fiber.alpha_per_km * link.span_length_km)
    nu = _SPEED_OF_LIGHT / (link.center_wavelength_nm * 1e-9)
    f = 10.0 ** (noise_figure_db / 10.0)
    density = max(f * gain - 1.0, 0.0) * _PLANCK * nu / 2.0  # W/Hz
    return math.sqrt(density * sampling_rate / 2.0)


def _outer_band_fraction(X: ArrayType, xp: ModuleType, sampling_rate: float) -> float:
    """Fraction of the spectrum's power beyond 0.4 f_s."""
    f = xp.abs(xp.fft.fftfreq(X.shape[-1], d=1.0 / sampling_rate))
    power = X.real * X.real + X.imag * X.imag
    outer = xp.sum(power[:, f > 0.4 * sampling_rate])
    return float(outer / xp.maximum(xp.sum(power), 1e-300))


def _run_span(
    ops: _Ops,
    x: ArrayType,
    plan: _SpanPlan,
    *,
    direction: int,
    kerr_scale: float,
    alpha: float,
    sections: np.ndarray | None,
    on_last_spectrum: Callable[[ArrayType], None] | None = None,
) -> ArrayType:
    """One span's steps, time domain in and out (no amplifier).

    Forward: ``L_0 N_0 L_1 ... N_{K-1} L_K``.  Backward (``direction=-1``)
    applies the inverse operators in reverse order; PMD sections are
    forward-only.
    """
    K = plan.kerr.size
    order = range(K + 1) if direction > 0 else range(K, -1, -1)
    X = ops.fft(x)
    for i in order:
        length = direction * plan.linear_km[i]
        section = None
        if sections is not None and i < K:
            section = (sections[i], float(plan.pmd_half_tau_s[i]))  # type: ignore[index]
        X = ops.linear(X, length, math.exp(-alpha * length / 2.0), section)
        nxt = i if direction > 0 else i - 1  # Kerr stage after this linear one
        if 0 <= nxt < K:
            x = ops.ifft(X)
            x = ops.nonlinear(x, direction * kerr_scale * plan.kerr[nxt])
            X = ops.fft(x)
    if on_last_spectrum is not None:
        on_last_spectrum(X)
    return ops.ifft(X)


def propagate(
    x: ArrayType,
    xp: ModuleType,
    *,
    link: _Link,
    sampling_rate: float,
    direction: int,
    nonlinear_scale: float = 1.0,
    noise_figure_db: float | None = None,
    rng: Any = None,
) -> ArrayType:
    """Propagate the ``(C, N)`` field ``x`` (sqrt(W), C in {1, 2}) over the
    link; ``direction=-1`` backpropagates (DBP: no ASE, no PMD).

    ``x`` is consumed: it is the working buffer.
    """
    fiber = link.fiber
    C, N = x.shape
    alpha = fiber.alpha_per_km
    forward = direction > 0
    plan = _span_plan(link, C)
    phi, omega = _dispersion_phase(N, sampling_rate, link, xp, x.real.dtype)
    ops = _make_ops(x, xp, phi, omega)
    span_gain = math.exp(alpha * link.span_length_km / 2.0)  # field gain

    pmd = std = None
    if forward and (noise_figure_db is not None or fiber.pmd_ps_sqrt_km > 0):
        gen = as_generator(rng)
        if fiber.pmd_ps_sqrt_km > 0:
            pmd = _haar_unitaries(gen, (link.num_spans, plan.kerr.size))
        if noise_figure_db is not None:
            std = x.real.dtype.type(_ase_std(link, noise_figure_db, sampling_rate))

    def check_aliasing(X: ArrayType) -> None:
        fraction = _outer_band_fraction(X, xp, sampling_rate)
        if fraction > _ALIASING_FRACTION:
            logger.warning(
                "Fiber propagation: %.2g of the power lies beyond 0.4 f_s after "
                "the first span; nonlinear spectral broadening may alias. "
                "Increase the sampling rate.",
                fraction,
            )

    run = _span_runner(ops, plan, direction, nonlinear_scale, alpha)
    spans = range(link.num_spans) if forward else range(link.num_spans - 1, -1, -1)
    for n, span in enumerate(spans):
        if not forward:
            x = _scale(x, 1.0 / span_gain)  # undo the amplifier
        sections = None if pmd is None else pmd[span]
        x = run(x, sections, check_aliasing if forward and n == 0 else None)
        if forward:
            x = _scale(x, span_gain)  # the amplifier restores the span loss
            if std is not None:
                x += complex_normal(gen, x.shape, std=std, dtype=x.real.dtype, xp=xp)
    return x


def _scale(x: ArrayType, factor: float) -> ArrayType:
    """In-place real scaling (keeps the working buffer)."""
    x *= x.real.dtype.type(factor)
    return x


def _pointer(x: Any) -> int:
    """Device address of a CuPy array's buffer."""
    return int(x.data.ptr)


def _span_runner(
    ops: _Ops,
    plan: _SpanPlan,
    direction: int,
    kerr_scale: float,
    alpha: float,
) -> Callable[..., ArrayType]:
    """``run(x, sections, on_last_spectrum) -> x`` for one span.

    With fused CuPy operations every stage works in place, so a span always
    reads and writes the same buffer.  A span without PMD sections or a
    callback then runs eagerly once (priming cuFFT plans and the memory
    pool), is captured into a CUDA graph on its next run, and is replayed
    after that: one launch per span instead of about four per step.  A span
    on another buffer runs eagerly; a failed capture falls back to eager
    spans for the rest of the call.
    """

    def eager(
        x: ArrayType,
        sections: np.ndarray | None = None,
        on_last: Callable[[ArrayType], None] | None = None,
    ) -> ArrayType:
        return _run_span(
            ops,
            x,
            plan,
            direction=direction,
            kerr_scale=kerr_scale,
            alpha=alpha,
            sections=sections,
            on_last_spectrum=on_last,
        )

    if not ops.fused:
        return eager

    state: dict[str, Any] = {"graph": None, "ptr": None, "warm": False, "ok": True}

    def run(
        x: ArrayType,
        sections: np.ndarray | None,
        on_last: Callable[[ArrayType], None] | None,
    ) -> ArrayType:
        ptr = _pointer(x)
        if sections is not None or on_last is not None or not state["ok"]:
            return eager(x, sections, on_last)
        if state["graph"] is not None:
            if ptr != state["ptr"]:
                return eager(x)
            state["graph"].launch()
            return x
        if not state["warm"]:
            out = eager(x)
            state["warm"] = True
            state["ok"] = _pointer(out) == ptr  # capturable only in place
            return out
        import cupy as cp

        stream = cp.cuda.Stream(non_blocking=True)
        try:
            ready = cp.cuda.Event()
            ready.record()  # order the work queued so far before the capture
            stream.wait_event(ready)
            with stream:
                stream.begin_capture()
                out = eager(x)
                graph = stream.end_capture()
            if _pointer(out) != ptr:
                raise RuntimeError("the span did not run in place")
            graph.launch(stream)
            stream.synchronize()
            state.update(graph=graph, ptr=ptr)
            return x
        except Exception as exc:  # pragma: no cover - hw/version dependent
            with contextlib.suppress(Exception):
                stream.end_capture()
            state["ok"] = False
            logger.warning(
                "Fiber propagation: CUDA-graph capture failed (%s); continuing "
                "with eager spans.",
                exc,
            )
            return eager(x)

    return run
