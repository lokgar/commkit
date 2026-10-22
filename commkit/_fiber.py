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

Solver.  The loss is taken out of the field: ``A = u e^{-αz/2}``, so ``u``
obeys the lossless equation with a Kerr coefficient ``γ e^{-αz}``, its
linear operator is a pure phase, and an amplifier that restores the span
loss leaves ``u`` unchanged (no rounding of gains that should cancel).
Each span of length ``L`` is cut into ``K`` equal steps.  A step is
``D(h/2) N(h) D(h/2)``: the Kerr phase uses the power at the step's midpoint
weighted over the step, ``γ (e^{-αz_{k-1}} - e^{-αz_k}) / α``, so the scheme
is second order (and exact for a continuous wave).  Equal steps measured about
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

Devices.  With equal steps a span has two step lengths, so the CPU and the
plain CuPy stages precompute their responses (float64 phase, cast to the
field's precision) and the linear stage is one multiply.  The CPU runs the
FFTs on ``scipy.fft`` (``_fft``), the Kerr stage as Numba power/phase and
rotation passes around NumPy's SIMD sin/cos (Numba's are scalar without
SVML) and the PMD section as one Numba pass.  CuPy input runs three fused
elementwise kernels (linear stage, linear stage with a PMD section, Kerr
stage) that compute the phase in registers, on in-place, unscaled cuFFT
transforms, so each stage reads and writes the field once and a span
allocates nothing; the span is captured once into a CUDA graph and
replayed, with the next span's waveplates copied into a fixed device buffer
before each replay.  Any failure falls back to the plain CuPy stages, which
complex128 input also uses.
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
from ._dispersion import (
    _SPEED_OF_LIGHT,
    beta2_length,
    beta3_length,
    dispersion_phase,
)
from ._random import as_generator, complex_normal
from .backend import ArrayType
from .logger import logger
from .math import _unit_phasor, db_to_linear

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
    scale = math.sqrt(1e-3 * db_to_linear(launch_power_dbm) / power)
    field *= field.real.dtype.type(scale)
    return field, was_1d, scale


def _haar_unitaries(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Haar-random SU(2) matrices ``[[a, b], [-b*, a*]]``, ``(*shape, 2, 2)``."""
    q = rng.standard_normal((*shape, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    a = q[..., 0] + 1j * q[..., 1]
    b = q[..., 2] + 1j * q[..., 3]
    return np.stack(
        [np.stack([a, b], axis=-1), np.stack([-b.conj(), a.conj()], axis=-1)], axis=-2
    )


def jones_section(
    X: ArrayType, u: np.ndarray, hm: ArrayType, hp: ArrayType
) -> ArrayType:
    """``X ← U^H diag(hm, hp) U X`` for a ``(2, N)`` spectrum, in place.

    ``u`` (host 2x2) rotates into the principal states of the section, which
    get the per-bin responses ``hm`` and ``hp``: for a differential group
    delay τ, ``e^{∓jωτ/2}`` times any common response.  Shared by
    :func:`commkit.impairments.apply_pmd` (one section) and the fiber's PMD.
    """
    u = u.astype(X.dtype)
    y0 = (u[0, 0] * X[0] + u[0, 1] * X[1]) * hm
    y1 = (u[1, 0] * X[0] + u[1, 1] * X[1]) * hp
    X[0] = u[0, 0].conj() * y0 + u[1, 0].conj() * y1
    X[1] = u[0, 1].conj() * y0 + u[1, 1].conj() * y1
    return X


# -----------------------------------------------------------------------------
# Span plan
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _SpanPlan:
    """One span: ``K`` equal steps of ``step_km``, so ``K + 1`` linear stages
    (half, ``K - 1`` full, half) around ``K`` Kerr stages."""

    step_km: float
    kerr: np.ndarray  # (K,) rad/W: γ_eff · power-weighted step length
    half_tau_s: float  # half DGD of every PMD section (0 without PMD)

    @property
    def num_steps(self) -> int:
        return int(self.kerr.size)

    def linear_km(self, i: int) -> float:
        """Length of linear stage ``i`` (``0 <= i <= K``)."""
        return self.step_km / 2 if i in (0, self.num_steps) else self.step_km


def _span_plan(link: _Link, num_channels: int) -> _SpanPlan:
    fiber = link.fiber
    alpha = fiber.alpha_per_km
    K = link.steps_per_span
    h = link.span_length_km / K
    if alpha == 0.0:
        weight = np.full(K, h)
    else:  # ∫ e^{-αz} dz over each step: e^{-α z_{k-1}} (1 - e^{-αh}) / α
        weight = np.exp(-alpha * h * np.arange(K)) * (-math.expm1(-alpha * h) / alpha)
    gamma = fiber.nonlinearity_per_w_km * (8.0 / 9.0 if num_channels == 2 else 1.0)
    half_tau = 0.0
    if fiber.pmd_ps_sqrt_km > 0:
        half_tau = (
            0.5 * fiber.pmd_ps_sqrt_km * 1e-12 * math.sqrt(3.0 * math.pi * h / 8.0)
        )
    return _SpanPlan(h, gamma * weight, half_tau)


# -----------------------------------------------------------------------------
# Stage operations
# -----------------------------------------------------------------------------


class _Ops:
    """Stage operations on NumPy or (unfused) CuPy arrays.

    ``phase_km`` is the dispersion phase of 1 km on the FFT grid and
    ``omega`` the angular frequency grid, both float64 on the field's
    device.  The response of each distinct stage length (two with equal
    steps, per direction) is computed once in float64 and cast to the
    field's precision.  ``waveplates`` holds every span's PMD sections,
    ``(spans, K, 2, 2)`` on the host, or ``None``.
    """

    fused = False

    def __init__(
        self,
        xp: ModuleType,
        dtype: Any,
        phase_km: ArrayType,
        omega: ArrayType,
        half_tau: float,
        waveplates: np.ndarray | None,
    ) -> None:
        self.xp = xp
        self.dtype = dtype
        self.phase_km = phase_km
        self.omega = omega
        self.half_tau = half_tau
        self.waveplates = waveplates
        self.span = 0
        self._responses: dict[tuple[str, float], Any] = {}

    def set_span(self, span: int) -> None:
        """Select the PMD sections of ``span`` for the next span's stages."""
        self.span = span

    def fft(self, x: ArrayType) -> ArrayType:
        return _fft.fft(x, axis=-1, overwrite_x=True)

    def ifft(self, x: ArrayType) -> ArrayType:
        return _fft.ifft(x, axis=-1, overwrite_x=True)

    def _response(self, length_km: float) -> ArrayType:
        key = ("h", length_km)
        if key not in self._responses:
            self._responses[key] = _unit_phasor(
                self.phase_km * length_km, self.dtype, self.xp
            )
        return self._responses[key]

    def _pmd_responses(self, length_km: float) -> tuple[ArrayType, ArrayType]:
        """``H e^{-jωτ/2}`` and ``H e^{+jωτ/2}`` of a stage with a section."""
        key = ("pmd", length_km)
        if key not in self._responses:
            phase = self.phase_km * length_km
            delay = self.omega * self.half_tau
            self._responses[key] = (
                _unit_phasor(phase - delay, self.dtype, self.xp),
                _unit_phasor(phase + delay, self.dtype, self.xp),
            )
        hm, hp = self._responses[key]
        return hm, hp

    def linear(self, X: ArrayType, length_km: float, section: int | None) -> ArrayType:
        """Linear stage of ``length_km`` (negative backward), with PMD
        section ``section`` of the current span or none."""
        if section is None:
            X *= self._response(length_km)
            return X
        assert self.waveplates is not None
        hm, hp = self._pmd_responses(length_km)
        return jones_section(X, self.waveplates[self.span, section], hm, hp)

    def nonlinear(self, x: ArrayType, k: float) -> ArrayType:
        """``x ← x · exp(-j k Σ_c |x_c|²)`` (the power summed over channels)."""
        power = (x.real * x.real + x.imag * x.imag).sum(axis=0)
        x *= _unit_phasor(power * power.dtype.type(-k), x.dtype, self.xp)
        return x


_NUMBA: dict[str, Callable[..., None]] = {}

# Below this many samples the NumPy Kerr stage is faster than the Numba
# passes, whose thread-pool start costs more than the passes they save
# (4 cores: 0.5 vs 0.7 ms at 2^16, 3.3 vs 3.1 ms at 2^18, 17 vs 12 ms at 2^20).
# The trig-free Jones pass wins at every size (2x at 2^12, 10x at 2^20).
_HYBRID_KERR_MIN = 1 << 18


def _numba_kernels() -> dict[str, Callable[..., None]]:
    """Trig-free CPU passes (in place, complex64 or complex128).

    Numba compiles sin/cos to scalar libm calls (no SVML), while NumPy 2 has
    SIMD sin/cos; so the Kerr stage is a Numba power/phase pass, NumPy
    sin/cos, then a Numba rotation pass, from ``_HYBRID_KERR_MIN`` samples.  Real arithmetic throughout: a
    complex literal (1j) would promote complex64 work to complex128.
    """
    if not _NUMBA:
        import numba

        jit = numba.njit(cache=True, fastmath=True, nogil=True, parallel=True)

        @jit
        def kerr_phase(x, k, out):  # type: ignore[no-untyped-def]
            C, N = x.shape
            for i in numba.prange(N):  # type: ignore[attr-defined]
                p = x[0, i].real * x[0, i].real + x[0, i].imag * x[0, i].imag
                for c in range(1, C):
                    p += x[c, i].real * x[c, i].real + x[c, i].imag * x[c, i].imag
                out[i] = -k * p

        @jit
        def rotate(x, cos, sin):  # type: ignore[no-untyped-def]
            C, N = x.shape
            for i in numba.prange(N):  # type: ignore[attr-defined]
                cr = cos[i]
                ci = sin[i]
                for c in range(C):
                    xr = x[c, i].real
                    xi = x[c, i].imag
                    x[c, i] = complex(xr * cr - xi * ci, xr * ci + xi * cr)

        @jit
        def jones(X, u, hm, hp):  # type: ignore[no-untyped-def]
            for i in numba.prange(X.shape[1]):  # type: ignore[attr-defined]
                x0 = X[0, i]
                x1 = X[1, i]
                y0 = (u[0, 0] * x0 + u[0, 1] * x1) * hm[i]
                y1 = (u[1, 0] * x0 + u[1, 1] * x1) * hp[i]
                X[0, i] = u[0, 0].conjugate() * y0 + u[1, 0].conjugate() * y1
                X[1, i] = u[0, 1].conjugate() * y0 + u[1, 1].conjugate() * y1

        _NUMBA.update(kerr_phase=kerr_phase, rotate=rotate, jones=jones)
    return _NUMBA


class _CpuOps(_Ops):
    """NumPy field: Numba Kerr and PMD passes on preallocated buffers."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.k = _numba_kernels()
        n = self.phase_km.shape[-1]
        real = np.empty(0, self.dtype).real.dtype
        self._phase = np.empty(n, real)
        self._cos = np.empty(n, real)
        self._sin = np.empty(n, real)

    def linear(self, X: ArrayType, length_km: float, section: int | None) -> ArrayType:
        if section is None:
            return super().linear(X, length_km, None)
        assert self.waveplates is not None
        X = np.ascontiguousarray(X)
        hm, hp = self._pmd_responses(length_km)
        u = self.waveplates[self.span, section].astype(X.dtype)
        self.k["jones"](X, u, hm, hp)
        return X

    def nonlinear(self, x: ArrayType, k: float) -> ArrayType:
        if x.shape[-1] < _HYBRID_KERR_MIN:
            return super().nonlinear(x, k)
        x = np.ascontiguousarray(x)
        self.k["kerr_phase"](x, self._phase.dtype.type(k), self._phase)
        np.cos(self._phase, out=self._cos)
        np.sin(self._phase, out=self._sin)
        self.k["rotate"](x, self._cos, self._sin)
        return x


_KERNELS: dict[str, Any] = {}


def _cupy_kernels() -> dict[str, Any]:
    """Fused CuPy elementwise kernels of the three stages (complex64)."""
    if not _KERNELS:
        import cupy as cp

        _KERNELS["linear"] = cp.ElementwiseKernel(
            "float32 phi, float32 length, float32 scale",
            "complex64 x",
            """
            float s, c;
            sincosf(length * phi, &s, &c);
            x = x * complex<float>(scale * c, scale * s);
            """,
            "commkit_fiber_linear",
        )
        _KERNELS["linear_pmd"] = cp.ElementwiseKernel(
            "float32 phi, float32 omega, float32 length, float32 scale, "
            "float32 half_tau, raw complex64 u, int32 k",
            "complex64 x0, complex64 x1",
            """
            float s, c, sd, cd;
            sincosf(length * phi, &s, &c);
            sincosf(omega * half_tau, &sd, &cd);
            const complex<float> h(scale * c, scale * s);
            const complex<float> u00 = u[4 * k], u01 = u[4 * k + 1];
            const complex<float> u10 = u[4 * k + 2], u11 = u[4 * k + 3];
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

    The kernels compute the phase in registers (a GPU stage is limited by
    memory traffic, so reading a precomputed response would cost more than
    the ``sincosf``).  Both transforms run unscaled and every linear stage
    (one between each forward and inverse FFT) carries the 1/N, so there is
    no separate scaling pass.  The current span's waveplates live in a fixed
    device buffer, refreshed by :meth:`set_span` outside the graph.  Nothing
    in a span allocates, which is what lets it be captured into a CUDA graph.
    """

    fused = True

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        xp = self.xp
        self.k = _cupy_kernels()
        self.phi = self.phase_km.astype(xp.float32)
        self.omega32 = self.omega.astype(xp.float32)
        self.inv_n = np.float32(1.0 / self.phase_km.shape[-1])
        self.u_all: Any = None
        self.u_span: Any = None
        if self.waveplates is not None:
            spans, K = self.waveplates.shape[:2]
            self.u_all = xp.asarray(
                self.waveplates.astype(np.complex64).reshape(spans, 4 * K)
            )
            self.u_span = xp.empty(4 * K, dtype=xp.complex64)

    def set_span(self, span: int) -> None:
        super().set_span(span)
        if self.u_span is not None:
            self.u_span[...] = self.u_all[span]

    def ifft(self, x: ArrayType) -> ArrayType:
        return _fft.ifft(x, axis=-1, overwrite_x=True, norm="forward")

    def linear(self, X: ArrayType, length_km: float, section: int | None) -> ArrayType:
        f32 = np.float32
        if section is None:
            self.k["linear"](self.phi, f32(length_km), self.inv_n, X)
            return X
        self.k["linear_pmd"](
            self.phi,
            self.omega32,
            f32(length_km),
            self.inv_n,
            f32(self.half_tau),
            self.u_span,
            np.int32(section),
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


def _make_ops(x: ArrayType, xp: ModuleType, *args: Any) -> _Ops:
    """The stage operations for the field ``x``; ``args`` as :class:`_Ops`
    after ``xp`` and the dtype."""
    if xp is np:
        return _CpuOps(xp, x.dtype, *args)
    if x.dtype != xp.complex64:
        return _Ops(xp, x.dtype, *args)
    try:
        return _FusedOps(xp, x.dtype, *args)
    except Exception as exc:  # pragma: no cover - GPU toolchain dependent
        logger.warning(
            "Fiber propagation: fused CUDA kernels unavailable (%s); using the "
            "plain CuPy path.",
            exc,
        )
        return _Ops(xp, x.dtype, *args)


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------


def _ase_std(link: _Link, noise_figure_db: float, sampling_rate: float) -> float:
    """Per-component std (sqrt(W)) of one amplifier's ASE in one polarization.

    Gain ``G = e^{αL}`` and noise figure ``F`` give the ASE spectral density
    ``n_sp hν (G - 1) = (F G - 1) hν / 2`` per polarization (exact
    ``n_sp = (F G - 1) / (2 (G - 1))``), over the simulated bandwidth f_s.
    """
    gain = math.exp(link.fiber.alpha_per_km * link.span_length_km)
    nu = _SPEED_OF_LIGHT / (link.center_wavelength_nm * 1e-9)
    f = db_to_linear(noise_figure_db)
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
    pmd: bool,
    on_last_spectrum: Callable[[ArrayType], None] | None = None,
) -> ArrayType:
    """One span's steps, time domain in and out (the amplifier is the
    identity on the loss-normalized field).

    Forward: ``L_0 N_0 L_1 ... N_{K-1} L_K``, PMD section ``k`` in ``L_k``.
    Backward (``direction=-1``) applies the inverse operators in reverse
    order, without PMD.
    """
    K = plan.num_steps
    order = range(K + 1) if direction > 0 else range(K, -1, -1)
    X = ops.fft(x)
    for i in order:
        section = i if pmd and i < K else None
        X = ops.linear(X, direction * plan.linear_km(i), section)
        nxt = i if direction > 0 else i - 1  # Kerr stage after this linear one
        if 0 <= nxt < K:
            x = ops.ifft(X)
            x = ops.nonlinear(x, direction * kerr_scale * float(plan.kerr[nxt]))
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
    forward = direction > 0
    plan = _span_plan(link, C)
    kw = {
        "dispersion_ps_nm_km": fiber.dispersion_ps_nm_km,
        "fiber_length_km": 1.0,
        "center_wavelength_nm": link.center_wavelength_nm,
    }
    slope = fiber.dispersion_slope_ps_nm2_km
    phase_km = dispersion_phase(
        N,
        sampling_rate=sampling_rate,
        beta2_length=beta2_length(**kw),
        beta3_length=(
            0.0
            if slope is None
            else beta3_length(dispersion_slope_ps_nm2_km=slope, **kw)
        ),
        xp=xp,
    )
    omega = 2.0 * np.pi * xp.fft.fftfreq(N, d=1.0 / sampling_rate)

    waveplates = std = None
    if forward and (noise_figure_db is not None or fiber.pmd_ps_sqrt_km > 0):
        gen = as_generator(rng)
        if fiber.pmd_ps_sqrt_km > 0:
            waveplates = _haar_unitaries(gen, (link.num_spans, plan.num_steps))
        if noise_figure_db is not None:
            std = x.real.dtype.type(_ase_std(link, noise_figure_db, sampling_rate))
    ops = _make_ops(x, xp, phase_km, omega, plan.half_tau_s, waveplates)

    def check_aliasing(X: ArrayType) -> None:
        fraction = _outer_band_fraction(X, xp, sampling_rate)
        if fraction > _ALIASING_FRACTION:
            logger.warning(
                "Fiber propagation: %.2g of the power lies beyond 0.4 f_s after "
                "the first span; nonlinear spectral broadening may alias. "
                "Increase the sampling rate.",
                fraction,
            )

    run = _span_runner(ops, plan, direction, nonlinear_scale, waveplates is not None)
    spans = range(link.num_spans) if forward else range(link.num_spans - 1, -1, -1)
    for n, span in enumerate(spans):
        ops.set_span(span)
        x = run(x, check_aliasing if forward and n == 0 else None)
        if std is not None:  # the amplifier's ASE (its gain cancels the loss)
            x += complex_normal(gen, x.shape, std=std, dtype=x.real.dtype, xp=xp)
    return x


def _pointer(x: Any) -> int:
    """Device address of a CuPy array's buffer."""
    return int(x.data.ptr)


def _span_runner(
    ops: _Ops,
    plan: _SpanPlan,
    direction: int,
    kerr_scale: float,
    pmd: bool,
) -> Callable[..., ArrayType]:
    """``run(x, on_last_spectrum) -> x`` for one span.

    With fused CuPy operations every stage works in place, so a span always
    reads and writes the same buffer, and the waveplates come from a fixed
    device buffer.  A span without a callback then runs eagerly once
    (priming cuFFT plans and the memory pool), is captured into a CUDA graph
    on its next run, and is replayed after that: one launch per span instead
    of about four per step.  A span on another buffer runs eagerly; a failed
    capture falls back to eager spans for the rest of the call.
    """

    def eager(
        x: ArrayType, on_last: Callable[[ArrayType], None] | None = None
    ) -> ArrayType:
        return _run_span(
            ops,
            x,
            plan,
            direction=direction,
            kerr_scale=kerr_scale,
            pmd=pmd,
            on_last_spectrum=on_last,
        )

    if not ops.fused:
        return eager

    state: dict[str, Any] = {"graph": None, "ptr": None, "warm": False, "ok": True}

    def run(x: ArrayType, on_last: Callable[[ArrayType], None] | None) -> ArrayType:
        ptr = _pointer(x)
        if on_last is not None or not state["ok"]:
            return eager(x, on_last)
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
