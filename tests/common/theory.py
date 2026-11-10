"""Closed-form and numerically integrated expectations for the tests.

Written from textbook definitions with NumPy and SciPy only, never through
commkit, so a test that compares a library result with these is independent
of the implementation.  ``esn0`` is linear Es/N0 on a unit-energy
constellation; ``noise_var`` is the complex noise variance N0.
"""

import numpy as np
from scipy.special import erfc


def q_function(x: float | np.ndarray) -> float | np.ndarray:
    """Gaussian tail ``Q(x) = P(N(0, 1) > x)``."""
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))


def qpsk_ber(esn0: float) -> float:
    """Gray QPSK bit error rate, ``Q(sqrt(Es/N0))``."""
    return float(q_function(np.sqrt(esn0)))


def qam_ser(order: int, esn0: float) -> float:
    """Symbol error rate of square M-QAM with ML decisions:
    ``1 - (1 - 2 (1 - 1/√M) Q(sqrt(3 Es/N0 / (M - 1))))²`` (Proakis)."""
    m = int(round(np.sqrt(order)))
    if m * m != order:
        raise ValueError("square QAM only")
    p_axis = 2.0 * (1.0 - 1.0 / m) * q_function(np.sqrt(3.0 * esn0 / (order - 1)))
    return float(1.0 - (1.0 - p_axis) ** 2)


def awgn_mi(points: np.ndarray, noise_var: float, pmf: np.ndarray | None = None,
            nodes: int = 40) -> float:  # fmt: skip
    """Mutual information (bit/symbol) of a discrete input in complex AWGN.

    ``I = -Σ_i p_i E_z[log2 Σ_j p_j exp(-(|x_i - x_j + z|² - |z|²)/N0)]``
    with ``z ~ CN(0, N0)``, the expectation by 2-D Gauss–Hermite quadrature
    (``z = sqrt(N0) (t_a + j t_b)``, weights ``w_a w_b / π``).
    """
    x = np.asarray(points, dtype=np.complex128)
    p = np.full(x.size, 1.0 / x.size) if pmf is None else np.asarray(pmf, float)
    t, w = np.polynomial.hermite.hermgauss(nodes)
    z = np.sqrt(noise_var) * (t[:, None] + 1j * t[None, :]).ravel()
    wz = (w[:, None] * w[None, :]).ravel() / np.pi
    total = 0.0
    for i in range(x.size):
        d = x[i] - x[None, :] + z[:, None]  # (Z, M)
        expo = -(np.abs(d) ** 2 - np.abs(z[:, None]) ** 2) / noise_var
        top = expo.max(axis=1, keepdims=True)
        lse = top[:, 0] + np.log(np.sum(p[None, :] * np.exp(expo - top), axis=1))
        total -= p[i] * np.sum(wz * lse) / np.log(2.0)
    return float(total)


def mmse_mse(h_fft: np.ndarray, noise_var: float, signal_power: float) -> float:
    """MSE per sample of the frequency-domain MMSE (Wiener) inverse of a
    channel with response ``h_fft``, white input of power ``signal_power``
    and white noise ``noise_var``: ``mean_k σ² P / (P |H_k|² + σ²)``."""
    h2 = np.abs(np.asarray(h_fft)) ** 2
    return float(np.mean(noise_var * signal_power / (signal_power * h2 + noise_var)))


def wiener_smoother_var(process_var: float, measurement_var: float) -> float:
    """Error variance of the two-sided Wiener smoother of a random walk
    (increment variance ``q``) seen in white noise (``r``):
    ``∫ q r / (q + 4 r sin²(ω/2)) dω/2π = q r / sqrt(q² + 4 q r)``."""
    q, r = process_var, measurement_var
    return float(q * r / np.sqrt(q * q + 4.0 * q * r))


def tone_frequency_crb(snr: float, num_samples: int) -> float:
    """Cramér–Rao bound on the variance of a tone's frequency, in
    (cycles/sample)², from ``N`` samples at linear SNR (Rife–Boorstyn):
    ``6 / ((2π)² SNR N (N² - 1))``."""
    n = num_samples
    return float(6.0 / ((2.0 * np.pi) ** 2 * snr * n * (n * n - 1)))


def qam16_gray_ber(esn0: float) -> float:
    """Bit error rate of Gray-labelled 16-QAM: per axis a Gray 4-PAM with
    ``a = sqrt(Es/(5 N0))``, ``(3 Q(a) + 2 Q(3a) - Q(5a)) / 4`` (exact)."""
    a = np.sqrt(esn0 / 5.0)
    return float((3 * q_function(a) + 2 * q_function(3 * a) - q_function(5 * a)) / 4)


def pam_gray_ber(order: int, esn0: float, *, unipolar: bool = False) -> float:
    """Bit error rate of binary-reflected Gray M-PAM with midpoint decisions
    (exact).  The levels ``0, 1, ..., M-1`` (centred unless ``unipolar``) are
    scaled to unit mean energy and see real noise of variance ``N0/2``;
    ``P(j | i) = Q((t_j - a_i)/σ) - Q((t_{j+1} - a_i)/σ)`` with thresholds
    ``t`` halfway between levels, weighted by the Hamming distance of the
    labels ``i ^ (i >> 1)``.  Square Gray M-QAM is ``pam_gray_ber(√M,
    esn0 / 2)``: each axis carries half the energy and half the noise."""
    levels = np.arange(order, dtype=float)
    if not unipolar:
        levels -= levels.mean()
    levels /= np.sqrt(np.mean(levels**2))
    sigma = np.sqrt(1.0 / (2.0 * esn0))
    edges = np.concatenate(([-np.inf], (levels[:-1] + levels[1:]) / 2, [np.inf]))
    gray = np.arange(order) ^ (np.arange(order) >> 1)
    ham = np.array([[bin(a ^ b).count("1") for b in gray] for a in gray])
    p = q_function((edges[None, :-1] - levels[:, None]) / sigma) - q_function(
        (edges[None, 1:] - levels[:, None]) / sigma
    )
    return float(np.sum(p * ham) / order / np.log2(order))


def qam_gray_ber(order: int, esn0: float) -> float:
    """Bit error rate of square Gray M-QAM (exact), see :func:`pam_gray_ber`."""
    m = int(round(np.sqrt(order)))
    if m * m != order:
        raise ValueError("square QAM only")
    return pam_gray_ber(m, esn0 / 2.0)


def bpsk_ber(esn0: float) -> float:
    """Coherent BPSK (2-PAM) bit error rate, ``Q(sqrt(2 Es/N0))``: the
    in-phase noise of variance ``N0 / 2`` on unit-energy symbols."""
    return float(q_function(np.sqrt(2.0 * esn0)))


def bounded_distance_fer(p: float, n: int, t: int) -> float:
    """Block error rate of a code correcting every pattern of at most ``t``
    of ``n`` independent bit errors of probability ``p``, and no other:
    ``1 - Σ_{i <= t} C(n, i) p^i (1 - p)^(n - i)``."""
    from math import comb

    return float(
        1.0 - sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(t + 1))
    )


def raised_cosine_slope_energy(rolloff: float) -> float:
    """``Σ_{n≠0} g'(n)²`` for the raised-cosine pulse ``g`` (``T = 1``).

    Sampling the matched-filter output ``τ`` symbol periods off the symbol
    instants adds the ISI ``Σ_n a_n g(τ - n)`` beyond the symbol itself,
    whose power for small ``τ`` and unit-power i.i.d. symbols is
    ``τ² · Σ_{n≠0} g'(n)²`` (``g'(0) = 0``).  A timing jitter of variance
    ``σ²`` therefore adds the noise ``σ² · Σ_{n≠0} g'(n)²``; ``π²/3`` for
    ``rolloff → 0``.  The derivative by central differences of the
    closed-form pulse, the sum over ``|n| ≤ 2000``.
    """

    def g(t: np.ndarray) -> np.ndarray:
        den = 1 - (2 * rolloff * t) ** 2
        return np.sinc(t) * np.cos(np.pi * rolloff * t) / den

    n = np.arange(1, 2001, dtype=np.float64)
    h = 1e-6
    d = (g(n + h) - g(n - h)) / (2 * h)
    return float(2 * np.sum(d**2))


def ls_slope_std(
    jitter_rms: float, correlation_length: float, num_points: int
) -> float:
    """Standard deviation of a least-squares slope through ``num_points``
    equally spaced samples of a line plus noise of rms ``jitter_rms`` that
    stays correlated over ``correlation_length`` samples.

    With ``K / L`` independent samples, ``var = σ² / Σ(k - k̄)² · L =
    12 σ² L / K³``.
    """
    return float(np.sqrt(12 * jitter_rms**2 * correlation_length / num_points**3))


def psk_ser(order: int, esn0: float) -> float:
    """Exact M-PSK symbol error rate over complex AWGN (Craig's form),
    ``(1/π) ∫_0^{(M-1)π/M} exp(-Es/N0 · sin²(π/M) / sin²θ) dθ``."""
    from scipy.integrate import quad

    a = np.sin(np.pi / order) ** 2
    val, _ = quad(
        lambda t: np.exp(-esn0 * a / np.sin(t) ** 2), 0, (order - 1) * np.pi / order
    )
    return float(val / np.pi)
