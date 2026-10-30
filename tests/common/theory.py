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
