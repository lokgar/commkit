"""Fiber equalization: estimate_chromatic_dispersion (ClockToneScan, FourthMomentScan,
QuadraticPhaseFit)."""

import logging
from typing import Any

import numpy as np
import pytest

from commkit import equalization, filtering


def _dispersed_signal(xp: Any, ps_nm: float, rolloff: float, noise: bool) -> Any:
    """Dual-polarization 16-QAM at 32 GBd, 2 sps, dispersed by ``ps_nm``
    (as D = 17 ps/(nm km) over ps_nm / 17 km, or negative D), with 12 dB
    Es/N0, 10 ps of DGD and a 1 GHz frequency offset when ``noise``."""
    from commkit import Constellation, generate
    from commkit.filtering import RRC
    from commkit.impairments import apply_awgn, apply_chromatic_dispersion, apply_pmd
    from commkit.spectral import shift_frequency

    tx = generate(
        Constellation.qam(16),
        num_symbols=2**16,
        symbol_rate=32e9,
        sps=2,
        pulse=RRC(rolloff=max(rolloff, 1e-3)),
        num_channels=2,
        rng=1,
        device="gpu" if xp.__name__ == "cupy" else "cpu",
    ).replace(domain="OPT", center_frequency=193.4e12)
    y = apply_chromatic_dispersion(
        tx,
        dispersion_ps_nm_km=17.0 if ps_nm >= 0 else -17.0,
        fiber_length_km=abs(ps_nm) / 17.0,
    )
    if noise:
        y = apply_pmd(y, dgd=10e-12, theta=0.6)
        y = apply_awgn(y, esn0_db=12, rng=2)
        y = shift_frequency(y, frequency=1e9)
    return y


class TestEstimateChromaticDispersion:
    """estimate_chromatic_dispersion with ClockToneScan: blind, joint over channels."""

    @pytest.mark.parametrize(
        ("rolloff", "tol"), [(0.1, 100.0), (0.35, 25.0)], ids=["r010", "r035"]
    )
    @pytest.mark.parametrize("ps_nm", [0.0, 1360.0, -8500.0, 34000.0])
    def test_recovers_the_accumulated_dispersion(
        self, xp: Any, rolloff: float, tol: float, ps_nm: float
    ) -> None:
        """Against the dispersion applied, with noise, PMD and a frequency
        offset: within 62 ps/nm at rolloff 0.1 and 7 at 0.35 (measured on
        CPU and GPU realizations); the sign included, which a folded cyclic
        frequency would flip."""
        y = _dispersed_signal(xp, ps_nm, rolloff, noise=True)
        est = equalization.estimate_chromatic_dispersion(
            y, equalization.ClockToneScan()
        )
        assert est.value.shape == (2,)
        assert float(est.value[0]) == float(est.value[1])
        assert abs(float(est.value[0]) - ps_nm) < tol

    @pytest.mark.parametrize("rolloff", [0.0, 0.1], ids=["r000", "r010"])
    @pytest.mark.parametrize("ps_nm", [1360.0, -8500.0, 34000.0])
    def test_fourth_moment_scan(self, xp: Any, rolloff: float, ps_nm: float) -> None:
        """Blind and without an excess band: within 11 ps/nm at rolloff 0
        and 0.1 with noise, PMD and a frequency offset (measured)."""
        y = _dispersed_signal(xp, ps_nm, rolloff, noise=True)
        est = equalization.estimate_chromatic_dispersion(
            y, equalization.FourthMomentScan()
        )
        assert isinstance(est, equalization.DispersionScanEstimate)
        assert abs(float(est.value[0]) - ps_nm) < 25.0
        k = int(np.argmax(est.metric))
        assert est.metric[k] == 1.0
        assert abs(est.candidates[k] - float(est.value[0])) < 30.0

    @pytest.mark.parametrize("rolloff", [0.0, 0.1], ids=["r000", "r010"])
    @pytest.mark.parametrize("ps_nm", [1360.0, -8500.0, 34000.0])
    def test_quadratic_phase_fit(self, xp: Any, rolloff: float, ps_nm: float) -> None:
        """Data-aided, against the Signal's reference symbols, with noise,
        PMD and laser phase noise: within 12 ps/nm (measured).  The measured
        group delay, noisy per bin, follows the line β₂L·ω."""
        from commkit.impairments import apply_awgn, apply_phase_noise

        y = _dispersed_signal(xp, ps_nm, rolloff, noise=False)
        y = apply_phase_noise(y, linewidth=100e3, shared_lo=True, rng=4)
        y = apply_awgn(y, esn0_db=12, rng=2)
        est = equalization.estimate_chromatic_dispersion(
            y, equalization.QuadraticPhaseFit()
        )
        assert abs(float(est.value[0]) - ps_nm) < 25.0
        assert isinstance(est, equalization.QuadraticPhaseFitEstimate)
        b2l = -ps_nm * 1e-3 * (1550.12e-9) ** 2 / (2 * np.pi * 299_792_458.0)
        inband = np.isfinite(est.group_delay) & (np.abs(est.frequencies) < 10e9)
        omega = 2 * np.pi * est.frequencies[inband]
        slope = np.polyfit(omega, est.group_delay[inband], 1)[0]
        assert slope == pytest.approx(b2l, rel=0.05)

    def test_quadratic_phase_fit_explicit_host_symbols(self, xp: Any) -> None:
        """Symbols passed to QuadraticPhaseFit as a host array, with array
        samples on either device, give the estimate of the Signal path."""
        from commkit.backend import to_device

        y = _dispersed_signal(xp, 1360.0, 0.1, noise=False)
        symbols = to_device(y.reference.symbols, "cpu")
        kw = dict(
            sampling_rate=y.sampling_rate,
            symbol_rate=y.symbol_rate,
            center_wavelength_nm=y.center_wavelength_nm,
        )
        est = equalization.estimate_chromatic_dispersion(
            y.samples, equalization.QuadraticPhaseFit(symbols=symbols), **kw
        )
        ref = equalization.estimate_chromatic_dispersion(
            y, equalization.QuadraticPhaseFit()
        )
        assert float(est.value[0]) == pytest.approx(float(ref.value[0]), abs=1e-3)
        assert abs(float(est.value[0]) - 1360.0) < 25.0

    def test_quadratic_phase_fit_needs_symbols(self, xp: Any) -> None:
        x = xp.ones((2, 8192), dtype=xp.complex64)
        kw = dict(sampling_rate=64e9, symbol_rate=32e9, center_wavelength_nm=1550.0)
        with pytest.raises(ValueError, match="known symbols"):
            equalization.estimate_chromatic_dispersion(
                x, equalization.QuadraticPhaseFit(), **kw
            )
        with pytest.raises(ValueError, match="max_dispersion_ps_nm"):
            equalization.QuadraticPhaseFit(max_dispersion_ps_nm=0.0)

    def test_clock_tone_warns_without_an_excess_band(
        self, xp: Any, caplog: Any
    ) -> None:
        """A Nyquist pulse has no clock tone (the scan returns a spurious
        peak, measured 1940 ps/nm off); the Signal's pulse says so."""
        y = _dispersed_signal(xp, 1360.0, 0.0, noise=False)
        with caplog.at_level(logging.WARNING, logger="commkit"):
            equalization.estimate_chromatic_dispersion(y, equalization.ClockToneScan())
        assert "excess band" in caplog.text

    def test_scan_peaks_at_the_estimate(self, xp: Any) -> None:
        y = _dispersed_signal(xp, 8500.0, 0.35, noise=False)
        est = equalization.estimate_chromatic_dispersion(
            y.replace(samples=y.samples[0]), equalization.ClockToneScan()
        )
        assert est.value.shape == ()
        k = int(np.argmax(est.metric))
        assert est.metric[k] == 1.0
        assert abs(est.candidates[k] - float(est.value)) < 10.0
        assert np.all(np.diff(est.candidates) > 0)

    def test_correct_with_the_method_compensates(self, xp: Any) -> None:
        """correct(x, ClockToneScan()) estimates then compensates; at 20 dB
        Es/N0 the SNR after the matched filter is within 1 dB of the
        known-dispersion compensation (measured 0.5 dB: the estimate's
        25 ps/nm residual, which an adaptive equalizer absorbs), and an
        uncompensated record is far below (the baseline)."""
        from commkit import metrics, multirate
        from commkit.impairments import apply_awgn

        y = apply_awgn(
            _dispersed_signal(xp, 1360.0, 0.1, noise=False), esn0_db=20, rng=5
        )

        def snr(z: Any) -> float:
            d = multirate.decimate_to_symbol_rate(filtering.matched_filter(z))
            return float(np.mean(metrics.snr(d, num_skip_symbols=64)))

        known = snr(
            equalization.correct_chromatic_dispersion(
                y, dispersion_ps_nm=1360.0, method="whole_record"
            )
        )
        blind = snr(
            equalization.correct_chromatic_dispersion(
                y, equalization.ClockToneScan(), method="whole_record"
            )
        )
        assert blind > known - 1.0
        assert snr(y) < known - 10

    def test_estimate_feeds_the_taps(self, xp: Any) -> None:
        """The estimate's value designs the filter to study."""
        y = _dispersed_signal(xp, 1360.0, 0.35, noise=False)
        est = equalization.estimate_chromatic_dispersion(
            y, equalization.ClockToneScan()
        )
        taps = equalization.dispersion_compensation_taps(
            dispersion_ps_nm=est.value,
            center_wavelength_nm=1550.0,
            sampling_rate=64e9,
        )
        ref = equalization.dispersion_compensation_taps(
            dispersion_ps_nm=1360.0, center_wavelength_nm=1550.0, sampling_rate=64e9
        )
        assert taps.size == ref.size

    def test_peak_at_the_range_edge_warns(self, xp: Any, caplog: Any) -> None:
        y = _dispersed_signal(xp, 8500.0, 0.35, noise=False)
        with caplog.at_level(logging.WARNING, logger="commkit"):
            equalization.estimate_chromatic_dispersion(
                y, equalization.ClockToneScan(search_range=(-2000.0, 2000.0))
            )
        assert "outside search_range" in caplog.text

    def test_invalid_input_raises(self, xp: Any) -> None:
        x = xp.ones(512, dtype=xp.complex64)
        kw = dict(sampling_rate=64e9, symbol_rate=32e9, center_wavelength_nm=1550.0)
        with pytest.raises(ValueError, match="blocks"):
            equalization.estimate_chromatic_dispersion(
                x, equalization.ClockToneScan(), **kw
            )
        with pytest.raises(TypeError, match="FourthMomentScan"):
            equalization.estimate_chromatic_dispersion(x, "clock", **kw)
        with pytest.raises(ValueError, match="search_range"):
            equalization.ClockToneScan(search_range=(10.0, -10.0))
        cd_kw = dict(center_wavelength_nm=1550.0, sampling_rate=64e9)
        with pytest.raises(ValueError, match="exactly one"):
            equalization.correct_chromatic_dispersion(x, **cd_kw)
        with pytest.raises(ValueError, match="exactly one"):
            equalization.correct_chromatic_dispersion(
                x, equalization.ClockToneScan(), dispersion_ps_nm=1.0, **cd_kw
            )
        with pytest.raises(TypeError, match="how"):
            equalization.correct_chromatic_dispersion(x, 1360.0, **cd_kw)
