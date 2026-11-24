"""Fiber equalization: chromatic-dispersion compensation (correct_chromatic_dispersion,
dispersion_compensation_taps) and digital backpropagation (correct_fiber_propagation)."""

from typing import Any

import numpy as np
import pytest

from commkit import equalization, filtering
from commkit.core import Signal
from tests.common.conversions import to_numpy
from tests.common.signals import make_band_limited_field


class TestCorrectChromaticDispersion:
    """Tests for correct_chromatic_dispersion (electronic dispersion compensation)."""

    def test_round_trip_siso(self, xp: Any, xpt: Any) -> None:
        """Apply CD then compensate: SISO output should recover input."""
        from commkit.impairments import apply_chromatic_dispersion

        n = 1024
        rng = np.random.default_rng(42)
        samples = xp.asarray(
            (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
        )
        fs = 64e9
        d, l, lam = 17.0, 80.0, 1550.0

        distorted = apply_chromatic_dispersion(
            samples,
            sampling_rate=fs,
            dispersion_ps_nm_km=d,
            fiber_length_km=l,
            center_wavelength_nm=lam,
        )
        recovered = equalization.correct_chromatic_dispersion(
            distorted,
            sampling_rate=fs,
            dispersion_ps_nm=d * l,
            method="whole_record",
            center_wavelength_nm=lam,
        )

        xpt.assert_allclose(recovered, samples, atol=1e-3)

    def test_round_trip_mimo(self, xp: Any, xpt: Any) -> None:
        """Apply CD then compensate: MIMO output should recover input."""
        from commkit.impairments import apply_chromatic_dispersion

        n = 1024
        rng = np.random.default_rng(7)
        samples = xp.asarray(
            (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
                np.complex64
            )
        )
        fs = 64e9
        d, l, lam = 17.0, 40.0, 1550.0

        distorted = apply_chromatic_dispersion(
            samples,
            sampling_rate=fs,
            dispersion_ps_nm_km=d,
            fiber_length_km=l,
            center_wavelength_nm=lam,
        )
        recovered = equalization.correct_chromatic_dispersion(
            distorted,
            sampling_rate=fs,
            dispersion_ps_nm=d * l,
            method="whole_record",
            center_wavelength_nm=lam,
        )

        xpt.assert_allclose(recovered, samples, atol=1e-3)

    def test_zero_dispersion_is_identity(self, xp: Any, xpt: Any) -> None:
        """length=0 should return identical samples."""
        samples = xp.ones(512, dtype=xp.complex64)
        out = equalization.correct_chromatic_dispersion(
            samples,
            dispersion_ps_nm=0.0,
            method="whole_record",
            center_wavelength_nm=1550.0,
            sampling_rate=56e9,
        )
        xpt.assert_allclose(out, samples, atol=1e-6)

    def test_output_shape_and_dtype_preserved(self, xp: Any) -> None:
        """Output shape and dtype match input."""
        samples = xp.ones((2, 256), dtype=xp.complex64)
        out = equalization.correct_chromatic_dispersion(
            samples,
            dispersion_ps_nm=170.0,
            method="whole_record",
            center_wavelength_nm=1550.0,
            sampling_rate=56e9,
        )
        assert out.shape == (2, 256)
        assert out.dtype == xp.complex64

    def test_signal_input_returns_signal(self, xp: Any, xpt: Any) -> None:
        """Signal in -> Signal out with sampling_rate preserved."""
        samples = xp.ones(512, dtype=xp.complex64)
        sig = Signal(samples=samples, sampling_rate=56e9, symbol_rate=28e9)

        out = equalization.correct_chromatic_dispersion(
            sig,
            dispersion_ps_nm=340.0,
            method="whole_record",
            center_wavelength_nm=1550.0,
        )
        assert isinstance(out, Signal)
        assert out.sampling_rate == 56e9
        assert out.symbol_rate == 28e9
        assert out.samples.shape == samples.shape

    def test_signal_missing_fs_raises(self, xp: Any) -> None:
        """Missing sampling_rate on array input raises ValueError."""
        samples = xp.ones(512, dtype=xp.complex64)
        with pytest.raises(ValueError, match="sampling_rate"):
            equalization.correct_chromatic_dispersion(
                samples,
                dispersion_ps_nm=340.0,
                method="whole_record",
                center_wavelength_nm=1550.0,
            )

    def test_capture_window_is_compensated(self, xp: Any) -> None:
        """A capture is a window of a longer dispersed transmission.  The
        default FIR recovers the window beyond half the filter from its ends
        (measured -45 dB on a 2-sps RRC signal), and so does the
        ``method="whole_record"``, exact away from the ends
        (-68 dB beyond 50 samples).  Neither recovers the first samples,
        whose history lies before the capture (-7 and -2 dB), and no
        compensation at all fails everywhere (the baseline)."""
        from commkit import Constellation, generate
        from commkit.filtering import RRC
        from commkit.impairments import apply_chromatic_dispersion

        tx = generate(
            Constellation.qam(16),
            num_symbols=2**15,
            symbol_rate=32e9,
            sps=2,
            pulse=RRC(rolloff=0.1),
            rng=1,
            device="gpu" if xp.__name__ == "cupy" else "cpu",
        )
        rx = apply_chromatic_dispersion(
            tx.samples,
            dispersion_ps_nm_km=17.0,
            fiber_length_km=80.0,
            center_wavelength_nm=1550.0,
            sampling_rate=64e9,
        )
        start, width = 2**14, 2**14
        window, ref = rx[start : start + width], tx.samples[start : start + width]
        kw = dict(
            dispersion_ps_nm=1360.0, center_wavelength_nm=1550.0, sampling_rate=64e9
        )

        def error_db(z: Any, sl: slice) -> float:
            e = z[sl] - ref[sl]
            return float(
                10 * xp.log10(xp.mean(xp.abs(e) ** 2) / xp.mean(xp.abs(ref[sl]) ** 2))
            )

        fir = equalization.correct_chromatic_dispersion(window, bandwidth=35.2e9, **kw)
        whole = equalization.correct_chromatic_dispersion(
            window, method="whole_record", **kw
        )
        assert error_db(fir, slice(64, -64)) < -40
        assert error_db(whole, slice(64, -64)) < -60
        assert error_db(fir, slice(0, 8)) > -10
        assert error_db(whole, slice(0, 8)) > -10
        assert error_db(window, slice(64, -64)) > -3

    @pytest.mark.parametrize("fn", ["correct_chromatic_dispersion", "dbp"])
    def test_unknown_method_raises(self, xp: Any, fn: str) -> None:
        from commkit.impairments import Fiber

        x = xp.ones((2, 64), dtype=xp.complex64)
        with pytest.raises(ValueError, match="method"):
            if fn == "dbp":
                equalization.correct_fiber_propagation(
                    x,
                    fiber=Fiber(),
                    span_length_km=80.0,
                    num_spans=1,
                    launch_power_dbm=0.0,
                    center_wavelength_nm=1550.0,
                    sampling_rate=64e9,
                    method="fir",
                )
            else:
                equalization.correct_chromatic_dispersion(
                    x,
                    dispersion_ps_nm=17.0,
                    center_wavelength_nm=1550.0,
                    sampling_rate=64e9,
                    method="blocks",
                )

    def test_whole_record_takes_no_fir_design(self, xp: Any) -> None:
        with pytest.raises(ValueError, match="whole_record"):
            equalization.correct_chromatic_dispersion(
                xp.ones(64, dtype=xp.complex64),
                dispersion_ps_nm=17.0,
                center_wavelength_nm=1550.0,
                sampling_rate=64e9,
                method="whole_record",
                num_taps=9,
            )


class TestDispersionCompensationTaps:
    """dispersion_compensation_taps: the FIR form of correct_chromatic_dispersion."""

    FS = 64e9
    KW = {"center_wavelength_nm": 1550.0}

    @staticmethod
    def _beta2_length(km: float) -> float:
        # β₂ = -D λ² / (2π c), D in s/m², λ in m, L in m.
        c = 299_792_458.0
        return -(17.0e-6) * (1550e-9) ** 2 / (2 * np.pi * c) * km * 1e3

    def test_whole_band_is_the_truncated_band_limited_response(self) -> None:
        """Over the whole band the taps are the inverse DTFT of
        exp(+j β₂L ω²/2) on |f| < fs/2, here by a dense inverse FFT (the
        aliasing of the dense grid is below 1e-8)."""
        km, n_taps, m = 80.0, 61, 2**20
        taps = equalization.dispersion_compensation_taps(
            dispersion_ps_nm=17.0 * km,
            sampling_rate=self.FS,
            num_taps=n_taps,
            **self.KW,
        )
        omega = 2 * np.pi * np.fft.fftfreq(m, d=1 / self.FS)
        h = np.fft.ifft(np.exp(0.5j * self._beta2_length(km) * omega**2))
        n = np.arange(n_taps) - (n_taps - 1) // 2
        np.testing.assert_allclose(taps, h[n % m], atol=1e-8)

    def test_default_length_follows_the_spread(self) -> None:
        """1.25 times the group-delay spread 2π|β₂L|·B·fs, odd; over the
        whole band the spread is D λ² L / (c T²) (45 samples at
        80 km, 64 GS/s, so 57 taps), and a short link keeps 7."""
        spread = 17.0e-6 * (1550e-9) ** 2 * 80e3 / (299_792_458.0 / self.FS**2)
        assert 44 < spread < 45
        full = equalization.dispersion_compensation_taps(
            dispersion_ps_nm=1360.0, sampling_rate=self.FS, **self.KW
        )
        assert len(full) == 57
        band = equalization.dispersion_compensation_taps(
            dispersion_ps_nm=1360.0, sampling_rate=self.FS, bandwidth=32e9, **self.KW
        )
        assert len(band) == 2 * int(np.ceil((1.25 * spread / 2 - 1) / 2)) + 1
        short = equalization.dispersion_compensation_taps(
            dispersion_ps_nm=17.0, sampling_rate=self.FS, **self.KW
        )
        assert len(short) == 7

    def test_zero_dispersion_is_a_delta(self) -> None:
        taps = equalization.dispersion_compensation_taps(
            dispersion_ps_nm=0.0, sampling_rate=self.FS, num_taps=5, **self.KW
        )
        np.testing.assert_array_equal(taps, [0, 0, 1, 0, 0])

    def test_band_fit_beats_truncation_inside_the_band(self) -> None:
        """At one length, the least-squares fit over a band has a smaller
        in-band response error than the truncated response, which spends its
        taps on the whole band (measured: -61 dB against -15 dB)."""
        km, n_taps, band = 80.0, 31, 35.2e9
        f = np.linspace(-band / 2, band / 2, 2001)
        target = np.exp(0.5j * self._beta2_length(km) * (2 * np.pi * f) ** 2)

        def in_band_error_db(taps: np.ndarray) -> float:
            n = np.arange(len(taps)) - (len(taps) - 1) // 2
            resp = np.exp(-2j * np.pi * np.outer(f, n) / self.FS) @ taps
            return float(10 * np.log10(np.mean(np.abs(resp - target) ** 2)))

        common = {
            "dispersion_ps_nm": 17.0 * km,
            "sampling_rate": self.FS,
            "num_taps": n_taps,
        }
        fit = equalization.dispersion_compensation_taps(
            bandwidth=band, **common, **self.KW
        )
        trunc = equalization.dispersion_compensation_taps(**common, **self.KW)
        assert in_band_error_db(fit) < -50
        assert in_band_error_db(trunc) > -25

    @pytest.mark.parametrize("km", [80.0, 1000.0])
    def test_fir_matches_frequency_domain_compensation(
        self, xp: Any, km: float
    ) -> None:
        """Applied by overlap-save to a 2-sps RRC signal, the default taps
        fitted over the occupied band match correct_chromatic_dispersion away
        from the record ends (measured -45 and -50 dB); a filter half the
        spread long, the baseline, does not (-15 and -5 dB)."""
        from commkit import Constellation, generate
        from commkit.filtering import RRC
        from commkit.impairments import apply_chromatic_dispersion

        tx = generate(
            Constellation.qam(16),
            num_symbols=2**14,
            symbol_rate=32e9,
            sps=2,
            pulse=RRC(rolloff=0.1),
            rng=1,
            device="gpu" if xp.__name__ == "cupy" else "cpu",
        )
        kw = {"dispersion_ps_nm": 17.0 * km, **self.KW}
        rx = apply_chromatic_dispersion(
            tx, dispersion_ps_nm_km=17.0, fiber_length_km=km, **self.KW
        )
        ref = to_numpy(
            equalization.correct_chromatic_dispersion(
                rx, method="whole_record", **kw
            ).samples
        )

        def residual_db(taps: np.ndarray) -> float:
            y = to_numpy(filtering.fir_filter(rx, taps).samples)
            k = 2 * len(taps)
            err = y[k:-k] - ref[k:-k]
            return float(
                10 * np.log10(np.mean(np.abs(err) ** 2) / np.mean(np.abs(ref) ** 2))
            )

        taps = equalization.dispersion_compensation_taps(
            sampling_rate=64e9, bandwidth=35.2e9, **kw
        )
        assert residual_db(taps) < -40
        short = equalization.dispersion_compensation_taps(
            sampling_rate=64e9, bandwidth=35.2e9, num_taps=len(taps) // 2 | 1, **kw
        )
        assert residual_db(short) > -30

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sampling_rate": 0.0}, "sampling_rate"),
            ({"bandwidth": 0.0}, "bandwidth"),
            ({"bandwidth": 70e9}, "bandwidth"),
            ({"num_taps": 0}, "num_taps"),
            ({"num_taps": 2.5}, "num_taps"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs: dict[str, Any], match: str) -> None:
        args = {
            "dispersion_ps_nm": 1360.0,
            "sampling_rate": self.FS,
            **self.KW,
            **kwargs,
        }
        error = TypeError if kwargs.get("num_taps") == 2.5 else ValueError
        with pytest.raises(error, match=match):
            equalization.dispersion_compensation_taps(**args)


class TestCorrectFiberPropagation:
    """Digital backpropagation: the inverse of apply_fiber_propagation."""

    LINK = dict(
        span_length_km=80.0,
        num_spans=4,
        launch_power_dbm=6.0,
        center_wavelength_nm=1550.0,
    )

    def test_inverts_a_noiseless_link(self, xp: Any) -> None:
        """Same steps, ξ = 1: the time-reversible symmetric step makes DBP
        the exact inverse; complex64 keeps a float32 rounding floor."""
        from commkit.impairments import Fiber, apply_fiber_propagation

        for dtype, tol in ((np.complex128, 1e-11), (np.complex64, 1e-4)):
            x = xp.asarray(make_band_limited_field(2, 4096, dtype=dtype))
            kw = dict(fiber=Fiber(), steps_per_span=40, sampling_rate=64e9, **self.LINK)
            y = apply_fiber_propagation(x, **kw)
            z = equalization.correct_fiber_propagation(y, method="whole_record", **kw)
            err = float(xp.sqrt(xp.mean(xp.abs(z - x) ** 2) / xp.mean(xp.abs(x) ** 2)))
            assert err < tol, (dtype, err)
            assert z.dtype == dtype

    def test_without_kerr_it_is_dispersion_compensation(self, xp: Any) -> None:
        """ξ = 0 leaves the dispersion and loss: correct_chromatic_dispersion
        over the whole link (the amplifiers cancel the loss)."""
        from commkit.impairments import Fiber

        x = xp.asarray(make_band_limited_field(2, 4096))
        z = equalization.correct_fiber_propagation(
            x,
            method="whole_record",
            fiber=Fiber(),
            steps_per_span=3,
            nonlinear_scale=0.0,
            sampling_rate=64e9,
            **self.LINK,
        )
        ref = equalization.correct_chromatic_dispersion(
            x,
            dispersion_ps_nm=5440.0,
            method="whole_record",
            center_wavelength_nm=1550.0,
            sampling_rate=64e9,
        )
        err = float(xp.sqrt(xp.mean(xp.abs(z - ref) ** 2) / xp.mean(xp.abs(ref) ** 2)))
        assert err < 1e-5

    def test_beats_linear_compensation_on_a_nonlinear_link(self, xp: Any) -> None:
        """6 dBm over 4 x 80 km with ASE (NF 5 dB); distortion after
        dispersion compensation alone is about -16 dB.  Ten DBP steps per
        span reach the ASE floor (about -30 dB, measured); one step per span
        gains about 2 dB, but only with ξ near 0.5 (ξ = 1 overshoots)."""
        from commkit.impairments import Fiber, apply_fiber_propagation

        x = xp.asarray(make_band_limited_field(2, 1 << 14))
        y = apply_fiber_propagation(
            x,
            fiber=Fiber(),
            steps_per_span=50,
            noise_figure_db=5.0,
            rng=3,
            sampling_rate=64e9,
            **self.LINK,
        )

        def error(z: Any) -> float:
            g = xp.sum(xp.conj(x) * z) / xp.sum(xp.abs(x) ** 2)  # common phase
            return float(xp.mean(xp.abs(z - g * x) ** 2) / xp.mean(xp.abs(x) ** 2))

        cdc = error(
            equalization.correct_chromatic_dispersion(
                y,
                dispersion_ps_nm=5440.0,
                method="whole_record",
                center_wavelength_nm=1550.0,
                sampling_rate=64e9,
            )
        )
        kw = dict(fiber=Fiber(), sampling_rate=64e9, **self.LINK)
        full = error(
            equalization.correct_fiber_propagation(
                y, steps_per_span=10, method="whole_record", **kw
            )
        )
        one = error(
            equalization.correct_fiber_propagation(
                y, steps_per_span=1, nonlinear_scale=0.5, method="whole_record", **kw
            )
        )
        assert full < 0.1 * cdc
        assert one < 0.75 * cdc

    def test_capture_window_is_backpropagated_in_blocks(self, xp: Any) -> None:
        """A capture is a window of a longer transmission.  Propagate a long
        record, keep a window from its middle (with its true past and
        future outside), and backpropagate the window alone: the default
        blocks recover the transmitted field away from the window's ends
        (measured -71 dB), as does the whole window.  The first samples,
        whose history lies before the capture, are lost to either (-3 dB
        for the whole window)."""
        from commkit.impairments import Fiber, apply_fiber_propagation

        n, start, width = 1 << 15, 1 << 13, 1 << 14
        x = xp.asarray(make_band_limited_field(2, n))
        kw = dict(fiber=Fiber(), steps_per_span=10, sampling_rate=64e9, **self.LINK)
        y = apply_fiber_propagation(x, **kw)
        window = y[..., start : start + width]
        # The window's power differs slightly from the record's; the launch
        # power is given, so scale the window as the record was.
        ref = x[..., start : start + width]
        keep = slice(1024, width - 1024)  # beyond the transients (guard ~ 120)

        def error_db(z: Any, sl: slice) -> float:
            e = z[..., sl] - ref[..., sl]
            return float(
                10
                * xp.log10(xp.mean(xp.abs(e) ** 2) / xp.mean(xp.abs(ref[..., sl]) ** 2))
            )

        p_window = float(xp.mean(xp.abs(window) ** 2) / xp.mean(xp.abs(y) ** 2))
        launch = self.LINK["launch_power_dbm"] + 10 * np.log10(p_window)
        link = {**kw, "launch_power_dbm": launch}
        blocks = equalization.correct_fiber_propagation(window, **link)
        whole = equalization.correct_fiber_propagation(
            window, method="whole_record", **link
        )
        assert error_db(blocks, keep) < -40
        assert error_db(whole, keep) < -40
        assert error_db(blocks, slice(0, 64)) > -20
        assert error_db(whole, slice(0, 64)) > -20

    def test_blocks_match_the_whole_record_inside(self, xp: Any) -> None:
        """On one record, the blocks and the whole-record backpropagation
        agree beyond the guards: the guard covers the link's memory up to
        the band-limited tails it truncates (measured -71 dB, the same in
        complex128, so not rounding)."""
        from commkit.impairments import Fiber

        x = xp.asarray(make_band_limited_field(2, 1 << 14))
        kw = dict(fiber=Fiber(), steps_per_span=5, sampling_rate=64e9, **self.LINK)
        blocks = equalization.correct_fiber_propagation(x, **kw)
        whole = equalization.correct_fiber_propagation(x, method="whole_record", **kw)
        sl = slice(1024, -1024)
        e = blocks[..., sl] - whole[..., sl]
        rel = float(xp.mean(xp.abs(e) ** 2) / xp.mean(xp.abs(whole[..., sl]) ** 2))
        assert rel < 1e-6

    @pytest.mark.parametrize("n", [1 << 14, 3 * 3856, 1000])
    def test_batches_do_not_change_the_blocks(
        self, xp: Any, xpt: Any, monkeypatch: pytest.MonkeyPatch, n: int
    ) -> None:
        """Blocks of 4096 samples (guard 120, hop 3856) in batches of two
        (the last one shorter) give the output of a single batch: each block
        is propagated alone, and the input ahead of a batch survives the
        previous batch's write.  Lengths: a ragged end, an exact multiple of
        the hop, one block."""
        from commkit.equalization.fiber import compensation as fiber_module
        from commkit.impairments import Fiber

        x = xp.asarray(make_band_limited_field(2, n))
        kw = dict(fiber=Fiber(), steps_per_span=2, sampling_rate=64e9, **self.LINK)
        monkeypatch.setattr(fiber_module, "_BATCH_SAMPLES_CPU", 1 << 30)
        monkeypatch.setattr(fiber_module, "_BATCH_SAMPLES_GPU", 1 << 30)
        one = equalization.correct_fiber_propagation(x, **kw)
        monkeypatch.setattr(fiber_module, "_BATCH_SAMPLES_CPU", 2 * 4096)
        monkeypatch.setattr(fiber_module, "_BATCH_SAMPLES_GPU", 2 * 4096)
        batched = equalization.correct_fiber_propagation(x, **kw)
        xpt.assert_allclose(batched, one, rtol=0, atol=1e-6 * float(xp.abs(one).max()))

    def test_batches_bound_the_working_memory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With batches far shorter than the record (on the CPU, 2**15
        samples here against 2**16 by default), the peak is the working
        copy of the field and the power measurement beside a batch (2.5x the
        input, measured), not the blocks of the whole record and their
        spectra (5.2x in one batch)."""
        import tracemalloc

        from commkit.equalization.fiber import compensation as fiber_module
        from commkit.impairments import Fiber

        x = make_band_limited_field(2, 1 << 19)
        kw = dict(fiber=Fiber(), steps_per_span=1, sampling_rate=64e9, **self.LINK)
        monkeypatch.setattr(fiber_module, "_BATCH_SAMPLES_CPU", 1 << 15)
        equalization.correct_fiber_propagation(x, **kw)
        tracemalloc.start()
        try:
            equalization.correct_fiber_propagation(x, **kw)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 3 * x.nbytes

    def test_bad_nonlinear_scale_raises(self, xp: Any) -> None:
        from commkit.impairments import Fiber

        with pytest.raises(ValueError, match="nonlinear_scale"):
            equalization.correct_fiber_propagation(
                xp.asarray(make_band_limited_field(2, 4096)),
                fiber=Fiber(),
                nonlinear_scale=float("nan"),
                sampling_rate=64e9,
                **self.LINK,
            )
