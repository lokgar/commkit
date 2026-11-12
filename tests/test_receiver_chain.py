"""Receiver chains end to end, from public functions only.

Each chain sends a known payload through several impairments at once, runs
the receiver that undoes them and scores the result against the transmitted
truth and against AWGN theory.  Each also runs with one correction left out,
which must fail: otherwise the chain would pass without that block working.

- Coded SISO: LDPC-coded 16-QAM frame, RRC at 2 sps, ISI, a fractional
  delay, a frequency offset, laser phase noise and AWGN; frequency-offset
  estimation, frame timing, matched filter, trained LMS with BPS, LLRs and
  decoding.  Also through ``save_npz`` / ``load_npz``.
- Dual-polarization: two independent QPSK streams through a polarization
  rotation and half a symbol of DGD, with a frequency offset, shared-LO phase
  noise and AWGN; frequency-offset estimation, blind block RDE, BPS.
- Fiber: 16-QAM over one 80 km SSMF span, phase noise and AWGN; chromatic
  dispersion compensation, matched filter, trained LMS with BPS.
"""

from typing import Any

import numpy as np
import pytest

import commkit as ck
from commkit import RRC, Constellation, Preamble, SingleCarrierFrame
from commkit.coding import LDPCCode, decode
from commkit.frequency import MthPower
from commkit.impairments import Fiber
from commkit.recovery import BPS, CycleSlip
from tests.common import scoring
from tests.common.signals import make_channel_signal
from tests.common.theory import qam16_gray_ber, qam_ser

RS = 32e9


def _frequency_offset(sig: Any, frequency: float) -> Any:
    """``sig`` times an exact ramp ``exp(j 2π f t)`` (not snapped to a grid)."""
    xp = sig.xp
    n = xp.arange(sig.samples.shape[-1])
    ramp = xp.exp(2j * np.pi * frequency / sig.sampling_rate * n)
    return sig.replace(samples=(sig.samples * ramp).astype(sig.samples.dtype))


class TestCodedSisoChain:
    """LDPC rate 3/4 (short frames) on 16-QAM at Es/N0 = 14 dB: pre-FEC BER
    near 1e-2, well inside the code's threshold, so every codeword decodes
    when each impairment is undone."""

    C16 = Constellation.qam(16)
    CODE = LDPCCode.dvbs2("3/4", frame="short")
    ESN0_DB = 14.0
    TRAIN = 2000

    def _transmit(self, device: str, blocks: int = 8) -> tuple[Any, Any, Any]:
        frame = SingleCarrierFrame(
            payload_blocks=blocks,
            payload_constellation=self.C16,
            payload_code=self.CODE,
            payload_rng=4,
            preamble=Preamble(sequence_type="zc", num_symbols=127),
        )
        tx = frame.to_signal(sps=2, symbol_rate=RS, pulse=RRC(rolloff=0.1),
                             device=device)  # fmt: skip
        xp = tx.xp
        guard = xp.zeros(1000, dtype=tx.samples.dtype)
        x = tx.replace(samples=xp.concatenate([guard, tx.samples, guard]))
        isi = xp.asarray([0.1j, 1.0, 0.25 - 0.15j, -0.08], dtype=x.samples.dtype)
        x = ck.filtering.fir_filter(x, isi)
        x = ck.timing.apply_delay(x, delay=0.4)
        x = _frequency_offset(x, 0.9e9)
        x = ck.impairments.apply_phase_noise(x, linewidth=2e5, rng=2)
        rx = ck.impairments.apply_awgn(x, esn0_db=self.ESN0_DB, rng=3)
        return frame, tx, rx

    def _receive(self, rx: Any, frame: Any, tx: Any, omit: str | None) -> Any:
        y = rx
        if omit != "frequency":
            foe = ck.frequency.estimate_frequency_offset(
                y, MthPower(), constellation=self.C16
            )
            y = ck.frequency.correct_frequency_offset(y, foe)
        timing = ck.timing.estimate_timing(y) if omit != "timing" else 0.0
        y = ck.timing.correct_timing(y, timing, mode="slice")
        y = y.replace(samples=y.samples[..., : tx.samples.shape[-1]])
        y = ck.filtering.matched_filter(y)
        known = rx.xp.asarray(
            np.concatenate([frame.preamble.symbols, frame.body_symbols], axis=-1)
        )
        res = ck.equalization.lms(
            y,
            known[..., : self.TRAIN],
            num_taps=15,
            step_size=0.0 if omit == "equalizer" else 1e-3,
            constellation=self.C16,
            cpr=None if omit == "cpr" else BPS(cycle_slip=CycleSlip(history=100)),
        )
        return ck.core.extract_payload(y.replace(samples=res.y_hat, sampling_rate=RS))

    def _score(self, payload: Any) -> tuple[float, float]:
        skip = 2 * self.TRAIN
        nv = float(ck.metrics.noise_variance(payload, num_skip_symbols=skip))
        dec = decode(ck.mapping.compute_llr(payload, noise_var=nv), code=self.CODE)
        pre = ck.metrics.ber(payload, num_skip_symbols=skip)
        return pre, ck.metrics.fer(dec, payload)

    @pytest.mark.parametrize(
        "blocks",
        [
            pytest.param(8, id="smoke"),
            pytest.param(96, id="slow", marks=pytest.mark.slow),
        ],
    )
    def test_every_codeword_decodes(self, backend_device: str, blocks: int) -> None:
        """FER 0 against the source bits, and a pre-FEC BER between AWGN
        theory at 14 dB (9.4e-3, less a 4-sigma margin) and theory 1.5 dB
        lower (2.2e-2), the stated budget for the ISI's noise enhancement,
        the phase noise and the LMS misadjustment (measured: 1.6e-2).

        0 failures in 8 codewords bound the FER below 0.31 at 95 %; the slow
        run's 96 codewords bound it below 0.031."""
        frame, tx, rx = self._transmit(backend_device, blocks)
        pre, fer = self._score(self._receive(rx, frame, tx, None))
        assert fer == 0.0
        p = qam16_gray_ber(10 ** (self.ESN0_DB / 10))
        n_bits = blocks * self.CODE.n - 4 * 2 * self.TRAIN  # after the skip
        assert p - scoring.binomial_margin(p, n_bits) < pre
        assert pre < qam16_gray_ber(10 ** ((self.ESN0_DB - 1.5) / 10))

    @pytest.mark.allow_warnings
    @pytest.mark.parametrize("omit", ["frequency", "timing", "cpr", "equalizer"])
    def test_each_correction_is_needed(self, backend_device: str, omit: str) -> None:
        """Leaving the frequency, timing or phase correction out breaks the
        chain: the pre-FEC BER rises above 0.1 and at least a quarter of the
        codewords fail (measured: FER 0.88-1.0, pre-FEC BER 0.45-0.50).  This
        channel's ISI is mild, so without the equalizer the code still
        decodes, but the pre-FEC BER leaves the budget of
        :meth:`test_every_codeword_decodes`, AWGN theory 1.5 dB lower (2.2e-2;
        measured 3.1e-2)."""
        frame, tx, rx = self._transmit(backend_device)
        pre, fer = self._score(self._receive(rx, frame, tx, omit))
        if omit == "equalizer":
            assert pre > qam16_gray_ber(10 ** ((self.ESN0_DB - 1.5) / 10)), pre
        else:
            assert pre > 0.1 and fer > 0.25, (pre, fer)

    def test_through_a_capture_file(self, backend_device: str, tmp_path: Any) -> None:
        """The received Signal saved and loaded again (samples, rates, frame
        and reference) decodes the same."""
        frame, tx, rx = self._transmit(backend_device)
        ck.save_npz(rx, tmp_path / "rx.npz")
        loaded = ck.load_npz(tmp_path / "rx.npz", device=backend_device)
        pre, fer = self._score(self._receive(loaded, frame, tx, None))
        assert fer == 0.0
        assert pre == pytest.approx(
            self._score(self._receive(rx, frame, tx, None))[0], rel=1e-3
        )


class TestDualPolChain:
    """Two independent QPSK streams at Es/N0 = 10 dB (40000 symbols, scored
    over the second half after one permutation and one quadrant per stream):
    a blind receiver separates them and comes within 1 dB of AWGN theory."""

    C = Constellation.psk(4)
    ESN0_DB = 10.0
    N = 40000

    def _run(self, xp: Any, omit: str | None) -> np.ndarray:
        x = make_channel_signal(self.C, self.N, theta=np.deg2rad(30.0),
                                dgd=0.5 / RS, dgd_theta=np.pi / 8, num_channels=2,
                                symbol_rate=RS, seed=4, xp=xp)  # fmt: skip
        x = _frequency_offset(x, 0.6e9)
        x = ck.impairments.apply_phase_noise(x, linewidth=5e5, shared_lo=True, rng=2)
        y = ck.impairments.apply_awgn(x, esn0_db=self.ESN0_DB, rng=3)
        if omit != "frequency":
            foe = ck.frequency.estimate_frequency_offset(
                y, MthPower(), constellation=self.C
            )
            y = ck.frequency.correct_frequency_offset(y, foe)
        step = {"step_size": 0.0} if omit == "equalizer" else {}
        z = ck.equalization.block_rde(y, num_taps=15, constellation=self.C, **step)
        z = z.y_hat
        if omit != "cpr":
            z = ck.recovery.correct_carrier_phase(z, BPS(), constellation=self.C)
        return scoring.blind_ser(z, x.reference.symbols, self.C, skip=self.N // 2)

    def test_streams_separate_near_theory(self, xp: Any) -> None:
        """SER per stream between half the AWGN theory (1.6e-3) and theory
        1 dB lower (4.8e-3), the budget for misadjustment, phase noise and
        the frequency estimate's residual (measured: 1.6e-3 to 2.7e-3)."""
        s = self._run(xp, None)
        p = qam_ser(4, 10 ** (self.ESN0_DB / 10))
        assert np.all(s > 0.5 * p), s
        assert np.all(s < qam_ser(4, 10 ** ((self.ESN0_DB - 1) / 10))), s

    @pytest.mark.allow_warnings
    @pytest.mark.parametrize("omit", ["frequency", "cpr", "equalizer"])
    def test_each_correction_is_needed(self, xp: Any, omit: str) -> None:
        """Without any one block both streams fail (SER above 0.3; measured
        0.52-0.75)."""
        assert np.all(self._run(xp, omit) > 0.3)


class TestFiberChain:
    """16-QAM at 32 GBd over one 80 km span of SSMF (17 ps/nm/km) at -10 dBm,
    where the Kerr effect is negligible, then phase noise and AWGN at
    Es/N0 = 16 dB: with dispersion compensated, the SER is that of AWGN
    within 1 dB."""

    C16 = Constellation.qam(16)
    ESN0_DB = 16.0
    N = 30000
    TRAIN = 3000

    @pytest.mark.parametrize(
        ("dispersion_sign", "works"), [(1.0, True), (-1.0, False), (0.0, False)],
        ids=["compensated", "wrong-sign", "uncompensated"],
    )  # fmt: skip
    def test_dispersion_compensation(
        self, backend_device: str, dispersion_sign: float, works: bool, caplog: Any
    ) -> None:
        """Compensated: SER between half the AWGN theory (7.2e-3) and theory
        1 dB lower (1.8e-2; measured 8.9e-3).  The opposite sign doubles the
        dispersion and none leaves about 11 symbols of memory, beyond the 15
        taps at 2 sps: both fail (SER near 0.9)."""
        caplog.set_level("WARNING", logger="commkit")
        fiber = Fiber()
        tx = ck.generate(self.C16, num_symbols=self.N, sps=2, symbol_rate=RS,
                         pulse=RRC(rolloff=0.1), rng=1, device=backend_device)  # fmt: skip
        tx = tx.replace(domain="OPT", center_frequency=193.4e12)
        z = ck.impairments.apply_fiber_propagation(
            tx, fiber=fiber, span_length_km=80.0, num_spans=1,
            launch_power_dbm=-10.0, steps_per_span=20,
        )  # fmt: skip
        z = ck.impairments.apply_phase_noise(z, linewidth=1e5, rng=2)
        y = ck.impairments.apply_awgn(z, esn0_db=self.ESN0_DB, rng=3)
        if dispersion_sign:
            y = ck.filtering.correct_chromatic_dispersion(
                y,
                dispersion_ps_nm=dispersion_sign * fiber.dispersion_ps_nm_km * 80.0,
                method="whole_record",
            )
        y = ck.filtering.matched_filter(y)
        ref = tx.reference.symbols
        res = ck.equalization.lms(y, ref[: self.TRAIN], num_taps=15, step_size=1e-3,
                                  constellation=self.C16, cpr=BPS())  # fmt: skip
        s = scoring.blind_ser(res.y_hat, ref, self.C16, skip=self.N // 2)
        p = qam_ser(16, 10 ** (self.ESN0_DB / 10))
        if works:
            assert 0.5 * p < s < qam_ser(16, 10 ** ((self.ESN0_DB - 1) / 10)), s
        else:
            assert s > 0.5, s
        assert ("decided symbols carry" in caplog.text) is not works


class TestIntensityModulationChain:
    """Unipolar PAM-4 intensity modulation and direct detection at 32 GBd:
    NRZ levels through a 4th-order Bessel modulator response at half the
    symbol rate (its bulk delay and the NRZ half-symbol offset removed at the
    transmitter), the square root as the optical field, -150 dB/Hz laser RIN,
    a photodiode at -14 dBm with 30 pA/sqrt(Hz) thermal noise and shot
    noise, resampling to 2 sps, and a blind decision-directed LMS deciding on
    the unipolar constellation."""

    C = Constellation.pam(4, unipolar=True)
    N = 2**17
    SKIP = 2**17 - 2**15

    def _detected(self, backend_device: str) -> Any:
        from commkit import Rect

        tx = ck.generate(self.C, num_symbols=self.N, symbol_rate=RS, sps=4,
                         pulse=Rect(), rng=1, device=backend_device)  # fmt: skip
        sos = ck.filtering.bessel_sos(
            sampling_rate=tx.sampling_rate, cutoff=0.5 * RS, order=4, norm="mag"
        )
        b, a = sos[:, :3], sos[:, 3:]
        k = np.arange(3)
        group_delay = float(np.sum(b @ k / b.sum(axis=1) - a @ k / a.sum(axis=1)))
        intensity = ck.timing.apply_delay(
            ck.filtering.iir_filter(tx, sos), delay=-(group_delay + 1.5)
        )
        xp = intensity.xp
        field = intensity.replace(
            samples=xp.sqrt(xp.maximum(intensity.samples, 0.0)).astype(xp.complex64),
            domain="OPT",
            center_frequency=193.4e12,
        )
        field = ck.impairments.apply_intensity_noise(field, rin_db_hz=-150.0, rng=2)
        detected = ck.impairments.apply_photodetection(
            field, received_power_dbm=-14.0, thermal_noise_density=30e-12, rng=3
        )
        return ck.multirate.resample(detected, up=1, down=2)

    def test_blind_equalizer_reaches_theory(self, backend_device: str) -> None:
        """The blind LMS output over the last quarter, scored against the
        transmitted bits: BER within 1.5 times the unipolar PAM-4 theory at
        the equalized output's SNR (measured against the truth; real noise,
        Es/N0 = SNR/2), plus 5 binomial standard deviations (measured 2.6e-3
        against 2.2e-3), and within 2 times the trained equalizer's.
        Baseline: the symbol-rate slicer without the equalizer leaves the
        modulator's ISI (BER 0.062), above 10 times theory."""
        from tests.common.theory import pam_gray_ber

        rx = self._detected(backend_device)
        blind = ck.equalization.lms(rx, num_taps=11, step_size=1e-3)
        trained = ck.equalization.lms(
            rx, rx.reference.symbols[:20000], num_taps=11, step_size=1e-3
        )
        y = scoring.to_numpy(trained.y_hat)[self.SKIP :].real
        ref = scoring.to_numpy(rx.reference.symbols)[self.SKIP :]
        snr = float(np.mean(ref**2) / np.mean((y - ref) ** 2))
        p = pam_gray_ber(4, snr / 2, unipolar=True)
        n_bits = 2 * (self.N - self.SKIP)
        ber_blind = ck.metrics.ber(blind.signal, num_skip_symbols=self.SKIP)
        ber_trained = ck.metrics.ber(trained.signal, num_skip_symbols=self.SKIP)
        assert ber_blind < 1.5 * p + 5 * np.sqrt(p / n_bits), (ber_blind, p)
        assert ber_blind <= 2.0 * ber_trained + 1e-4
        plain = ck.multirate.decimate_to_symbol_rate(rx)
        assert ck.metrics.ber(plain) > 10 * p
