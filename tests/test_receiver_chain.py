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
    CODE = LDPCCode.dvbs2("3/4", "short")
    ESN0_DB = 14.0
    TRAIN = 2000

    def _transmit(self, device: str) -> tuple[Any, Any, Any]:
        frame = SingleCarrierFrame(
            payload_blocks=8,
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
        x = ck.timing.fft_fractional_delay(x, delay=0.4)
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
        nv = float(ck.metrics.estimate_noise_variance(payload, num_skip_symbols=skip))
        dec = decode(ck.mapping.compute_llr(payload, noise_var=nv), code=self.CODE)
        pre = ck.metrics.ber(payload, num_skip_symbols=skip)
        return pre, ck.metrics.fer(dec, payload)

    def test_every_codeword_decodes(self, backend_device: str) -> None:
        """FER 0 against the source bits, and a pre-FEC BER between AWGN
        theory at 14 dB (9.4e-3, less a 4-sigma margin) and theory 1.5 dB
        lower (2.2e-2), the stated budget for the ISI's noise enhancement,
        the phase noise and the LMS misadjustment (measured: 1.6e-2)."""
        frame, tx, rx = self._transmit(backend_device)
        pre, fer = self._score(self._receive(rx, frame, tx, None))
        assert fer == 0.0
        p = qam16_gray_ber(10 ** (self.ESN0_DB / 10))
        assert p - scoring.binomial_margin(p, 4 * 30000) < pre
        assert pre < qam16_gray_ber(10 ** ((self.ESN0_DB - 1.5) / 10))

    @pytest.mark.parametrize("omit", ["frequency", "timing", "cpr", "equalizer"])
    def test_each_correction_is_needed(self, backend_device: str, omit: str) -> None:
        """Leaving any one block out breaks the chain: the pre-FEC BER rises
        above 0.1 and at least a quarter of the codewords fail (measured:
        FER 0.6-1.0, pre-FEC BER 0.19-0.50)."""
        frame, tx, rx = self._transmit(backend_device)
        pre, fer = self._score(self._receive(rx, frame, tx, omit))
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
        self, backend_device: str, dispersion_sign: float, works: bool
    ) -> None:
        """Compensated: SER between half the AWGN theory (7.2e-3) and theory
        1 dB lower (1.8e-2; measured 8.9e-3).  The opposite sign doubles the
        dispersion and none leaves about 11 symbols of memory, beyond the 15
        taps at 2 sps: both fail (SER near 0.9)."""
        fiber = Fiber()
        tx = ck.generate(self.C16, self.N, sps=2, symbol_rate=RS,
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
                dispersion_ps_nm_km=dispersion_sign * fiber.dispersion_ps_nm_km,
                fiber_length_km=80.0,
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
