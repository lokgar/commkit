"""Tests for the metrics' metrics of Signals and their input resolution (metrics/_common.py)."""

import dataclasses
from typing import Any

import numpy as np
import pytest

from commkit import generate, metrics, multirate
from commkit.impairments import apply_awgn
from commkit.mapping import Constellation, compute_llr
from tests.common.conversions import device_of


class TestSignalMetricsIntegration:
    """Metrics of a 1-SPS Signal against its reference."""

    @staticmethod
    def _sig(c: Constellation, n: int, xp: Any, **kw: Any):
        sig = generate(c, num_symbols=n, symbol_rate=1e6, **kw)
        return multirate.decimate_to_symbol_rate(
            sig.to("gpu" if xp is not np else "cpu")
        )

    def test_signal_evm(self, xp: Any) -> None:
        """evm(sig) uses reference.symbols."""
        assert metrics.evm(self._sig(Constellation.qam(4), 100, xp)) < 1e-4

    def test_signal_ber(self, xp: Any) -> None:
        """ber(sig) hard-decides the samples against reference.bits."""
        assert metrics.ber(self._sig(Constellation.qam(4), 100, xp)) == 0.0

    def test_signal_evm_blind(self, xp: Any) -> None:
        """evm(sig, blind=True) near zero for a clean signal."""
        sig = self._sig(Constellation.qam(16), 2000, xp)
        assert metrics.evm(sig, blind=True) < 3.0

    def test_signal_ser(self, xp: Any) -> None:
        """ser(sig) is 0 for a clean signal."""
        assert metrics.ser(self._sig(Constellation.qam(16), 200, xp)) == 0.0

    def test_signal_gmi_and_mi(self, xp: Any) -> None:
        """gmi(sig)/mi(sig) compute from the samples, divided by the
        data-aided gain against the reference, with the Signal's
        constellation; equal to the array path on those symbols."""
        c = Constellation.qam(16)
        sig = self._sig(c, 500, xp)
        rx = sig.replace(samples=apply_awgn(sig.samples, esn0_db=10.0, sps=1, rng=1))
        s = rx.reference.symbols
        g = abs(complex(xp.mean(rx.samples * xp.conj(s)))) / float(
            xp.mean(xp.abs(s) ** 2)
        )
        y = rx.samples / xp.float32(g)
        nv = 0.1
        llrs = compute_llr(y, noise_var=nv, constellation=c)
        assert metrics.gmi(rx, noise_var=nv) == pytest.approx(
            metrics.gmi(llrs, rx.reference.bits, constellation=c), rel=1e-6
        )
        assert metrics.mi(rx, noise_var=nv) == pytest.approx(
            metrics.mi(y, noise_var=nv, constellation=c), rel=1e-6
        )

    def test_signal_multichannel_returns_host_array(self, xp: Any) -> None:
        """(C, N) Signals give host (C,) arrays."""
        sig = self._sig(Constellation.qam(4), 100, xp, num_channels=2)
        for out in (metrics.evm(sig), metrics.ser(sig), metrics.ber(sig)):
            assert isinstance(out, np.ndarray)
            assert out.shape == (2,)

    def test_explicit_reference_wins(self, xp: Any) -> None:
        """An explicit reference replaces the Signal's."""
        sig = self._sig(Constellation.qam(4), 100, xp)
        assert metrics.ser(sig, -sig.reference.symbols) == 1.0

    def test_oversampled_signal_raises(self, xp: Any) -> None:
        """Metrics need one sample per symbol; they never decimate."""
        sig = generate(Constellation.qam(4), num_symbols=100, symbol_rate=1e6, sps=2)
        with pytest.raises(ValueError, match="one sample per symbol"):
            metrics.evm(sig)

    def test_frame_signal_raises(self, xp: Any) -> None:
        """A frame Signal mixes segments: extract the payload first."""
        from commkit.core import SingleCarrierFrame

        sig = SingleCarrierFrame(payload_num_symbols=100).to_signal(
            sps=1, symbol_rate=1e6
        )
        with pytest.raises(ValueError, match="extract_payload"):
            metrics.ser(sig)

    def test_length_mismatch_raises(self, xp: Any) -> None:
        """The reference is never clamped to the received length."""
        sig = self._sig(Constellation.qam(4), 100, xp)
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.snr(sig, sig.reference.symbols[:-1])

    def test_missing_reference_raises(self, xp: Any) -> None:
        """A Signal without a reference needs reference= (or blind=True)."""
        sig = self._sig(Constellation.qam(4), 100, xp)
        sig = sig.replace(reference=None)
        with pytest.raises(ValueError, match="reference"):
            metrics.evm(sig)


class TestLowSNRScale:
    """Symbols with a reference are scaled by the data-aided gain, not to unit
    total power, which shrinks them by 1/sqrt(1 + 1/SNR).

    At -10 dB the gain estimate from 1e5 symbols is good to ~0.5% (1 sigma),
    about 0.04 dB of SNR and 1% of MI; the tolerances are ~3 sigma.  The old
    total-power scaling read -1.45 dB here.
    """

    SNR_DB = -10.0

    def _rx(self, xp: Any, c: Constellation, n: int = 100_000):
        """Unit-power symbols at SNR_DB, then normalized to unit total power
        (what decimate_to_symbol_rate does), and the true-scale samples."""
        rng = np.random.default_rng(5)
        s = c.points[rng.choice(c.order, n, p=c.pmf)]
        nv = 10 ** (-self.SNR_DB / 10)
        r = s + np.sqrt(nv / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        normalized = r / np.sqrt(np.mean(np.abs(r) ** 2))
        sig = generate(c, num_symbols=n, symbol_rate=1e6).replace(
            samples=xp.asarray(normalized.astype(np.complex64)),
        )
        bits = c.bit_labels[np.argmin(np.abs(s[:, None] - c.points), axis=1)]
        from commkit.core import Reference

        sig = sig.replace(
            reference=Reference(
                symbols=xp.asarray(s.astype(np.complex64)),
                bits=xp.asarray(bits.reshape(-1).astype(np.int8)),
            )
        )
        return sig, xp.asarray(r.astype(np.complex64)), nv

    def test_snr(self, xp: Any) -> None:
        sig, _, _ = self._rx(xp, Constellation.qam(16))
        assert metrics.snr(sig) == pytest.approx(self.SNR_DB, abs=0.15)

    def test_evm(self, xp: Any) -> None:
        sig, _, nv = self._rx(xp, Constellation.qam(16))
        assert metrics.evm(sig) == pytest.approx(100 * np.sqrt(nv), rel=0.01)

    @pytest.mark.parametrize(
        "c",
        [Constellation.qam(16), Constellation.qam(256).shaped(entropy=7.0)],
        ids=["16qam", "ps-256qam"],
    )
    def test_mi_and_gmi_match_the_true_scale(self, xp: Any, c: Constellation) -> None:
        """MI/GMI of the normalized Signal equal those of the samples on the
        transmit scale, with the transmit noise variance."""
        sig, r, nv = self._rx(xp, c)
        true_mi = metrics.mi(r, noise_var=nv, constellation=c)
        true_gmi = metrics.gmi(
            compute_llr(r, noise_var=nv, constellation=c),
            sig.reference.bits,
            constellation=c,
        )
        assert metrics.mi(sig, noise_var=nv) == pytest.approx(true_mi, rel=0.04)
        assert metrics.gmi(sig, noise_var=nv) == pytest.approx(true_gmi, rel=0.04)
        assert true_mi <= np.log2(1 + 10 ** (self.SNR_DB / 10)) + 0.01

    def test_rotation_is_not_corrected(self, xp: Any) -> None:
        """Only the gain's magnitude is used: a pi rotation stays an error,
        |-s - n - s|^2 = 4 + nv, instead of being undone (SNR_DB)."""
        sig, _, nv = self._rx(xp, Constellation.qam(16))
        flipped = sig.replace(samples=-sig.samples)
        assert metrics.snr(flipped) == pytest.approx(-10 * np.log10(4 + nv), abs=0.15)
        assert metrics.evm(flipped) == pytest.approx(100 * np.sqrt(4 + nv), rel=0.02)


class TestReferenceOnConstellation:
    """review F1: ser and evm raise when the reference symbols are not points
    of the decision constellation (a remapped record with its old reference),
    where they used to report SER 0.75 and EVM 138 %."""

    def _remapped(self, xp):
        """Unipolar PAM-4 shifted to bipolar levels; the Signal takes the
        bipolar constellation but keeps the unipolar reference."""
        uni = Constellation.pam(4, unipolar=True)
        sig = generate(uni, num_symbols=4000, symbol_rate=1e9, sps=1, rng=3)
        sig = sig.to(device_of(xp))
        bip = Constellation.pam(4)
        mapped = xp.asarray(bip.points)[uni._nearest(sig.samples)]
        return sig.replace(samples=mapped, constellation=bip)

    def test_ser_raises(self, xp):
        with pytest.raises(ValueError, match="not points of the decision"):
            metrics.ser(self._remapped(xp))

    def test_evm_raises(self, xp):
        with pytest.raises(ValueError, match="not points of the decision"):
            metrics.evm(self._remapped(xp))

    def test_matching_reference_passes(self, xp):
        """With the reference remapped too, both are exact."""
        sig = self._remapped(xp)
        ref = dataclasses.replace(sig.reference, symbols=sig.samples.copy())
        sig = sig.replace(reference=ref)
        assert metrics.ser(sig) == 0.0
        assert metrics.evm(sig) < 1e-4

    def test_rescaled_reference_passes(self, xp):
        """A reference normalized to its measured power is a few 0.1 % off
        the points: well inside the d_min/4 tolerance."""
        c = Constellation.qam(16)
        rng = np.random.default_rng(0)
        ref = c.points[rng.integers(0, 16, 500)].astype(np.complex64)
        ref = xp.asarray(ref / np.sqrt(np.mean(np.abs(ref) ** 2)))
        assert metrics.ser(ref, ref, constellation=c) == 0.0
