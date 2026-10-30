"""Tests for the LTE turbo code (coding/turbo.py)."""

from typing import Any

import numpy as np
import pytest

from commkit import generate, mapping, metrics
from commkit.backend import to_device
from commkit.coding import TurboCode, TurboIterations, decode, encode
from commkit.coding.turbo import _QPP
from commkit.impairments import apply_awgn
from tests.common.coding import random_bits
from tests.common.scoring import binomial_margin
from tests.common.theory import qpsk_ber


def _rsc(bits: np.ndarray) -> tuple[list[int], list[tuple[int, int]]]:
    """Plain LTE constituent encoder: g0 = 1 + D^2 + D^3, g1 = 1 + D + D^3."""
    s = [0, 0, 0]  # a_(t-1), a_(t-2), a_(t-3)
    parity = []
    for u in bits:
        a = int(u) ^ s[1] ^ s[2]
        parity.append(a ^ s[0] ^ s[2])
        s = [a, s[0], s[1]]
    tail = []
    for _ in range(3):
        x = s[1] ^ s[2]  # the input that makes a = 0
        tail.append((x, 0 ^ s[0] ^ s[2]))
        s = [0, s[0], s[1]]
    return parity, tail


class TestTables:
    def test_qpp_parameters(self) -> None:
        """TS 36.212 Table 5.1.3-3: 188 sizes, each a permutation."""
        assert len(_QPP) == 188
        assert _QPP[0] == (40, 3, 10) and _QPP[-1] == (6144, 263, 480)
        for k, f1, f2 in _QPP[::17]:
            i = np.arange(k)
            assert np.unique((f1 * i + f2 * i * i) % k).size == k


class TestEncoding:
    def test_matches_the_constituent_encoders(self, xp: Any) -> None:
        code = TurboCode.lte(40)
        u = to_device(random_bits(40, xp, seed=1), "cpu")
        d = to_device(encode(xp.asarray(u), code=code), "cpu")
        d0, d1, d2 = d[:44], d[44:88], d[88:]
        p1, t1 = _rsc(u)
        p2, t2 = _rsc(u[code.interleaver])
        np.testing.assert_array_equal(d0[:40], u)
        np.testing.assert_array_equal(d1[:40], p1)
        np.testing.assert_array_equal(d2[:40], p2)
        # Clause 5.1.3.2.2: x_K, z_K+1, x'_K, z'_K+1 / z_K, x_K+2, z'_K, x'_K+2 / ...
        (x0, z0), (x1, z1), (x2, z2) = t1
        (y0, w0), (y1, w1), (y2, w2) = t2
        np.testing.assert_array_equal(d0[40:], [x0, z1, y0, w1])
        np.testing.assert_array_equal(d1[40:], [z0, x2, w0, y2])
        np.testing.assert_array_equal(d2[40:], [x1, z2, y1, w2])


class TestDecoding:
    def test_noiseless(self, xp: Any, xpt: Any) -> None:
        code = TurboCode.lte(104)
        u = random_bits((4, 104), xp, seed=2)
        res = decode(encode(u, code=code), code=code, method=TurboIterations(2))
        xpt.assert_array_equal(res.bits, u)

    @pytest.mark.parametrize("num_frames", [
        pytest.param(30, id="smoke"),
        pytest.param(300, id="validation", marks=pytest.mark.slow),
    ])  # fmt: skip
    def test_waterfall(self, backend_device: str, num_frames: int) -> None:
        """K = 1024, rate 1/3, 8 iterations: the LTE code reaches FER 1e-2
        near Eb/N0 = 0.7 dB; all decode at 1.2 dB, most fail at -0.3 dB.  The
        pre-FEC BER is first checked against Gray QPSK theory (4 sigma).

        Smoke run: 0 failures in 30 frames bounds the FER below 0.1 (95 %);
        the slow run's 300 frames bound it below 0.01."""
        code = TurboCode.lte(1024)
        c = mapping.Constellation.qam(4)
        sig = generate(c, code.n // 2 * num_frames, symbol_rate=1e9, code=code,
                       rng=3, device=backend_device)  # fmt: skip
        for ebn0, check in ((1.2, lambda f: f == 0.0), (-0.3, lambda f: f > 0.5)):
            esn0 = ebn0 + 10 * np.log10(2 * code.rate)
            rx = apply_awgn(sig, esn0_db=esn0, rng=4)
            p = qpsk_ber(10 ** (esn0 / 10))
            assert abs(metrics.ber(rx) - p) < binomial_margin(p, code.n * num_frames)
            res = decode(
                mapping.compute_llr(rx, noise_var=10 ** (-esn0 / 10)), code=code
            )
            fer = metrics.fer(res.bits, sig.reference.info_bits, block_length=code.k)
            assert check(fer), (ebn0, fer)

    @pytest.mark.parametrize("num_frames", [
        pytest.param(60, id="smoke"),
        pytest.param(600, id="validation", marks=pytest.mark.slow),
    ])  # fmt: skip
    def test_log_map_is_better(self, num_frames: int) -> None:
        """At Eb/N0 = 0.5 dB, in the waterfall, exact log-MAP decodes more
        frames than max-log (both 4 iterations).  The operating point is
        checked first: max-log's FER inside (0.05, 0.95), else both would sit
        at 0 or 1 and the comparison would be empty.

        Smoke run (60 frames, about 0.6 against 0.4): only the ordering.  The
        slow run (600 frames, about 0.52 against 0.37) requires the gap to
        exceed 4 sigma of the difference of two binomial estimates."""
        code = TurboCode.lte(512)
        sig = generate(mapping.Constellation.qam(4), code.n // 2 * num_frames,
                       symbol_rate=1e9, code=code, rng=5)  # fmt: skip
        esn0 = 0.5 + 10 * np.log10(2 * code.rate)
        llrs = mapping.compute_llr(
            apply_awgn(sig, esn0_db=esn0, rng=6), noise_var=10 ** (-esn0 / 10)
        )
        maxlog, exact = (
            metrics.fer(
                decode(llrs, code=code, method=m).bits,
                sig.reference.info_bits,
                block_length=code.k,
            )  # fmt: skip
            for m in (TurboIterations(4), TurboIterations(4, method="exact"))
        )
        assert 0.05 < maxlog < 0.95, maxlog
        if num_frames < 600:
            assert exact < maxlog, (exact, maxlog)
        else:
            margin = np.hypot(binomial_margin(maxlog, num_frames),
                              binomial_margin(exact, num_frames))  # fmt: skip
            assert maxlog - exact > margin, (exact, maxlog, margin)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="Table 5.1.3-3"):
            TurboCode(41)
        with pytest.raises(TypeError, match="info_bits must be an integer"):
            TurboCode(40.0)
        with pytest.raises(TypeError, match="info_bits must be an integer"):
            TurboCode(True)
        assert TurboCode(np.int64(40)) == TurboCode(40)
        with pytest.raises(ValueError, match="iterations"):
            TurboIterations(0)
        with pytest.raises(ValueError, match="scale"):
            TurboIterations(scale=1.5)
        with pytest.raises(ValueError, match="method"):
            TurboIterations(method="exact-ish")
        assert repr(TurboCode.lte(40)) == "TurboCode.lte(40)"
