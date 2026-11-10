"""Cold start of the decision-directed equalizers ``lms`` and ``block_lms``.

A blind decision-directed equalizer recovers from a gain error only while the
outer levels still fall into their own decision regions.  Starting from a
single centre tap after the all-sample RMS normalization, the on-symbol
samples of an RRC record at 2 sps reach the slicer with a gain of about 0.72:
inside the margin of bipolar PAM-4 (0.67) and 16-QAM, outside that of unipolar
PAM-4 (0.83) and 64-QAM (0.86), where the loop settles at a shrunk false
solution.

Each case is scored against the transmitted bits over the last quarter of a
2^18-symbol record and compared with the same equalizer trained on the first
20000 symbols, which shares its structure (11 taps on an unmatched RRC) and so
its steady state: decision-directed adaptation at a symbol error rate of a few
1e-3 converges to the trained solution.  The bound is twice the trained BER.
Measured (seed 3): the blind runs give 0.25 (unipolar PAM-4) and 0.20
(64-QAM) against 2.2e-3 and 0.9e-3 trained.
"""

import numpy as np
import pytest

from commkit.equalization import block_lms, lms
from commkit.mapping import Constellation
from commkit.metrics import ber
from tests.common.signals import make_channel_signal

NUM_SYMBOLS = 2**18
SKIP = NUM_SYMBOLS - 2**16
NUM_TRAIN = 20000

_FALSE_LOCK = pytest.mark.xfail(
    strict=True,
    reason=(
        "blind cold start locks to a shrunk solution: measured BER 0.25 "
        "(unipolar PAM-4, 18 dB) and 0.20 (64-QAM, 24 dB)"
    ),
)

CASES = [
    pytest.param(Constellation.pam(4), 18.0, id="pam4"),
    pytest.param(Constellation.qam(16), 18.0, id="qam16"),
    pytest.param(
        Constellation.pam(4, unipolar=True), 18.0, id="pam4-unipolar", marks=_FALSE_LOCK
    ),
    pytest.param(Constellation.qam(64), 24.0, id="qam64", marks=_FALSE_LOCK),
]


@pytest.mark.parametrize("equalizer", [lms, block_lms], ids=["lms", "block_lms"])
@pytest.mark.parametrize(("constellation", "esn0_db"), CASES)
def test_blind_start_reaches_trained_ber(xp, equalizer, constellation, esn0_db):
    """Blind decision-directed adaptation reaches the trained equalizer's BER."""
    rx = make_channel_signal(constellation, NUM_SYMBOLS, esn0_db=esn0_db, seed=3, xp=xp)
    blind = equalizer(rx, num_taps=11, step_size=1e-3)
    trained = equalizer(
        rx,
        num_taps=11,
        step_size=1e-3,
        training_symbols=rx.reference.symbols[..., :NUM_TRAIN],
    )
    ber_blind = ber(blind.signal, num_skip_symbols=SKIP)
    ber_trained = ber(trained.signal, num_skip_symbols=SKIP)
    assert ber_trained < 1e-2  # the operating point
    assert ber_blind <= 2.0 * ber_trained + 1e-4, (ber_blind, ber_trained)
    assert np.isfinite(ber_blind)


@pytest.mark.parametrize("equalizer", [lms, block_lms], ids=["lms", "block_lms"])
def test_false_lock_logs_warning(xp, caplog, equalizer):
    """A start at half the gain locks 64-QAM one level low on the outer points
    (BER near 0.2) and the decided symbols lose about 3 dB of power: the
    result is returned with a WARNING."""
    rx = make_channel_signal(Constellation.qam(64), 2**16, esn0_db=24.0, seed=3, xp=xp)
    taps = np.zeros(11, np.complex64)
    taps[5] = 0.5
    with caplog.at_level("WARNING", logger="commkit"):
        res = equalizer(rx, num_taps=11, step_size=1e-3, initial_taps=taps)
    assert ber(res.signal, num_skip_symbols=2**15) > 0.05
    assert "decided symbols carry" in caplog.text


@pytest.mark.parametrize("equalizer", [lms, block_lms], ids=["lms", "block_lms"])
def test_mismatched_decision_constellation_logs_warning(xp, caplog, equalizer):
    """Bipolar decisions on an untrained unipolar PAM-4 record lock with the
    signal's mean on the decision grid (BER about 0.26, decided power
    -1.8 dB): returned with a WARNING."""
    rx = make_channel_signal(
        Constellation.pam(4, unipolar=True), 2**15, esn0_db=18.0, seed=3, xp=xp
    )
    with caplog.at_level("WARNING", logger="commkit"):
        equalizer(rx, num_taps=11, step_size=1e-3, constellation=Constellation.pam(4))
    assert "decided symbols carry" in caplog.text
