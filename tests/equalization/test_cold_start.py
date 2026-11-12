"""Cold start of the decision-directed equalizers ``lms`` and ``block_lms``.

A blind decision-directed equalizer recovers from a gain error only while the
outer levels still fall into their own decision regions: down to a gain of
0.67 for bipolar PAM-4, 0.83 for unipolar PAM-4 and 0.86 for 64-QAM.  The
equalizers divide each input channel by its RMS and start from the unit
centre tap.  After the matched filter the samples at the symbol instants then
reach the slicer within ``β/4`` of unit power, at any timing phase.  The
former normalization by ``rms · sqrt(sps)`` handed it a gain of ``1/sqrt(2)``
at 2 sps, below the margins of unipolar PAM-4 and 64-QAM, where the loop
settles at a shrunk false solution.

Each case is scored against the transmitted bits over the last quarter of a
2^18-symbol matched-filtered RRC record and compared with the same equalizer
trained on the first 20000 symbols, which shares its structure (11 taps) and
so its steady state: decision-directed adaptation at a symbol error rate of a
few 1e-3 converges to the trained solution.  The bound is twice the trained
BER.  Measured (seed 3): blind and trained agree to the last error, 0 for
bipolar PAM-4, 1.9e-4 for 16-QAM, 1.2e-3 to 1.8e-3 for unipolar PAM-4 and
1.9e-4 to 2.0e-4 for 64-QAM (AWGN theory 1.9e-7, 1.4e-4, 1.0e-3, 1.6e-4).
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

CASES = [
    pytest.param(Constellation.pam(4), 18.0, id="pam4"),
    pytest.param(Constellation.qam(16), 18.0, id="qam16"),
    pytest.param(Constellation.pam(4, unipolar=True), 18.0, id="pam4-unipolar"),
    pytest.param(Constellation.qam(64), 24.0, id="qam64"),
]
FALSE_LOCK_CASES = [
    pytest.param(Constellation.pam(4, unipolar=True), 18.0, id="pam4-unipolar"),
    pytest.param(Constellation.qam(64), 24.0, id="qam64"),
]
EQUALIZERS = pytest.mark.parametrize(
    "equalizer", [lms, block_lms], ids=["lms", "block_lms"]
)


@EQUALIZERS
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


@pytest.mark.allow_warnings
@EQUALIZERS
@pytest.mark.parametrize(("constellation", "esn0_db"), FALSE_LOCK_CASES)
def test_former_gain_locks_falsely(xp, equalizer, constellation, esn0_db):
    """Baseline: started at the former normalization's slicer gain, a centre
    tap of ``1/sqrt(2)``, the same blind runs lock to a shrunk solution
    (measured BER 0.25 and 0.20)."""
    rx = make_channel_signal(constellation, 2**16, esn0_db=esn0_db, seed=3, xp=xp)
    taps = np.zeros(11, np.complex64)
    taps[5] = 2**-0.5
    res = equalizer(rx, num_taps=11, step_size=1e-3, initial_taps=taps)
    assert ber(res.signal, num_skip_symbols=2**15) > 0.1


@EQUALIZERS
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


@EQUALIZERS
def test_mismatched_decision_constellation_logs_warning(xp, caplog, equalizer):
    """16-QAM decisions on a 4-QAM record: the equalizer scales the four points
    onto the corners of 16-QAM, whose power is 1.8 against the
    constellation's 1 (+2.5 dB): returned with a WARNING."""
    rx = make_channel_signal(Constellation.qam(4), 2**15, esn0_db=18.0, seed=3, xp=xp)
    with caplog.at_level("WARNING", logger="commkit"):
        equalizer(rx, num_taps=11, step_size=1e-3, constellation=Constellation.qam(16))
    assert "decided symbols carry" in caplog.text


@EQUALIZERS
def test_collapse_onto_one_level_logs_warning(xp, caplog, equalizer):
    """review F4: blind unipolar PAM-4 at 4 dB Es/N0 decides every symbol as
    the lowest level (BER 0.5).  The decisions use one of the four points,
    where a correct lock uses all of them (measured: unused probability mass
    0 down to a BER of 0.1, the threshold is 0.4): returned with a
    WARNING that names the collapse."""
    uni = Constellation.pam(4, unipolar=True)
    rx = make_channel_signal(uni, 2**16, esn0_db=4.0, seed=3, xp=xp)
    with caplog.at_level("WARNING", logger="commkit"):
        res = equalizer(rx, num_taps=11, step_size=1e-3)
    assert ber(res.signal, num_skip_symbols=2**15) > 0.4
    assert "on only 1 of its 4 points" in caplog.text


@EQUALIZERS
def test_correct_lock_at_low_snr_does_not_warn(xp, caplog, equalizer):
    """Baseline: trained 16-QAM at 9 dB (BER near 0.1) uses every point and
    keeps the power: no WARNING."""
    rx = make_channel_signal(Constellation.qam(16), 2**16, esn0_db=9.0, seed=3, xp=xp)
    with caplog.at_level("WARNING", logger="commkit"):
        equalizer(
            rx,
            num_taps=11,
            step_size=1e-3,
            training_symbols=rx.reference.symbols[..., :NUM_TRAIN],
        )
    assert "decided symbols" not in caplog.text
