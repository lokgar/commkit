"""state= continuation: chunked runs equal one uninterrupted run exactly."""

import dataclasses

import numpy as np
import pytest

from commkit import equalization as E
from commkit.backend import to_device
from commkit.equalization import EqualizerState
from commkit.mapping import Constellation
from commkit.recovery import BPS, PLL, CycleSlip

C16 = Constellation.qam(16)


def _signal(num_ch: int, n_sym: int, sps: int, seed: int = 3):
    """16-QAM through a short ISI channel (and cross-talk for MIMO), with a
    slow phase ramp and noise.  Returns ``(C, n_sym*sps)`` and ``(C, n_sym)``."""
    rng = np.random.default_rng(seed)
    s = C16.points[rng.integers(0, 16, (num_ch, n_sym))]
    x = np.repeat(s, sps, axis=-1)
    x = np.stack([np.convolve(r, [0.1, 1, 0.2j], mode="same") for r in x])
    if num_ch == 2:
        x = np.stack([x[0] + 0.2 * x[1], x[1] - 0.15 * x[0]])
    x = x * np.exp(1j * (0.3 + 0.001 * np.arange(x.shape[-1])))
    x = x + 0.02 * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x.astype(np.complex64), s.astype(np.complex64)


def _cold_like(state: EqualizerState) -> EqualizerState:
    """A state that starts cold (identity taps, zero left padding, cold CPR
    and P) but with ``state``'s normalization, so an uninterrupted run
    normalizes exactly like the first chunk did."""
    w = np.zeros_like(state.weights)
    for c in range(state.num_channels):
        w[c, c, state.num_taps // 2] = 1
    return dataclasses.replace(
        state,
        weights=w,
        pending=np.zeros((state.num_channels, state.lead), np.complex64),
        overlap=0,
        carrier=None,
        inverse_correlation=None,
    )


def _host(a):
    return None if a is None else np.asarray(to_device(a, "cpu"))


def _stitch(first, second, overlap):
    first = _host(first)
    return np.concatenate([first[..., : first.shape[-1] - overlap], _host(second)], -1)


TRAINED = {"lms", "rls", "block_lms"}

CASES = [
    ("lms", {}),
    ("lms", {"cpr": PLL(cycle_slip=CycleSlip())}),
    ("lms", {"cpr": BPS(test_phases=16, block_size=8, joint_channels=True)}),
    ("rls", {}),
    ("rls", {"cpr": BPS(test_phases=16, block_size=8, cycle_slip=CycleSlip())}),
    ("cma", {}),
    ("rde", {}),
    ("block_lms", {"block_size": 64}),
    (
        "block_lms",
        {
            "block_size": 64,
            "cpr": BPS(test_phases=16, block_size=8, cycle_slip=CycleSlip()),
        },
    ),
    ("block_cma", {"block_size": 64}),
    ("block_rde", {"block_size": 64}),
]


@pytest.mark.parametrize(
    ("name", "extra"), CASES, ids=[f"{n}-{i}" for i, (n, _) in enumerate(CASES)]
)
@pytest.mark.parametrize("num_ch", [1, 2])
@pytest.mark.parametrize("sps", [1, 2])
def test_chunked_equals_uninterrupted(name, extra, num_ch, sps, xp):
    """Outputs, errors, phase and final weights of two chunks stitched with
    ``state.overlap`` equal one run over the whole record, bit for bit."""
    n1, n2 = 1001 * sps + (sps - 1), 1500 * sps  # chunk 1 ends mid-symbol
    x_np, s_np = _signal(num_ch, (n1 + n2) // sps + 1, sps)
    x_np = x_np[:, : n1 + n2]
    if num_ch == 1:
        x_np, s_np = x_np[0], s_np[0]
    x = xp.asarray(x_np)
    fn = getattr(E, name)
    kw = dict(sps=sps, num_taps=7, constellation=C16, **extra)
    data = (xp.asarray(s_np[..., :300]),) if name in TRAINED else ()

    r1 = fn(x[..., :n1], *data, **kw)
    r2 = fn(x[..., n1:], *((None,) if data else ()), **kw, state=r1.state)
    full = fn(x, *data, **kw, state=_cold_like(r1.state))
    ov = r1.state.overlap

    np.testing.assert_array_equal(_stitch(r1.y_hat, r2.y_hat, ov), _host(full.y_hat))
    np.testing.assert_array_equal(_stitch(r1.error, r2.error, ov), _host(full.error))
    np.testing.assert_array_equal(_host(r2.weights), _host(full.weights))
    if full.phase_trajectory is not None:
        np.testing.assert_array_equal(
            _stitch(r1.phase_trajectory, r2.phase_trajectory, ov),
            _host(full.phase_trajectory),
        )


@pytest.mark.parametrize("name", ["cma", "block_cma"])
def test_chunked_pilots_with_deboost(name, xp):
    """Pilot references and de-boosting continue at the resumed symbol."""
    sps, n_sym1, n_sym2 = 2, 1000, 1500
    x_np, s_np = _signal(1, n_sym1 + n_sym2, sps)
    mask = np.zeros(n_sym1 + n_sym2, dtype=bool)
    mask[::10] = True
    gain = 10 ** (3.0 / 20)
    x_np = x_np[0] * np.repeat(np.where(mask, gain, 1.0), sps).astype(np.float32)
    ref, pm = E.build_pilot_ref(s_np[0][mask], mask, n_sym=mask.size, num_ch=1)
    fn = getattr(E, name)
    kw = dict(sps=sps, num_taps=7, constellation=C16, pilot_gain_db=3.0)
    if name == "block_cma":
        kw["block_size"] = 64
    x = xp.asarray(x_np)
    n1 = n_sym1 * sps

    r1 = fn(x[:n1], **kw, pilot_ref=ref[:, :n_sym1], pilot_mask=pm[:n_sym1])
    k = n_sym1 - r1.state.overlap  # first symbol of the continued call
    r2 = fn(x[n1:], **kw, pilot_ref=ref[:, k:], pilot_mask=pm[k:], state=r1.state)
    full = fn(x, **kw, pilot_ref=ref, pilot_mask=pm, state=_cold_like(r1.state))

    np.testing.assert_array_equal(
        _stitch(r1.y_hat, r2.y_hat, r1.state.overlap), _host(full.y_hat)
    )


def test_state_resumes_at_the_last_complete_window():
    """LMS: the state is taken where the zero-padded tail starts; RLS, whose
    outputs already stop there at sps=1, has nothing to recompute."""
    x, s = _signal(1, 1000, 1)
    lms = E.lms(x[0], s[0, :100], sps=1, num_taps=7, constellation=C16)
    rls = E.rls(x[0], s[0, :100], sps=1, num_taps=7, constellation=C16)
    assert lms.state.overlap == 7 // 2
    assert rls.state.overlap == 0
    assert lms.state.pending.shape == (1, 7 // 2 + 7 // 2)  # lead + overlap windows


class TestValidation:
    def _state(self):
        x, s = _signal(1, 500, 2)
        return x[0], E.lms(x[0], s[0, :50], sps=2, num_taps=7, constellation=C16).state

    def test_other_equalizer_raises(self):
        x, st = self._state()
        with pytest.raises(ValueError, match="equalizer='lms'"):
            E.cma(x, sps=2, num_taps=7, state=st)

    @pytest.mark.parametrize(
        "change", [{"num_taps": 9}, {"sps": 1}, {"cpr": PLL()}], ids=str
    )
    def test_other_configuration_raises(self, change):
        x, st = self._state()
        kw = dict(sps=2, num_taps=7, constellation=C16) | change
        with pytest.raises(ValueError, match="the state was made with"):
            E.lms(x, **kw, state=st)

    def test_initial_taps_with_state_raises(self):
        x, st = self._state()
        with pytest.raises(ValueError, match="initial_taps or state"):
            E.lms(
                x,
                sps=2,
                num_taps=7,
                constellation=C16,
                state=st,
                initial_taps=st.weights,
            )

    def test_center_tap_with_state_raises(self):
        x, st = self._state()
        with pytest.raises(ValueError, match="center_tap"):
            E.lms(x, sps=2, num_taps=7, constellation=C16, state=st, center_tap=2)

    def test_not_a_state_raises(self):
        x, _ = self._state()
        with pytest.raises(TypeError, match="EqualizerState"):
            E.lms(x, sps=2, num_taps=7, constellation=C16, state={"weights": 1})
