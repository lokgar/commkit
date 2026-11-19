"""Tests for SignalInput, the array/Signal argument of Signal-aware functions."""

from typing import Any

import numpy as np
import pytest

from commkit import Signal
from commkit.core import Reference
from commkit.core._signal_input import (
    SignalInput,
    reference_of,
    require_integer_sps,
    require_value,
)
from commkit.filtering import RRC
from commkit.mapping import Constellation
from tests.common.signals import make_adapter_test_signal


class TestSamples:
    """Unwrapping and wrapping the samples."""

    def test_array_input_is_held_directly(self, xp: Any) -> None:
        samples = xp.ones(8)
        inp = SignalInput(samples, "example()")
        assert inp.samples is samples
        assert inp.signal is None
        assert inp.function_name == "example()"

    def test_signal_input_unwraps_samples(self, xp: Any) -> None:
        sig = make_adapter_test_signal(xp)
        inp = SignalInput(sig, "example()")
        assert inp.samples is sig.samples
        assert inp.signal is sig

    def test_wrap_array_passes_through(self, xp: Any) -> None:
        out = xp.zeros(4)
        inp = SignalInput(xp.ones(4), "f()")
        assert inp.wrap(out, sampling_rate=5.0, num_symbols=1) is out

    def test_wrap_signal_replaces_samples_and_metadata(self, xp: Any) -> None:
        sig = make_adapter_test_signal(xp)
        replacement = xp.zeros(8, dtype=xp.complex64)
        out = SignalInput(sig, "example()").wrap(replacement, sampling_rate=1e6)
        assert out is not sig
        assert out.samples is replacement
        assert out.sampling_rate == 1e6
        assert sig.sampling_rate == 2e6

    def test_wrap_none_raises_for_signal(self, xp: Any) -> None:
        inp = SignalInput(make_adapter_test_signal(xp), "f()")
        with pytest.raises(ValueError, match=r"f\(\): input Signal field is empty"):
            inp.wrap(None)

    def test_wrap_num_symbols_slices_the_reference(self, xp: Any) -> None:
        """num_symbols keeps the reference aligned with the kept periods."""
        symbols = xp.arange(8, dtype=xp.complex64)
        sig = make_adapter_test_signal(xp, reference=Reference(symbols=symbols))
        out = SignalInput(sig, "f()").wrap(sig.samples[:10], num_symbols=5)
        assert out.reference is not None
        xp.testing.assert_array_equal(out.reference.symbols, symbols[:5])
        kept = SignalInput(sig, "f()").wrap(sig.samples)
        assert kept.reference is sig.reference

    def test_wrap_num_symbols_without_reference(self, xp: Any) -> None:
        sig = make_adapter_test_signal(xp)
        assert SignalInput(sig, "f()").wrap(sig.samples, num_symbols=3).reference is (
            None
        )

    def test_symbols_requires_one_sample_per_symbol(self, xp: Any) -> None:
        """symbols() passes arrays and 1-SPS Signals, rejects the rest."""
        sig = make_adapter_test_signal(xp)  # 2 samples per symbol
        with pytest.raises(ValueError, match=r"example\(\) needs one sample"):
            SignalInput(sig, "example()").symbols()
        one = sig.replace(sampling_rate=sig.symbol_rate)
        assert SignalInput(one, "f()").symbols() is one.samples
        assert SignalInput(sig.samples, "f()").symbols() is sig.samples


class TestProperties:
    """Properties of the samples: from the Signal, a conflict raises."""

    @pytest.mark.parametrize(
        ("method", "value"),
        [("sampling_rate", 2e6), ("symbol_rate", 1e6), ("sps", 2.0), ("domain", "DIG")],
    )
    def test_from_signal(self, xp: Any, method: str, value: Any) -> None:
        inp = SignalInput(make_adapter_test_signal(xp), "f()")
        assert getattr(inp, method)() == value
        assert getattr(inp, method)(value) == value

    @pytest.mark.parametrize(
        ("method", "value"),
        [("sampling_rate", 5.0), ("symbol_rate", 3.0), ("sps", 1.5), ("domain", "RF")],
    )
    def test_array_input_needs_the_argument(
        self, xp: Any, method: str, value: Any
    ) -> None:
        inp = SignalInput(xp.ones(4), "f()")
        assert getattr(inp, method)(value) == value
        with pytest.raises(ValueError, match=rf"f\(\) requires {method} for array"):
            getattr(inp, method)()

    def test_conflict_raises(self, xp: Any) -> None:
        inp = SignalInput(make_adapter_test_signal(xp), "f()")
        with pytest.raises(
            ValueError, match=r"f\(\): sampling_rate=1000000.0 conflicts"
        ):
            inp.sampling_rate(1e6)
        with pytest.raises(ValueError, match=r"domain='OPT' conflicts"):
            inp.domain("OPT")

    def test_relative_tolerance(self, xp: Any) -> None:
        """Values within a relative 1e-9 agree, beyond it they conflict."""
        inp = SignalInput(make_adapter_test_signal(xp), "f()")
        assert inp.sps(2.0 * (1 + 1e-12)) == 2.0
        with pytest.raises(ValueError, match="conflicts"):
            inp.sps(2.0 * (1 + 1e-8))

    def test_integer_sps(self, xp: Any) -> None:
        assert SignalInput(make_adapter_test_signal(xp), "f()").integer_sps() == 2
        inp = SignalInput(xp.ones(4), "f()")
        assert inp.integer_sps(3.0000000000000004) == 3
        with pytest.raises(ValueError, match=r"f\(\) requires sps to be a positive"):
            inp.integer_sps(1.5)
        with pytest.raises(ValueError, match="requires sps for array input"):
            inp.integer_sps()

    def test_wavelength_from_optical_signal(self, xp: Any) -> None:
        sig = make_adapter_test_signal(xp, domain="OPT", center_frequency=193.4e12)
        inp = SignalInput(sig, "f()")
        carrier = sig.center_wavelength_nm
        assert inp.wavelength_nm() == carrier
        assert inp.wavelength_nm(carrier * (1 + 0.9e-4)) == carrier
        with pytest.raises(ValueError, match="center_wavelength_nm=.* conflicts"):
            inp.wavelength_nm(carrier * (1 + 1.1e-4))

    def test_wavelength_needs_argument_without_optical_carrier(self, xp: Any) -> None:
        for value in (xp.ones(4), make_adapter_test_signal(xp)):
            inp = SignalInput(value, "f()")
            assert inp.wavelength_nm(1550) == 1550.0
            with pytest.raises(ValueError, match=r"f\(\) requires center_wavelength"):
                inp.wavelength_nm()


class TestDefaults:
    """Defaults: from the Signal, an explicit argument wins silently."""

    def test_constellation_explicit_wins_silently(self, xp: Any, caplog: Any) -> None:
        sig = make_adapter_test_signal(xp, constellation=Constellation.qam(16))
        inp = SignalInput(sig, "f()")
        assert inp.constellation() == Constellation.qam(16)
        assert inp.constellation(Constellation.psk(4)) == Constellation.psk(4)
        assert caplog.text == ""

    def test_constellation_for_array_input(self, xp: Any) -> None:
        inp = SignalInput(xp.ones(4), "f()")
        assert inp.constellation() is None
        assert inp.constellation(Constellation.qam(4)) == Constellation.qam(4)

    def test_constellation_required(self, xp: Any) -> None:
        inp = SignalInput(xp.ones(4), "f()")
        with pytest.raises(
            ValueError,
            match=r"f\(\) needs a constellation for its decisions: pass "
            r"constellation= or a Signal that has one",
        ):
            inp.constellation(required=True, purpose=" for its decisions")

    def test_constellation_of_wrong_type(self, xp: Any) -> None:
        inp = SignalInput(xp.ones(4), "f()")
        with pytest.raises(TypeError, match="constellation must be a Constellation"):
            inp.constellation("16qam")  # type: ignore[arg-type]

    def test_pulse(self, xp: Any) -> None:
        pulse = RRC(rolloff=0.1)
        inp = SignalInput(make_adapter_test_signal(xp, pulse=pulse), "f()")
        assert inp.pulse() is pulse
        taps = np.ones(3)
        assert inp.pulse(taps) is taps
        bare = SignalInput(xp.ones(4), "f()")
        assert bare.pulse() is None
        with pytest.raises(ValueError, match=r"f\(\) needs a pulse \(a Pulse"):
            bare.pulse(required=True, purpose=" (a Pulse or taps)")

    def test_reference_symbols(self, xp: Any) -> None:
        symbols = xp.ones(8, dtype=xp.complex64)
        own = xp.zeros(8, dtype=xp.complex64)
        sig = make_adapter_test_signal(xp, reference=Reference(symbols=symbols))
        inp = SignalInput(sig, "f()")
        assert inp.reference_symbols() is symbols
        assert inp.reference_symbols(own) is own
        bare = SignalInput(make_adapter_test_signal(xp), "f()")
        assert bare.reference_symbols() is None
        with pytest.raises(ValueError, match=r"f\(\) needs reference symbols"):
            bare.reference_symbols(required=True)

    def test_reference_of(self, xp: Any) -> None:
        ref = Reference(symbols=xp.ones(8, dtype=xp.complex64))
        sig: Signal = make_adapter_test_signal(xp, reference=ref)
        assert reference_of(sig) is ref
        assert reference_of(ref) is ref
        assert reference_of(None) is None


class TestRequire:
    """The free validators."""

    def test_require_value(self) -> None:
        assert require_value(3, "x", "f()") == 3
        with pytest.raises(ValueError, match=r"f\(\) needs a x: pass x="):
            require_value(None, "x", "f()")
        with pytest.raises(TypeError, match=r"f\(\): x must be a int, got str"):
            require_value("3", "x", "f()", kind=int)

    @pytest.mark.parametrize("value", [0.0, -1.0, 1.5, float("nan"), float("inf")])
    def test_require_integer_sps_rejects_invalid_values(self, value: float) -> None:
        """Non-integer, zero, negative, NaN, or Inf SPS values raise ValueError."""
        with pytest.raises(ValueError, match=r"example\(\).*positive integer"):
            require_integer_sps(value, "example()")

    @pytest.mark.parametrize("sps", [3.0000000000000004, 2.9999999999999996, 4.0, 1])
    def test_near_integer_sps_accepted(self, sps: float) -> None:
        assert require_integer_sps(sps, "f()") == round(sps)

    def test_sps_from_rates_is_accepted(self) -> None:
        sps = 3e9 / 1e9 * (1 + 2e-16)
        assert require_integer_sps(sps, "f()") == 3

    @pytest.mark.parametrize("sps", [1.5, 2.000001, 0.9999])
    def test_fractional_sps_never_truncated(self, sps: float) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            require_integer_sps(sps, "f()")
