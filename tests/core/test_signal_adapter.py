"""Tests for the array/Signal API boundary helpers."""

from typing import Any

import pytest

from commkit.core._signal_adapter import adapt_signal, require_integer_sps
from commkit.mapping import Constellation
from tests.common.signals import make_adapter_test_signal


class TestSignalAdapterMetadata:
    """Array input passes through the adapter unchanged."""

    def test_prepare_array_input_is_passed_through(self, xp: Any) -> None:
        """Array input is held directly with None signal container."""
        samples = xp.ones(8)
        signal_adapter = adapt_signal(samples, function_name="example()")

        assert signal_adapter.array is samples
        assert signal_adapter.signal is None
        assert signal_adapter.resolve_fact("sampling_rate", 1e6) == 1e6


class TestSignalAdapterTransforms:
    """Tests for SPS validation and return signal wrapping/field replacement."""

    @pytest.mark.parametrize("value", [0.0, -1.0, 1.5, float("nan"), float("inf")])
    def test_require_integer_sps_rejects_invalid_values(self, value: float) -> None:
        """Non-integer, zero, negative, NaN, or Inf SPS values raise ValueError."""
        with pytest.raises(ValueError, match=r"example\(\).*positive integer"):
            require_integer_sps(value, "example()")

    def test_signal_adapter_wrap_samples(self, xp: Any) -> None:
        """Wrapping samples produces a new Signal sharing the rest."""
        sig = make_adapter_test_signal(xp)
        signal_adapter = adapt_signal(sig, function_name="example()")
        replacement = xp.zeros(8, dtype=xp.complex64)

        transformed = signal_adapter.wrap_samples(replacement, sampling_rate=1e6)

        assert transformed is not sig
        assert transformed.samples is replacement
        assert transformed.sampling_rate == 1e6

    def test_symbol_array_requires_one_sample_per_symbol(self, xp: Any) -> None:
        """symbol_array() passes arrays and 1-SPS Signals, rejects the rest."""
        sig = make_adapter_test_signal(xp)  # 2 samples per symbol
        with pytest.raises(ValueError, match=r"example\(\) needs one sample"):
            adapt_signal(sig, function_name="example()").symbol_array()
        one = sig.replace(sampling_rate=sig.symbol_rate)
        assert adapt_signal(one, function_name="f()").symbol_array() is one.samples
        assert adapt_signal(sig.samples, function_name="f()").symbol_array() is (
            sig.samples
        )


class TestFactsAndChoices:
    """resolve_fact / resolve_choice (plan §2.5)."""

    def test_fact_from_signal(self, xp: Any) -> None:
        a = adapt_signal(make_adapter_test_signal(xp), function_name="f()")
        assert a.resolve_fact("sampling_rate") == 2e6
        assert a.resolve_fact("sampling_rate", 2e6) == 2e6
        assert a.resolve_fact("sps", 2.0 * (1 + 1e-12)) == 2.0

    def test_conflicting_fact_raises(self, xp: Any) -> None:
        a = adapt_signal(make_adapter_test_signal(xp), function_name="f()")
        with pytest.raises(
            ValueError, match=r"f\(\): sampling_rate=1000000.0 conflicts"
        ):
            a.resolve_fact("sampling_rate", 1e6)

    def test_fact_required_for_array_input(self, xp: Any) -> None:
        a = adapt_signal(xp.ones(4), function_name="f()")
        assert a.resolve_fact("sampling_rate", 5.0) == 5.0
        with pytest.raises(ValueError, match="requires sampling_rate"):
            a.resolve_fact("sampling_rate")

    def test_choice_explicit_wins_silently(self, xp: Any, caplog: Any) -> None:
        sig = make_adapter_test_signal(xp, constellation=Constellation.qam(16))
        a = adapt_signal(sig, function_name="f()")
        assert a.resolve_choice("constellation") == Constellation.qam(16)
        assert a.resolve_choice("constellation", Constellation.psk(4)) == (
            Constellation.psk(4)
        )
        assert caplog.text == ""

    def test_choice_for_array_input(self, xp: Any) -> None:
        a = adapt_signal(xp.ones(4), function_name="f()")
        assert a.resolve_choice("constellation") is None
        assert a.resolve_choice("constellation", Constellation.qam(4)) == (
            Constellation.qam(4)
        )

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
