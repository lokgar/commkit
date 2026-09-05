"""Tests for the array/Signal API boundary helpers."""

import pytest

from commkit.core import Signal
from commkit.core._signal_adapter import (
    adapt_signal,
    require_integer_sps,
)


def _signal(xp, **metadata):
    return Signal(
        samples=xp.ones(16, dtype=xp.complex64),
        sampling_rate=2e6,
        symbol_rate=1e6,
        **metadata,
    )


def test_prepare_array_input_is_passed_through(backend_device, xp):
    samples = xp.ones(8)

    signal_adapter = adapt_signal(samples, function_name="example()")

    assert signal_adapter.array is samples
    assert signal_adapter.signal is None
    assert signal_adapter.resolve_required("sampling_rate", 1e6) == 1e6


def test_required_signal_metadata_wins(backend_device, xp, caplog):
    sig = _signal(xp)
    signal_adapter = adapt_signal(sig, function_name="example()")

    value = signal_adapter.resolve_required("sampling_rate", 99.0)

    assert value == sig.sampling_rate
    assert "ignoring supplied sampling_rate" in caplog.text


def test_required_array_metadata_reports_function_name(backend_device, xp):
    signal_adapter = adapt_signal(xp.ones(8), function_name="example()")

    with pytest.raises(ValueError, match=r"example\(\).*sampling_rate"):
        signal_adapter.resolve_required("sampling_rate")


def test_optional_signal_metadata_precedence_and_fallback(backend_device, xp, caplog):
    populated = _signal(xp, mod_scheme="QAM")
    absent = _signal(xp)
    populated_adapter = adapt_signal(populated, function_name="example()")
    absent_adapter = adapt_signal(absent, function_name="example()")

    assert populated_adapter.resolve_optional("mod_scheme", "PSK") == "QAM"
    assert absent_adapter.resolve_optional("mod_scheme", "PSK") == "PSK"
    assert "falling back to supplied mod_scheme" in caplog.text


@pytest.mark.parametrize("value", [0.0, -1.0, 1.5, float("nan"), float("inf")])
def test_require_integer_sps_rejects_invalid_values(value):
    with pytest.raises(ValueError, match=r"example\(\).*positive integer"):
        require_integer_sps(value, "example()")


def test_signal_adapter_wrap_and_field_replacement(backend_device, xp):
    sig = _signal(xp)
    sig.resolved_bits = xp.asarray([1, 0])
    signal_adapter = adapt_signal(sig, function_name="example()")
    replacement = xp.zeros(8, dtype=xp.complex64)

    transformed = signal_adapter.wrap_samples(replacement, sampling_rate=1e6)
    resolved = signal_adapter.replace_signal_field("resolved_symbols", replacement)

    assert transformed is not sig
    assert transformed.samples is replacement
    assert transformed.sampling_rate == 1e6
    assert resolved.resolved_symbols is replacement
    assert resolved.resolved_bits is None
