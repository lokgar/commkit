"""Signal pipeline characterization tests used as architecture safety checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from commkit import equalization, filtering, frequency, multirate
from commkit.core import Preamble, Signal, SingleCarrierFrame, generation
from commkit.impairments import apply_awgn
from commkit.mapping import demap_symbols_hard, map_bits


def _signal(xp, *, sps: float = 2.0) -> Signal:
    samples = xp.asarray(([1.0 + 0.0j, -1.0 + 0.0j] * 64), dtype=xp.complex64)
    sig = Signal(
        samples=samples,
        sampling_rate=sps * 1e6,
        symbol_rate=1e6,
        mod_scheme="PSK",
        mod_order=2,
        source_bits=xp.asarray([0, 1] * 32),
        source_symbols=xp.asarray([1.0, -1.0] * 32, dtype=xp.complex64),
        pulse_shape="rrc",
        filter_span=4,
        rrc_rolloff=0.25,
        spectral_domain="INTERMEDIATE",
        physical_domain="RF",
    )
    sig.resolved_symbols = xp.asarray([1.0, -1.0], dtype=xp.complex64)
    sig.resolved_bits = xp.asarray([0, 1])
    return sig


def _identity_taps(xp):
    return xp.asarray([1.0 + 0.0j], dtype=xp.complex64)


@pytest.mark.parametrize(
    "operation",
    [
        lambda x, xp: apply_awgn(x, sps=2, esn0_db=30, seed=7),
        lambda x, xp: filtering.matched_filter(x, _identity_taps(xp)),
        lambda x, xp: multirate.resample(x, sps_in=2, sps_out=1.5),
        lambda x, xp: equalization.zf_equalizer(x, _identity_taps(xp)),
    ],
    ids=["awgn", "matched-filter", "fractional-resample", "equalizer"],
)
def test_array_pipeline_operations_return_arrays(backend_device, xp, operation):
    """Array input remains array output across representative pipeline stages."""
    samples = xp.ones(128, dtype=xp.complex64)

    result = operation(samples, xp)

    assert isinstance(result, xp.ndarray)


@pytest.mark.parametrize(
    "operation",
    [
        lambda sig, xp: apply_awgn(sig, esn0_db=30, seed=7),
        lambda sig, xp: filtering.matched_filter(sig, _identity_taps(xp)),
        lambda sig, xp: multirate.resample(sig, sps_out=1.5),
        lambda sig, xp: equalization.zf_equalizer(sig, _identity_taps(xp)),
    ],
    ids=["awgn", "matched-filter", "fractional-resample", "equalizer"],
)
def test_signal_pipeline_operations_are_functional(backend_device, xp, xpt, operation):
    """Signal transforms return a new container and leave their input untouched."""
    sig = _signal(xp)
    original_samples = sig.samples.copy()
    original_rate = sig.sampling_rate

    result = operation(sig, xp)

    assert isinstance(result, Signal)
    assert result is not sig
    xpt.assert_array_equal(sig.samples, original_samples)
    assert sig.sampling_rate == original_rate


def test_required_signal_metadata_takes_precedence(backend_device, xp, xpt):
    """A required Signal field wins over a contradictory duplicate argument."""
    sig = _signal(xp, sps=2.0)

    actual = apply_awgn(sig, sps=99.0, esn0_db=20, seed=11)
    expected = apply_awgn(sig.samples, sps=sig.sps, esn0_db=20, seed=11)

    xpt.assert_allclose(actual.samples, expected)


def test_optional_metadata_falls_back_only_when_signal_field_absent(backend_device, xp):
    """Optional modulation metadata uses arguments only when Signal lacks it."""
    n = 256
    sampling_rate = 1e6
    tone = xp.exp(1j * 2 * xp.pi * 25e3 * xp.arange(n) / sampling_rate)
    without_mod = Signal(samples=tone, sampling_rate=sampling_rate, symbol_rate=0.5e6)
    with_mod = without_mod.model_copy(update={"mod_scheme": "PSK", "mod_order": 4})

    fallback = frequency.estimate_frequency_offset_mth_power(
        without_mod, modulation="PSK", order=4
    )
    explicit = frequency.estimate_frequency_offset_mth_power(
        tone, sampling_rate=sampling_rate, modulation="PSK", order=4
    )
    signal_wins = frequency.estimate_frequency_offset_mth_power(
        with_mod, modulation="PSK", order=2
    )

    assert fallback == pytest.approx(explicit)
    assert signal_wins == pytest.approx(explicit)


@dataclass(frozen=True)
class MetadataCase:
    name: str
    consumed_fields: tuple[str, ...]
    transform: Callable
    expected_rate: float
    output_is_signal: bool = True
    expected_domains: tuple[str, str] = ("INTERMEDIATE", "RF")
    source_fields_valid: bool = True
    resolved_fields_valid: bool = False


METADATA_PROPAGATION_TABLE = (
    MetadataCase(
        "awgn",
        ("sps",),
        lambda sig, xp: apply_awgn(sig, esn0_db=30, seed=3),
        2e6,
    ),
    MetadataCase(
        "matched_filter",
        ("pulse_shape", "sps", "filter_span", "rrc_rolloff"),
        lambda sig, xp: filtering.matched_filter(sig),
        2e6,
    ),
    MetadataCase(
        "fractional_resample",
        ("sps", "symbol_rate"),
        lambda sig, xp: multirate.resample(sig, sps_out=1.5),
        1.5e6,
    ),
    MetadataCase(
        "static_equalizer",
        (),
        lambda sig, xp: equalization.zf_equalizer(sig, _identity_taps(xp)),
        2e6,
    ),
    MetadataCase(
        "symbol_rate_equalizer",
        ("sps", "symbol_rate"),
        lambda sig, xp: equalization.apply_taps(
            sig, _identity_taps(xp), normalize=False
        ),
        1e6,
    ),
)


@pytest.mark.parametrize("case", METADATA_PROPAGATION_TABLE, ids=lambda c: c.name)
def test_metadata_propagation_table(backend_device, xp, case):
    """Executable table defining rate, domain, and provenance propagation."""
    sig = _signal(xp)

    result = case.transform(sig, xp)

    assert isinstance(result, Signal) is case.output_is_signal
    assert result.sampling_rate == pytest.approx(case.expected_rate)
    assert (result.spectral_domain, result.physical_domain) == case.expected_domains
    assert (result.source_bits is not None and result.source_symbols is not None) is (
        case.source_fields_valid
    )
    assert (
        result.resolved_symbols is not None and result.resolved_bits is not None
    ) is case.resolved_fields_valid


def test_fractional_sps_is_preserved_exactly_by_resampling(backend_device, xp):
    """Fractional-SPS-capable paths retain the requested ratio in metadata."""
    sig = _signal(xp, sps=1.5)

    result = multirate.resample(sig, sps_out=2.5)

    assert result.sps == 2.5
    assert result.sampling_rate == 2.5 * sig.symbol_rate


@pytest.mark.parametrize(
    "operation",
    [
        lambda sig, xp: multirate.decimate_to_symbol_rate(sig),
        lambda sig, xp: filtering.shaping_filter_taps(
            sig.model_copy(update={"pulse_shape": "rect"})
        ),
        lambda sig, xp: equalization.apply_taps(
            sig, _identity_taps(xp), normalize=False
        ),
        lambda sig, xp: equalization.cma(sig, num_taps=5),
        lambda sig, xp: equalization.rde(sig, num_taps=5),
        lambda sig, xp: equalization.lms(sig, num_taps=5),
        lambda sig, xp: equalization.rls(sig, num_taps=5),
        lambda sig, xp: equalization.block_cma(sig, num_taps=5),
        lambda sig, xp: equalization.block_rde(sig, num_taps=5),
        lambda sig, xp: equalization.block_lms(sig, num_taps=5),
    ],
    ids=[
        "symbol-decimation",
        "rectangular-taps",
        "frozen-equalizer",
        "cma",
        "rde",
        "lms",
        "rls",
        "block-cma",
        "block-rde",
        "block-lms",
    ],
)
def test_integer_sps_only_paths_reject_fractional_signal_sps(
    backend_device, xp, operation
):
    """Integer-only operations reject 1.5 SPS rather than truncating it to 1."""
    sig = _signal(xp, sps=1.5)

    with pytest.raises(ValueError, match=r"sps.*positive integer|integer.*sps"):
        operation(sig, xp)


def test_pulse_shaping_rejects_fractional_sps_before_resample(backend_device, xp):
    """Direct pulse shaping cannot truncate a fractional resampling factor."""
    symbols = xp.asarray([1.0, -1.0], dtype=xp.complex64)

    with pytest.raises(ValueError, match=r"sps.*positive integer"):
        generation.shape_pulse(symbols, sps=1.5, pulse_shape="rrc")


def test_frame_sample_map_rejects_fractional_sps(backend_device):
    """A sample-domain frame mask requires an integral repeat count."""
    frame = SingleCarrierFrame(payload_len=16)

    with pytest.raises(ValueError, match=r"sps.*positive integer"):
        frame.get_structure_map(unit="samples", sps=1.5)


@pytest.mark.parametrize("sps", [0, -1, 1.5, float("nan"), float("inf")])
@pytest.mark.parametrize("operation", ["decimate", "apply_taps", "resolve"])
def test_array_symbol_operations_validate_sps(backend_device, xp, sps, operation):
    samples = xp.ones(16, dtype=xp.complex64)
    with pytest.raises(ValueError, match="sps to be a positive integer"):
        if operation == "decimate":
            multirate.decimate_to_symbol_rate(samples, sps=sps)
        elif operation == "apply_taps":
            equalization.apply_taps(samples, _identity_taps(xp), sps=sps)
        else:
            multirate.resolve_symbols(samples, sps=sps)


def test_array_symbol_operations_accept_integral_float_sps(backend_device, xp, xpt):
    samples = xp.arange(16, dtype=xp.float32).astype(xp.complex64)
    xpt.assert_array_equal(
        multirate.decimate_to_symbol_rate(samples, sps=2.0), samples[::2]
    )
    actual = equalization.apply_taps(
        samples, _identity_taps(xp), sps=2.0, normalize=False
    )
    expected = equalization.apply_taps(
        samples, _identity_taps(xp), sps=2, normalize=False
    )
    xpt.assert_array_equal(actual, expected)


@pytest.mark.parametrize("sps", [0, -1, 1.5, float("nan"), float("inf")])
@pytest.mark.parametrize("factory", ["qam", "psqam", "preamble", "frame"])
def test_generation_boundaries_validate_sps(backend_device, sps, factory):
    with pytest.raises(ValueError, match="sps to be a positive integer"):
        if factory == "qam":
            generation.generate_qam(16, sps=sps, symbol_rate=1e6, order=4)
        elif factory == "psqam":
            generation.generate_psqam(16, sps=sps, symbol_rate=1e6, order=16, nu=0.3)
        elif factory == "preamble":
            Preamble(sequence_type="barker", length=7).to_signal(
                sps=sps, symbol_rate=1e6
            )
        else:
            SingleCarrierFrame(payload_len=16).to_signal(sps=sps)


@pytest.mark.parametrize("stored_unipolar", [None, False, True])
def test_demap_optional_unipolar_metadata(backend_device, xp, xpt, stored_unipolar):
    bits = xp.asarray([0, 0, 0, 1, 1, 1, 1, 0], dtype=xp.uint8)
    effective = True if stored_unipolar is None else stored_unipolar
    symbols = map_bits(bits, "PAM", 4, unipolar=effective)
    sig = Signal(
        samples=symbols,
        sampling_rate=1e6,
        symbol_rate=1e6,
        mod_scheme="PAM",
        mod_order=4,
        mod_unipolar=stored_unipolar,
        resolved_symbols=symbols,
    )
    result = demap_symbols_hard(sig, unipolar=True)
    xpt.assert_array_equal(result.resolved_bits, bits)
    assert sig.resolved_bits is None


def test_frame_relationship_survives_pipeline(backend_device, xp):
    """Frame-backed Signals keep frame data and populated private caches attached."""
    frame = SingleCarrierFrame(
        payload_len=60,
        payload_mod_scheme="QAM",
        payload_mod_order=16,
        preamble=Preamble(sequence_type="barker", length=13),
        pilot_pattern="comb",
        pilot_period=4,
    )
    # Populate the lazy private arrays before exercising deep-copy rewraps.
    payload_bits = frame.payload_bits
    payload_symbols = frame.payload_symbols
    assert frame.pilot_bits is not None
    assert frame.pilot_symbols is not None
    sig = frame.to_signal(sps=4, symbol_rate=1e6, filter_span=4)
    sig.source_bits = payload_bits
    sig.source_symbols = payload_symbols

    transformed = apply_awgn(sig, esn0_db=25, seed=5)
    transformed = filtering.matched_filter(transformed)
    transformed = multirate.resample(transformed, sps_out=2)
    transformed = equalization.zf_equalizer(transformed, _identity_taps(xp))

    assert transformed.frame is not None
    assert transformed.frame is frame
    assert transformed.frame.payload_bits is not None
    assert transformed.frame.payload_symbols is not None
    assert transformed.frame.pilot_bits is not None
    assert transformed.frame.pilot_symbols is not None
    assert transformed.source_bits is not None
    assert transformed.source_symbols is not None
    assert (
        transformed.frame.get_structure_map().keys() == frame.get_structure_map().keys()
    )
