"""Equality of the frozen dataclasses (commkit._value).

Results and data (_Record) compare field by field and are unhashable;
value objects with host arrays (_Value) compare and hash by value.
"""

import pickle
from typing import Any

import numpy as np
import pytest

from commkit.core import Signal
from commkit.recovery import (
    BlockPhaseEstimate,
    CarrierPhaseEstimate,
    PilotInterpolation,
)


def _estimate(xp: Any, value: Any) -> CarrierPhaseEstimate:
    return CarrierPhaseEstimate(value=xp.asarray(value))


class TestRecord:
    def test_equal_by_value_not_identity(self, xp: Any) -> None:
        """Two estimates with equal fields are equal (3.10.2 compared
        identity); NaN equals NaN, as a copy of an estimate must equal it."""
        a = _estimate(xp, [0.1, np.nan, 0.3])
        b = _estimate(xp, [0.1, np.nan, 0.3])
        assert a is not b and a == b and not a != b
        assert a != _estimate(xp, [0.1, np.nan, 0.4])

    def test_shape_dtype_and_class_matter(self, xp: Any) -> None:
        a = _estimate(xp, np.zeros(3, np.float64))
        assert a != _estimate(xp, np.zeros(4, np.float64))
        assert a != _estimate(xp, np.zeros(3, np.float32))
        block = BlockPhaseEstimate(
            value=xp.zeros(3), block_centers=np.arange(1.0), block_phase=xp.zeros(1)
        )
        assert a != block and block != a
        assert a != "not an estimate"

    def test_nested_fields_and_sequences(self, xp: Any) -> None:
        """A Signal compares its samples, properties and nested values."""
        x = xp.asarray(np.arange(8, dtype=np.complex64))
        sig = Signal(samples=x, sampling_rate=2e9, symbol_rate=1e9)
        assert sig == sig.clone()
        assert sig != sig.replace(sampling_rate=4e9)
        assert sig != sig.replace(samples=x * 2)

    def test_unhashable(self, xp: Any) -> None:
        with pytest.raises(TypeError, match="unhashable"):
            hash(_estimate(xp, [0.0]))

    @pytest.mark.gpu_only
    def test_devices_never_compare_equal(self, xp: Any) -> None:
        """Equality transfers nothing: a GPU estimate and its host copy differ."""
        a = _estimate(xp, [1.0, 2.0])
        b = _estimate(np, [1.0, 2.0])
        assert a != b


class TestValue:
    def test_equal_and_hashed_by_value(self) -> None:
        """An algorithm object with arrays (identity until 3.10.2) compares
        and hashes by value, so it can key a cache."""
        a = PilotInterpolation(indices=np.arange(4), values=np.ones(4))
        b = PilotInterpolation(indices=np.arange(4), values=np.ones(4))
        assert a == b and hash(a) == hash(b) and len({a, b}) == 1
        assert a != PilotInterpolation(indices=np.arange(4), values=np.zeros(4))

    def test_pickle_drops_the_cached_hash(self) -> None:
        """Hashes of bytes differ between processes: an unpickled object
        recomputes its hash instead of carrying a stale one."""
        a = PilotInterpolation(indices=np.arange(4), values=np.ones(4))
        hash(a)
        assert "_hash" in a.__dict__
        b = pickle.loads(pickle.dumps(a))
        assert "_hash" not in b.__dict__ and a == b
