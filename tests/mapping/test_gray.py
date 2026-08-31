"""Tests for constellation geometry and Gray labelling."""

import numpy as np
import pytest

from commkit import mapping


def test_gray_code_edge_cases():
    """Verify Gray code generation for boundary bit depths."""

    # n = 0
    assert np.array_equal(mapping.gray_code(0), np.array([0]))
    assert np.array_equal(mapping.gray_to_binary(0), np.array([0]))

    # n = 2
    # binary: 0, 1, 2, 3
    # gray: 0, 1, 3, 2
    # gray_to_binary should be inverse: mapping[0]=0, mapping[1]=1, mapping[3]=2, mapping[2]=3
    assert np.array_equal(mapping.gray_to_binary(2), np.array([0, 1, 3, 2]))

    # n < 0
    with pytest.raises(ValueError):
        mapping.gray_code(-1)
    with pytest.raises(ValueError):
        mapping.gray_to_binary(-1)


def test_gray_constellation_advanced(xp):
    """Verify constellation generation edge cases."""
    # 1. Unipolar via argument
    const_unipol = mapping.gray_constellation("pam", 4, unipolar=True)
    assert xp.min(const_unipol) >= 0

    # 2. Custom scheme check (now simple string comparison)
    # Note: 'my-pam' no longer automatically resolves to 'ask' unless we use 'pam' or 'ask'
    # Actually gray_constellation uses "ask" in modulation.
    const_custom = mapping.gray_constellation("pam", 4)
    assert len(const_custom) == 4

    # 3. Order error
    with pytest.raises(ValueError, match="at least 2"):
        mapping.gray_constellation("psk", 1)

    # 4. QAM non-power-of-2 (already handled by log2 usually but let's check explicitly)
    # Actually gray_constellation checks if it's power of 2
    with pytest.raises(ValueError, match="power of 2"):
        mapping.gray_constellation("qam", 7)

    # 5. Unknown modulation
    with pytest.raises(ValueError, match="Unsupported modulation type"):
        mapping.gray_constellation("unknown", 4)


def test_gray_code_zero():
    """Verify Gray code for 0 bits."""
    assert np.array_equal(mapping.gray_code(0), [0])
    assert np.array_equal(mapping.gray_to_binary(0), [0])


def test_gray_code_negative():
    """Verify error for negative bits."""
    with pytest.raises(ValueError, match="n must be non-negative"):
        mapping.gray_code(-1)
    with pytest.raises(ValueError, match="n must be non-negative"):
        mapping.gray_to_binary(-1)


def test_constellation_unsupported():
    """Verify error for unknown modulation type."""
    with pytest.raises(ValueError, match="Unsupported modulation type"):
        mapping.gray_constellation("chaos", 4)


def test_constellation_order_error():
    """Verify errors for non-matching orders."""
    with pytest.raises(ValueError, match="Order must be at least 2"):
        mapping.gray_constellation("qam", 1)

    with pytest.raises(ValueError, match="Order must be power of 2"):
        mapping.gray_constellation("qam", 10)

    # Test internal helpers directly to trigger their own ValueError paths.
    with pytest.raises(ValueError, match="Order must be power of 2"):
        from commkit.mapping.gray import _gray_psk

        _gray_psk(3)

    with pytest.raises(ValueError, match="Order must be power of 2"):
        from commkit.mapping.gray import _gray_ask

        _gray_ask(6)


def test_constellation_unsupported_string():
    """Test that unsupported strings raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported modulation type: custom-unknown"):
        mapping.gray_constellation("custom-unknown", 4)


def test_qam_cross_fallback():
    """Trigger the fallback in cross-QAM for small N."""
    # _gray_qam_cross(8) -> n=2, m=1. width=4, height=2.
    # n < 3 triggers n_shift=0 branch
    from commkit.mapping.gray import _gray_qam_cross

    res = _gray_qam_cross(8)
    assert res.shape == (8,)
