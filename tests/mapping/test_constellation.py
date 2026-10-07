"""Tests for the Constellation value object (mapping.constellation)."""

import dataclasses
from typing import Any

import numpy as np
import pytest

from commkit import mapping
from commkit.mapping import Constellation
from commkit.mapping.gray import _gray_points


def _natural_labels(order: int) -> np.ndarray:
    k = int(np.log2(order))
    return ((np.arange(order)[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.int8)


class TestFactories:
    @pytest.mark.parametrize(
        "factory, family, order",
        [
            (Constellation.qam, "qam", 4),
            (Constellation.qam, "qam", 8),
            (Constellation.qam, "qam", 16),
            (Constellation.qam, "qam", 32),
            (Constellation.qam, "qam", 256),
            (Constellation.psk, "psk", 8),
            (Constellation.pam, "pam", 4),
        ],
    )
    def test_points_labels_and_unit_power(self, factory, family, order) -> None:
        c = factory(order)
        np.testing.assert_array_equal(
            c.points, mapping.gray._gray_points(family, order)
        )
        np.testing.assert_array_equal(c.bit_labels, _natural_labels(order))
        assert c.family == family
        assert c.order == order
        assert c.bits_per_symbol == int(np.log2(order))
        assert c.pmf is None
        assert c.power() == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("order", [16, 64])
    def test_square_qam_is_gray_labelled(self, order) -> None:
        """Nearest neighbours differ in exactly one bit (the Gray property)."""
        c = Constellation.qam(order)
        d = np.abs(c.points[:, None] - c.points[None, :])
        np.fill_diagonal(d, np.inf)
        d_min = d.min()
        for i, j in zip(*np.nonzero(np.isclose(d, d_min))):
            assert np.sum(c.bit_labels[i] != c.bit_labels[j]) == 1

    def test_pam_unipolar(self) -> None:
        c = Constellation.pam(4, unipolar=True)
        assert c.unipolar
        assert np.all(c.points >= 0)
        assert c.power() == pytest.approx(1.0)
        assert not Constellation.pam(4).unipolar
        assert not Constellation.qam(4).unipolar

    def test_factories_are_cached(self) -> None:
        assert Constellation.qam(16) is Constellation.qam(16)

    @pytest.mark.parametrize("order", [1, 6, 2.0])
    def test_invalid_order_raises(self, order) -> None:
        with pytest.raises(ValueError):
            Constellation.qam(order)

    def test_repr(self) -> None:
        assert repr(Constellation.qam(16)) == "Constellation(16-QAM)"
        assert repr(Constellation.pam(2, unipolar=True)) == (
            "Constellation(2-PAM, unipolar)"
        )
        assert "shaped H=5.000 bits" in repr(Constellation.qam(64).shaped(entropy=5))
        assert repr(Constellation([1j, -1j])) == "Constellation(2-custom)"


class TestConstruction:
    def test_arbitrary_points_default_to_natural_labels(self) -> None:
        c = Constellation([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        np.testing.assert_array_equal(c.bit_labels, _natural_labels(4))
        assert c.family is None
        assert c.points.dtype == np.complex128

    def test_real_points_stored_as_float64(self) -> None:
        c = Constellation([-1, 1])
        assert c.points.dtype == np.float64
        assert not c.is_complex

    def test_arrays_are_read_only_copies(self) -> None:
        pts = np.array([-1.0, 1.0])
        c = Constellation(pts)
        pts[0] = 5.0
        assert c.points[0] == -1.0
        for arr in (c.points, c.bit_labels):
            with pytest.raises(ValueError):
                arr[0] = 0

    def test_frozen(self) -> None:
        c = Constellation.qam(4)
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.points = np.zeros(4)  # type: ignore[misc]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(points=[1.0, 2.0, 3.0]), "power of 2"),
            (dict(points=[1.0]), "M >= 2"),
            (dict(points=[[1.0, 2.0]]), "shape"),
            (dict(points=[1.0, 1.0]), "distinct"),
            (dict(points=[1.0, np.inf]), "finite"),
            (dict(points=[-1.0, 1.0], bit_labels=[[0], [0]]), "once"),
            (dict(points=[-1.0, 1.0], bit_labels=[[0], [2]]), "0 and 1"),
            (dict(points=[-1.0, 1.0], bit_labels=[[0, 1], [1, 0]]), "shape"),
            (dict(points=[-1.0, 1.0], pmf=[0.5, 0.6]), "sum to 1"),
            (dict(points=[-1.0, 1.0], pmf=[1.5, -0.5]), "non-negative"),
            (dict(points=[-1.0, 1.0], pmf=[1.0]), "shape"),
            (dict(points=[-1.0, 1.0], family="ask"), "family"),
        ],
    )
    def test_validation(self, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            Constellation(**kwargs)

    @pytest.mark.gpu_only
    def test_device_points_rejected(self, xp: Any) -> None:
        with pytest.raises(TypeError, match="points"):
            Constellation(xp.asarray([-1.0, 1.0]))


class TestValueSemantics:
    def test_equality_by_value(self) -> None:
        q = Constellation.qam(16)
        custom = Constellation(q.points)
        assert custom == q
        assert hash(custom) == hash(q)
        assert q != Constellation.psk(16)
        assert q != q.shaped(nu=0.0)  # an explicit uniform pmf is still a pmf
        assert q != Constellation(q.points[::-1])

    def test_negative_zero_hashes_like_zero(self) -> None:
        a = Constellation([0.0, 1.0])
        b = Constellation([-0.0, 1.0])
        assert a == b
        assert hash(a) == hash(b)

    def test_usable_as_dict_key(self) -> None:
        assert {Constellation.qam(16): 1}[Constellation(Constellation.qam(16).points)]


class TestShaping:
    @pytest.mark.parametrize("order", [16, 32, 64, 256])
    def test_nu_is_defined_on_the_odd_integer_grid(self, order) -> None:
        """P(s) ∝ exp(-nu |s|^2) with s on the odd-integer QAM grid."""
        nu = 0.03
        grid = _gray_points("qam", order, normalize=False)
        expected = np.exp(-nu * np.abs(grid) ** 2)
        expected /= expected.sum()
        c = Constellation.qam(order).shaped(nu=nu)
        np.testing.assert_allclose(c.pmf, expected, rtol=1e-12)
        np.testing.assert_allclose(
            mapping.maxwell_boltzmann(Constellation.qam(order), nu=nu),
            expected,
            rtol=1e-12,
        )

    def test_unit_power_and_unchanged_geometry(self) -> None:
        q = Constellation.qam(64)
        s = q.shaped(nu=0.05)
        assert s.power() == pytest.approx(1.0, abs=1e-12)
        assert np.mean(np.abs(s.points) ** 2) > 1.0  # outer points are rarer
        ratio = s.points / q.points
        np.testing.assert_allclose(ratio, ratio[0])
        np.testing.assert_array_equal(s.bit_labels, q.bit_labels)
        assert s.family == "qam"

    def test_entropy_target(self) -> None:
        s = Constellation.qam(64).shaped(entropy=5.2)
        assert s.entropy == pytest.approx(5.2, abs=1e-9)
        nu = mapping.optimal_nu(Constellation.qam(64), entropy=5.2)
        np.testing.assert_allclose(
            s.pmf, mapping.maxwell_boltzmann(Constellation.qam(64), nu=nu), atol=1e-12
        )

    def test_full_entropy_and_zero_nu_are_uniform(self) -> None:
        for s in (
            Constellation.qam(16).shaped(entropy=4.0),
            Constellation.qam(16).shaped(nu=0.0),
        ):
            np.testing.assert_allclose(s.pmf, np.full(16, 1 / 16))
            assert s.entropy == pytest.approx(4.0)

    def test_pam_shaping_uses_integer_grid(self) -> None:
        """PAM levels ±1, ±3 are the integer grid; nu acts on those energies."""
        s = Constellation.pam(4).shaped(nu=0.1)
        levels = np.round(s.points / np.min(np.abs(s.points))).astype(int)
        expected = np.exp(-0.1 * levels**2.0)
        np.testing.assert_allclose(s.pmf, expected / expected.sum())

    @pytest.mark.parametrize(
        "kwargs", [dict(), dict(nu=0.1, entropy=3.0), dict(nu=-1.0), dict(entropy=5.0)]
    )
    def test_invalid_arguments(self, kwargs) -> None:
        with pytest.raises(ValueError):
            Constellation.qam(16).shaped(**kwargs)

    def test_constant_energy_raises(self) -> None:
        with pytest.raises(ValueError, match="same energy"):
            Constellation.psk(8).shaped(nu=0.1)


class TestOperations:
    def test_map_demap_roundtrip(self, xp: Any, xpt: Any) -> None:
        c = Constellation.qam(16)
        bits = xp.asarray(np.random.default_rng(0).integers(0, 2, (2, 400)), "int8")
        syms = c.map(bits)
        assert syms.shape == (2, 100)
        assert syms.dtype == xp.complex64
        assert isinstance(syms, type(bits))
        xpt.assert_array_equal(
            syms[0], mapping.map_bits(bits[0], constellation=Constellation.qam(16))
        )
        xpt.assert_array_equal(c.demap(syms), bits)

    def test_real_constellation_maps_to_float32(self, xp: Any) -> None:
        syms = Constellation.pam(4).map(xp.asarray([0, 0, 1, 1]))
        assert syms.dtype == xp.float32

    def test_map_follows_custom_labels(self, xp: Any, xpt: Any) -> None:
        """A point carries its own label, whatever its position in the array."""
        labels = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
        c = Constellation([1, 1j, -1, -1j], bit_labels=labels)
        bits = xp.asarray([0, 0, 0, 1, 1, 0, 1, 1])
        xpt.assert_array_equal(c.map(bits), xp.asarray([1j, -1j, -1, 1], "complex64"))
        xpt.assert_array_equal(c.demap(c.map(bits)), bits)

    def test_map_rejects_partial_symbol(self, xp: Any) -> None:
        with pytest.raises(ValueError, match="bits"):
            Constellation.qam(16).map(xp.zeros(6, dtype="int8"))

    def test_demap_decides_nearest_point(self, xp: Any, xpt: Any) -> None:
        c = Constellation([-1.0, 1.0])
        bits = c.demap(xp.asarray([-0.2, 0.1, 3.0]))
        xpt.assert_array_equal(bits, xp.asarray([0, 1, 1], "int8"))

    def test_llr_matches_compute_llr(self, xp: Any, xpt: Any) -> None:
        c = Constellation.qam(16)
        rng = np.random.default_rng(1)
        syms = xp.asarray(
            c.points[rng.integers(0, 16, 200)] + 0.2 * rng.standard_normal(200)
        )
        for method in ("maxlog", "exact"):
            xpt.assert_allclose(
                c.llr(syms, noise_var=0.1, method=method),
                mapping.compute_llr(
                    syms,
                    noise_var=0.1,
                    constellation=Constellation.qam(16),
                    method=method,
                ),
                atol=1e-5,
            )

    def test_llr_sign_follows_labels(self, xp: Any, xpt: Any) -> None:
        labels = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
        c = Constellation([1, 1j, -1, -1j], bit_labels=labels)
        llr = c.llr(xp.asarray(c.points.astype("complex64")), noise_var=0.01)
        xpt.assert_array_equal(llr.reshape(4, 2) < 0, xp.asarray(labels.astype(bool)))

    def test_shaped_llr_is_scale_consistent(self, xp: Any, xpt: Any) -> None:
        """Shaped LLRs equal the LLRs of the same prior on the unit-power grid."""
        pmf = mapping.maxwell_boltzmann(Constellation.qam(16), nu=0.1)
        s = Constellation.qam(16).shaped(nu=0.1)
        a = float(abs(s.points[0] / Constellation.qam(16).points[0]))
        rng = np.random.default_rng(2)
        y = Constellation.qam(16).points[rng.integers(0, 16, 100)] + 0.1 * (
            rng.standard_normal(100) + 1j * rng.standard_normal(100)
        )
        xpt.assert_allclose(
            s.llr(xp.asarray(y * a), noise_var=0.05 * a**2, method="exact"),
            mapping.compute_llr(
                xp.asarray(y),
                noise_var=0.05,
                constellation=Constellation(Constellation.qam(16).points, pmf=pmf),
                method="exact",
            ),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_llr_rejects_unknown_method(self, xp: Any) -> None:
        with pytest.raises(ValueError, match="method"):
            Constellation.qam(4).llr(
                xp.zeros(2, "complex64"), noise_var=1.0, method="x"
            )
