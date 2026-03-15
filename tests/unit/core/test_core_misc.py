# tests/unit/core/test_misc.py

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.misc import (
    PaddingConfig,
    advanced_padding,
    assimilate_shape,
    batched_diag_construct,
    cast_floating_to_numpy,
    ensure_slice_tuple,
    expand_matrix,
    expand_to_3x3,
    find_squarest_divisors,
    get_air_name,
    get_background_material_name,
    index_1d_array,
    index_by_slice,
    index_by_slice_take,
    index_by_slice_take_1d,
    invert_dict,
    is_float_divisible,
    is_index_in_slice,
    linear_interpolated_indexing,
    mask_1d_from_slice,
    normalize_polarization_for_source,
    pad_fields,
    prime_factorization,
)
from fdtdx.materials import Material

# ── expand_matrix ──────────────────────────────────────────────────────


class TestExpandMatrix:
    def test_3d_input(self):
        m = jnp.ones((2, 3, 4))
        result = expand_matrix(m, (2, 3, 1))
        assert result.shape == (4, 9, 4)

    def test_2d_input_gets_expanded(self):
        m = jnp.ones((2, 3))
        result = expand_matrix(m, (1, 1, 1))
        assert result.shape == (2, 3, 1)

    def test_values_are_repeated(self):
        m = jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        result = expand_matrix(m, (2, 3, 1))
        # axis 0: each row repeated 2x, axis 1: each col repeated 3x
        assert result.shape == (4, 6, 1)
        assert float(result[0, 0, 0]) == 1.0
        assert float(result[1, 0, 0]) == 1.0
        assert float(result[2, 0, 0]) == 3.0
        assert float(result[0, 3, 0]) == 2.0


# ── ensure_slice_tuple ─────────────────────────────────────────────────


class TestEnsureSliceTuple:
    def test_int_to_slice(self):
        result = ensure_slice_tuple([3])
        assert result == (slice(3, 4),)

    def test_slice_passthrough(self):
        s = slice(1, 5)
        result = ensure_slice_tuple([s])
        assert result == (s,)

    def test_tuple_to_slice(self):
        result = ensure_slice_tuple([(2, 7)])
        assert result == (slice(2, 7),)

    def test_list_to_slice(self):
        result = ensure_slice_tuple([[1, 4]])
        assert result == (slice(1, 4),)

    def test_mixed_inputs(self):
        result = ensure_slice_tuple([5, slice(0, 3), (1, 10)])
        assert result == (slice(5, 6), slice(0, 3), slice(1, 10))

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid location type"):
            ensure_slice_tuple(["bad"])

    def test_invalid_tuple_length_raises(self):
        with pytest.raises(ValueError, match="Invalid location type"):
            ensure_slice_tuple([(1, 2, 3)])


# ── is_float_divisible ─────────────────────────────────────────────────


class TestIsFloatDivisible:
    def test_exact_division(self):
        assert is_float_divisible(10.0, 5.0)

    def test_not_divisible(self):
        assert not is_float_divisible(10.0, 3.0)

    def test_zero_divisor(self):
        assert not is_float_divisible(10.0, 0.0)

    def test_floating_point_near_zero_remainder(self):
        # 0.3 / 0.1 should be divisible despite float imprecision
        assert is_float_divisible(0.3, 0.1)

    def test_near_divisor_remainder(self):
        # remainder close to b itself
        assert is_float_divisible(1.0, 0.5)


# ── is_index_in_slice ──────────────────────────────────────────────────


class TestIsIndexInSlice:
    def test_in_range(self):
        assert is_index_in_slice(2, slice(1, 5), 10)

    def test_out_of_range(self):
        assert not is_index_in_slice(6, slice(1, 5), 10)

    def test_at_start(self):
        assert is_index_in_slice(1, slice(1, 5), 10)

    def test_at_stop(self):
        assert not is_index_in_slice(5, slice(1, 5), 10)

    def test_negative_slice(self):
        assert is_index_in_slice(8, slice(-3, None), 10)


# ── cast_floating_to_numpy ─────────────────────────────────────────────


class TestCastFloatingToNumpy:
    def test_float_casting(self):
        vals = {"a": np.array([1.0, 2.0], dtype=np.float64)}
        result = cast_floating_to_numpy(vals, np.float32)
        assert result["a"].dtype == np.float32

    def test_complex_to_float_takes_real(self):
        vals = {"a": np.array([1 + 2j, 3 + 4j], dtype=np.complex128)}
        result = cast_floating_to_numpy(vals, np.float64)
        np.testing.assert_array_equal(result["a"], np.array([1.0, 3.0]))

    def test_complex_to_complex_preserves(self):
        vals = {"a": np.array([1 + 2j], dtype=np.complex128)}
        result = cast_floating_to_numpy(vals, jnp.complex64)
        assert np.iscomplexobj(result["a"])

    def test_multiple_keys(self):
        vals = {
            "x": np.array([1.0]),
            "y": np.array([2.0]),
        }
        result = cast_floating_to_numpy(vals, np.float32)
        assert all(v.dtype == np.float32 for v in result.values())


# ── batched_diag_construct ─────────────────────────────────────────────


class TestBatchedDiagConstruct:
    def test_2d_input(self):
        arr = jnp.array([[1, 2, 3], [4, 5, 6]])
        result = batched_diag_construct(arr)
        assert result.shape == (2, 3, 3)
        assert jnp.array_equal(result[0], jnp.diag(jnp.array([1, 2, 3])))
        assert jnp.array_equal(result[1], jnp.diag(jnp.array([4, 5, 6])))

    def test_3d_input(self):
        arr = jnp.ones((2, 3, 4))
        result = batched_diag_construct(arr)
        assert result.shape == (2, 3, 4, 4)

    def test_diagonal_values(self):
        arr = jnp.array([[5, 10]])
        result = batched_diag_construct(arr)
        expected = jnp.array([[[5, 0], [0, 10]]])
        assert jnp.array_equal(result, expected)


# ── invert_dict ────────────────────────────────────────────────────────


class TestInvertDict:
    def test_basic(self):
        assert invert_dict({"a": 1, "b": 2}) == {1: "a", 2: "b"}

    def test_empty(self):
        assert invert_dict({}) == {}


# ── prime_factorization ────────────────────────────────────────────────


class TestPrimeFactorization:
    def test_prime_number(self):
        assert prime_factorization(7) == [7]

    def test_composite(self):
        assert prime_factorization(12) == [2, 2, 3]

    def test_power_of_two(self):
        assert prime_factorization(8) == [2, 2, 2]

    def test_one(self):
        assert prime_factorization(1) == []

    def test_large_prime(self):
        assert prime_factorization(97) == [97]


# ── find_squarest_divisors ─────────────────────────────────────────────


class TestFindSquarestDivisors:
    def test_perfect_square(self):
        a, b = find_squarest_divisors(16)
        assert a * b == 16
        assert a == 4 and b == 4

    def test_prime(self):
        a, b = find_squarest_divisors(7)
        assert a * b == 7
        assert {a, b} == {1, 7}

    def test_composite(self):
        a, b = find_squarest_divisors(12)
        assert a * b == 12


# ── index_1d_array ─────────────────────────────────────────────────────


class TestIndex1dArray:
    def test_basic(self):
        arr = jnp.array([10, 20, 30, 40])
        assert int(index_1d_array(arr, jnp.array(30))) == 2

    def test_first_element(self):
        arr = jnp.array([5, 10, 15])
        assert int(index_1d_array(arr, jnp.array(5))) == 0

    def test_non_1d_raises(self):
        arr = jnp.ones((2, 3))
        with pytest.raises(Exception, match="1d-array"):
            index_1d_array(arr, jnp.array(1))


# ── index_by_slice ─────────────────────────────────────────────────────


class TestIndexBySlice:
    def test_basic(self):
        arr = jnp.arange(24).reshape(2, 3, 4)
        result = index_by_slice(arr, 1, 3, axis=1)
        assert result.shape == (2, 2, 4)
        assert jnp.array_equal(result, arr[:, 1:3, :])

    def test_none_start_stop(self):
        arr = jnp.arange(12).reshape(3, 4)
        result = index_by_slice(arr, None, None, axis=0)
        assert jnp.array_equal(result, arr)

    def test_with_step(self):
        arr = jnp.arange(10)
        result = index_by_slice(arr, 0, 10, axis=0, step=2)
        assert jnp.array_equal(result, arr[::2])


# ── index_by_slice_take_1d ─────────────────────────────────────────────


class TestIndexBySliceTake1d:
    def test_subset(self):
        arr = jnp.arange(10)
        result = index_by_slice_take_1d(arr, slice(2, 5), axis=0)
        assert jnp.array_equal(result, jnp.array([2, 3, 4]))

    def test_full_slice_returns_original(self):
        arr = jnp.arange(5)
        result = index_by_slice_take_1d(arr, slice(None), axis=0)
        assert jnp.array_equal(result, arr)

    def test_empty_slice_raises(self):
        arr = jnp.arange(5)
        with pytest.raises(Exception, match="Invalid slice"):
            index_by_slice_take_1d(arr, slice(3, 3), axis=0)

    def test_multidim(self):
        arr = jnp.arange(12).reshape(3, 4)
        result = index_by_slice_take_1d(arr, slice(1, 3), axis=1)
        assert jnp.array_equal(result, arr[:, 1:3])


# ── index_by_slice_take ────────────────────────────────────────────────


class TestIndexBySliceTake:
    def test_multi_axis(self):
        arr = jnp.arange(24).reshape(2, 3, 4)
        result = index_by_slice_take(arr, [slice(0, 1), slice(1, 3), slice(None)])
        assert result.shape == (1, 2, 4)

    def test_full_slices_noop(self):
        arr = jnp.arange(6).reshape(2, 3)
        result = index_by_slice_take(arr, [slice(None), slice(None)])
        assert jnp.array_equal(result, arr)

    def test_empty_slice_raises(self):
        arr = jnp.arange(6).reshape(2, 3)
        with pytest.raises(Exception, match="Invalid slice"):
            index_by_slice_take(arr, [slice(1, 1), slice(None)])


# ── mask_1d_from_slice ─────────────────────────────────────────────────


class TestMask1dFromSlice:
    def test_basic(self):
        mask = mask_1d_from_slice(slice(1, 4), 6)
        expected = jnp.array([False, True, True, True, False, False])
        assert jnp.array_equal(mask, expected)

    def test_full(self):
        mask = mask_1d_from_slice(slice(None), 3)
        assert jnp.all(mask)

    def test_with_step(self):
        mask = mask_1d_from_slice(slice(0, 6, 2), 6)
        expected = jnp.array([True, False, True, False, True, False])
        assert jnp.array_equal(mask, expected)


# ── assimilate_shape ───────────────────────────────────────────────────


class TestAssimilateShape:
    def test_basic_reshape(self):
        arr = jnp.array([1, 2, 3])
        ref = jnp.ones((3, 4, 5))
        result = assimilate_shape(arr, ref, ref_axes=(0,))
        assert result.shape == (3, 1, 1)

    def test_with_repeat(self):
        # size-1 dim mapped to ref axis → gets repeated
        arr = jnp.array([1])  # shape (1,)
        ref = jnp.ones((3,))
        result = assimilate_shape(arr, ref, ref_axes=(0,), repeat_single_dims=True)
        assert result.shape == (3,)

    def test_without_repeat(self):
        # size-1 dim mapped, no repeat → stays 1
        arr = jnp.array([1])  # shape (1,)
        ref = jnp.ones((3,))
        result = assimilate_shape(arr, ref, ref_axes=(0,))
        assert result.shape == (1,)

    def test_ndim_mismatch_raises(self):
        arr = jnp.array([1, 2])
        ref = jnp.ones((3, 4))
        with pytest.raises(Exception, match="Invalid axes"):
            assimilate_shape(arr, ref, ref_axes=(0, 1))  # arr is 1d, ref_axes has 2 entries - ok
        # but arr is 1d and ref_axes has 1 entry for a 3d ref
        with pytest.raises(Exception, match="Invalid axes"):
            assimilate_shape(arr, jnp.ones((3,)), ref_axes=(0, 1))

    def test_axis_out_of_range_raises(self):
        arr = jnp.array([1, 2, 3])
        ref = jnp.ones((3,))
        with pytest.raises(Exception, match="Invalid axes"):
            assimilate_shape(arr, ref, ref_axes=(5,))

    def test_shape_mismatch_raises(self):
        arr = jnp.array([1, 2])
        ref = jnp.ones((3, 4))
        with pytest.raises(Exception, match="Invalid shapes"):
            assimilate_shape(arr, ref, ref_axes=(0,))


# ── linear_interpolated_indexing ───────────────────────────────────────


class TestLinearInterpolatedIndexing:
    def test_exact_integer_point(self):
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = linear_interpolated_indexing(jnp.array([0.0, 1.0]), arr)
        assert jnp.allclose(result, 2.0, atol=1e-5)

    def test_midpoint(self):
        arr = jnp.array([[0.0, 0.0], [0.0, 4.0]])
        result = linear_interpolated_indexing(jnp.array([0.5, 0.5]), arr)
        assert jnp.allclose(result, 1.0, atol=1e-5)

    def test_dimension_mismatch_raises(self):
        arr = jnp.ones((3, 3))
        with pytest.raises(Exception, match="Invalid shape"):
            linear_interpolated_indexing(jnp.array([1.0]), arr)

    def test_2d_point_raises(self):
        arr = jnp.ones((3, 3))
        with pytest.raises(Exception, match="Invalid shape"):
            linear_interpolated_indexing(jnp.ones((2, 2)), arr)


# ── get_air_name ───────────────────────────────────────────────────────


class TestGetAirName:
    def test_fallback_to_first_material(self, capsys):
        # With 9-tuple permittivity, v.permittivity == 1 is always False
        # so fallback path is exercised
        materials = {"glass": Material(permittivity=2.25), "air": Material(permittivity=1.0)}
        result = get_air_name(materials)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert result == list(materials.keys())[0]

    def test_single_material_fallback(self, capsys):
        materials = {"silicon": Material(permittivity=11.7)}
        result = get_air_name(materials)
        assert result == "silicon"
        captured = capsys.readouterr()
        assert "Warning" in captured.out


# ── get_background_material_name ───────────────────────────────────────


class TestGetBackgroundMaterialName:
    def test_picks_lowest_permittivity(self):
        materials = {
            "silicon": Material(permittivity=11.7),
            "air": Material(permittivity=1.0),
            "glass": Material(permittivity=2.25),
        }
        assert get_background_material_name(materials) == "air"

    def test_empty_raises(self):
        with pytest.raises(Exception, match="Empty Material dictionary"):
            get_background_material_name({})


# ── PaddingConfig & advanced_padding ───────────────────────────────────


class TestAdvancedPadding:
    def test_constant_padding(self):
        arr = jnp.ones((3, 4))
        cfg = PaddingConfig(widths=[1], modes=["constant"], values=[0.0])
        result, slices = advanced_padding(arr, cfg)
        # Each dim padded by 1 on both sides: (3+2, 4+2)
        assert result.shape == (5, 6)
        assert jnp.array_equal(result[tuple(slices)], arr)

    def test_edge_padding(self):
        arr = jnp.arange(6).reshape(2, 3).astype(jnp.float32)
        cfg = PaddingConfig(widths=[2], modes=["edge"])
        result, slices = advanced_padding(arr, cfg)
        assert result.shape == (6, 7)
        assert jnp.array_equal(result[tuple(slices)], arr)

    def test_invalid_widths_raises(self):
        arr = jnp.ones((3, 4))
        cfg = PaddingConfig(widths=[1, 2, 3], modes=["constant"])
        with pytest.raises(Exception, match="Invalid padding width"):
            advanced_padding(arr, cfg)

    def test_invalid_modes_raises(self):
        arr = jnp.ones((3, 4))
        cfg = PaddingConfig(widths=[1], modes=["constant", "edge", "constant"])
        with pytest.raises(Exception, match="Invalid padding width"):
            advanced_padding(arr, cfg)

    def test_invalid_values_raises(self):
        arr = jnp.ones((3, 4))
        cfg = PaddingConfig(widths=[1], modes=["constant"], values=[0.0, 1.0, 2.0])
        with pytest.raises(Exception, match="Invalid padding width"):
            advanced_padding(arr, cfg)

    def test_per_edge_widths(self):
        arr = jnp.ones((3, 4))
        # 2d array -> 4 edges: before-dim0, after-dim0, before-dim1, after-dim1
        cfg = PaddingConfig(widths=[1, 2, 3, 0], modes=["constant"], values=[0.0])
        result, slices = advanced_padding(arr, cfg)
        assert result.shape == (3 + 1 + 2, 4 + 3 + 0)
        assert jnp.array_equal(result[tuple(slices)], arr)

    def test_none_values_default_to_zero(self):
        arr = jnp.ones((2, 3))
        cfg = PaddingConfig(widths=[1], modes=["constant"])
        result, slices = advanced_padding(arr, cfg)
        # corners should be 0
        assert float(result[0, 0]) == 0.0


# ── normalize_polarization_for_source ──────────────────────────────────


class TestNormalizePolarizationForSource:
    def test_e_pol_given(self):
        e_pol, h_pol = normalize_polarization_for_source(
            direction="+",
            propagation_axis=0,
            fixed_E_polarization_vector=(0.0, 1.0, 0.0),
        )
        assert e_pol.shape == (3,)
        assert h_pol.shape == (3,)
        assert jnp.allclose(jnp.linalg.norm(e_pol), 1.0, atol=1e-5)

    def test_h_pol_given(self):
        e_pol, h_pol = normalize_polarization_for_source(
            direction="-",
            propagation_axis=2,
            fixed_H_polarization_vector=(1.0, 0.0, 0.0),
        )
        assert e_pol.shape == (3,)
        assert h_pol.shape == (3,)
        assert jnp.allclose(jnp.linalg.norm(h_pol), 1.0, atol=1e-5)

    def test_neither_raises(self):
        with pytest.raises(Exception, match="Need to specify"):
            normalize_polarization_for_source(direction="+", propagation_axis=0)

    def test_both_given(self):
        e_pol, h_pol = normalize_polarization_for_source(
            direction="+",
            propagation_axis=1,
            fixed_E_polarization_vector=(0.0, 0.0, 1.0),
            fixed_H_polarization_vector=(1.0, 0.0, 0.0),
        )
        assert e_pol.shape == (3,)
        assert h_pol.shape == (3,)

    def test_normalization(self):
        e_pol, h_pol = normalize_polarization_for_source(
            direction="+",
            propagation_axis=0,
            fixed_E_polarization_vector=(0.0, 3.0, 4.0),
        )
        assert jnp.allclose(jnp.linalg.norm(e_pol), 1.0, atol=1e-5)


# ── expand_to_3x3 ─────────────────────────────────────────────────────


class TestExpandTo3x3:
    def test_none(self):
        assert expand_to_3x3(None) is None

    def test_scalar(self):
        result = expand_to_3x3(1.0)
        assert result.shape == (3, 3, 1, 1, 1)
        assert float(result[0, 0, 0, 0, 0]) == 1.0
        assert float(result[0, 1, 0, 0, 0]) == 0.0
        assert float(result[1, 1, 0, 0, 0]) == 1.0

    def test_isotropic(self):
        arr = jnp.ones((1, 4, 4, 4)) * 2.0
        result = expand_to_3x3(arr)
        assert result.shape == (3, 3, 4, 4, 4)
        assert jnp.allclose(result[0, 0], 2.0)
        assert jnp.allclose(result[0, 1], 0.0)

    def test_diagonal(self):
        arr = jnp.stack([jnp.full((2, 2, 2), i + 1.0) for i in range(3)])
        result = expand_to_3x3(arr)
        assert result.shape == (3, 3, 2, 2, 2)
        assert jnp.allclose(result[0, 0], 1.0)
        assert jnp.allclose(result[1, 1], 2.0)
        assert jnp.allclose(result[2, 2], 3.0)
        assert jnp.allclose(result[0, 1], 0.0)

    def test_full_tensor(self):
        arr = jnp.arange(9 * 2 * 2 * 2, dtype=jnp.float32).reshape(9, 2, 2, 2)
        result = expand_to_3x3(arr)
        assert result.shape == (3, 3, 2, 2, 2)
        assert jnp.array_equal(result, arr.reshape(3, 3, 2, 2, 2))


# ── pad_fields ─────────────────────────────────────────────────────────


class TestPadFields:
    def test_shape_all_periodic(self):
        fields = jnp.ones((3, 10, 10, 10))
        result = pad_fields(fields, (True, True, True))
        assert result.shape == (3, 12, 12, 12)

    def test_periodic_wrapping(self):
        fields = jnp.arange(3 * 5 * 5 * 5).reshape(3, 5, 5, 5).astype(jnp.float32)
        result = pad_fields(fields, (True, True, True))
        assert jnp.allclose(result[:, 0, 1:-1, 1:-1], fields[:, -1, :, :])
        assert jnp.allclose(result[:, -1, 1:-1, 1:-1], fields[:, 0, :, :])

    def test_constant_zeros(self):
        fields = jnp.arange(3 * 5 * 5 * 5).reshape(3, 5, 5, 5).astype(jnp.float32)
        result = pad_fields(fields, (False, False, False))
        assert jnp.allclose(result[:, 0, :, :], 0.0)
        assert jnp.allclose(result[:, -1, :, :], 0.0)
        assert jnp.allclose(result[:, 1:-1, 1:-1, 1:-1], fields)

    def test_mixed_periodic(self):
        fields = jnp.arange(3 * 5 * 5 * 5).reshape(3, 5, 5, 5).astype(jnp.float32)
        result = pad_fields(fields, (True, False, False))
        # X-axis wrapped
        assert jnp.allclose(result[:, 0, 1:-1, 1:-1], fields[:, -1, :, :])
        # Y-axis zeros
        assert jnp.allclose(result[:, :, 0, :], 0.0)
