"""Unit tests for the low-level mirror-symmetry primitives."""

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.physics.symmetry import (
    component_sits_on_plane,
    field_component_parity,
    mirror_about_interior_plane,
    mirror_edge_coordinates,
    mirror_extend_low_side,
    mirror_material_array,
    mirror_material_cross_section,
    project_onto_parity,
    restrict_to_kept_half,
)


class TestComponentSitsOnPlane:
    """Which Yee samples lie on a mirror plane, i.e. which mirror index map applies."""

    def test_electric_components(self):
        # E_c is offset half a cell along c only, so along any other axis it sits at integer
        # positions - on the plane.
        for axis in range(3):
            for component in range(3):
                assert component_sits_on_plane("E", component, axis) == (component != axis)

    def test_magnetic_components(self):
        # H_c is offset along the two axes other than c, so the reverse holds.
        for axis in range(3):
            for component in range(3):
                assert component_sits_on_plane("H", component, axis) == (component == axis)

    def test_components_that_vanish_are_exactly_the_on_plane_odd_ones(self):
        # Consistency of the two tables: whatever a wall drives to zero must be sampled on the plane,
        # otherwise the condition cannot be enforced pointwise (this is why a PMC wall cannot be
        # implemented by zeroing tangential H at the first cell).
        for wall, field_type, expect_on_plane in ((-1, "E", True), (1, "H", False)):
            for axis in range(3):
                for component in range(3):
                    if field_component_parity(field_type, component, axis, wall) == -1:
                        assert component_sits_on_plane(field_type, component, axis) == expect_on_plane

    def test_invalid_field_type(self):
        with pytest.raises(ValueError):
            component_sits_on_plane("B", 0, 0)  # type: ignore[arg-type]


class TestMirrorMaterialArray:
    def test_isotropic_and_diagonal_only_flip(self):
        for num_components in (1, 3):
            arr = jnp.asarray(np.arange(num_components * 4).reshape(num_components, 4, 1, 1), dtype=jnp.float32)
            out = mirror_material_array(arr, axis=0)
            assert jnp.allclose(out, jnp.flip(arr, axis=1))

    def test_full_tensor_flips_sign_of_mixed_off_diagonals(self):
        # eps_xy and eps_yx change sign under a mirror normal to x; eps_yz does not.
        arr = jnp.ones((9, 2, 1, 1), dtype=jnp.float32)
        out = mirror_material_array(arr, axis=0)
        expected_signs = [1, -1, -1, -1, 1, 1, -1, 1, 1]  # row-major (xx, xy, xz, yx, ...)
        for index, sign in enumerate(expected_signs):
            assert jnp.allclose(out[index], sign * jnp.ones((2, 1, 1)))

    def test_cross_section_doubles_and_restrict_inverts(self):
        arr = jnp.asarray(np.arange(6).reshape(1, 1, 6, 1), dtype=jnp.float32)
        full = mirror_material_cross_section(arr, axes=(1,))
        assert full.shape == (1, 1, 12, 1)
        assert jnp.allclose(full[:, :, 6:], arr)
        assert jnp.allclose(full[:, :, :6], jnp.flip(arr, axis=2))
        assert jnp.allclose(restrict_to_kept_half(full, axes=(1,)), arr)


class TestMirrorEdgeCoordinates:
    def test_widths_are_mirrored_and_kept_half_preserved(self):
        edges = jnp.asarray([0.0, 1.0, 3.0, 6.0])  # widths 1, 2, 3
        full = mirror_edge_coordinates(edges)
        assert full.shape == (7,)
        widths = np.diff(np.asarray(full))
        assert np.allclose(widths, [3.0, 2.0, 1.0, 1.0, 2.0, 3.0])
        # The kept half keeps its original coordinates, and the plane is at the original first edge.
        assert np.allclose(np.asarray(full)[3:], np.asarray(edges))


class TestMirrorIndexMaps:
    def test_interior_plane_off_plane_is_plain_flip(self):
        arr = jnp.asarray([[0.0, 1.0, 2.0, 3.0]]).reshape(1, 4, 1, 1)
        out = mirror_about_interior_plane(arr, axis=1, on_plane=False)
        assert np.allclose(np.asarray(out).ravel(), [3.0, 2.0, 1.0, 0.0])

    def test_interior_plane_on_plane_pairs_around_the_plane_row(self):
        # 4 samples, plane between index 1 and 2 -> index 2 is on the plane and its own mirror,
        # index 1 <-> 3, index 0 has no partner and is left alone.
        arr = jnp.asarray([0.0, 1.0, 2.0, 3.0]).reshape(1, 4, 1, 1)
        out = np.asarray(mirror_about_interior_plane(arr, axis=1, on_plane=True)).ravel()
        assert out[2] == 2.0  # self-mirrored
        assert out[1] == 3.0 and out[3] == 1.0  # swapped
        assert out[0] == 0.0  # unpaired, untouched

    def test_low_side_off_plane_mirrors_every_sample(self):
        arr = jnp.asarray([10.0, 20.0, 30.0]).reshape(1, 3, 1, 1)
        low = np.asarray(mirror_extend_low_side(arr, axis=1, parity=-1, on_plane=False)).ravel()
        assert np.allclose(low, [-30.0, -20.0, -10.0])

    def test_low_side_on_plane_does_not_duplicate_the_plane_row(self):
        # Index 0 is on the plane: only 1 and 2 have mirror images, and the outermost reconstructed
        # sample repeats its neighbour because its partner lies outside the kept half.
        arr = jnp.asarray([10.0, 20.0, 30.0]).reshape(1, 3, 1, 1)
        low = np.asarray(mirror_extend_low_side(arr, axis=1, parity=1, on_plane=True)).ravel()
        assert np.allclose(low, [30.0, 30.0, 20.0])


class TestProjectOntoParity:
    def _mode(self, values):
        return jnp.asarray(values, dtype=jnp.float32).reshape(3, 1, 4, 1)

    def test_symmetric_field_is_untouched(self):
        # Ex is tangential to a y-wall and even under PMC; sampled on the plane (index 2 of 4), so a
        # compatible profile satisfies f[1] == f[3].
        field = self._mode([[0.0, 7.0, 9.0, 7.0], [0.0] * 4, [0.0] * 4])
        projected, residual = project_onto_parity(field, "E", {1: 1})
        assert residual < 1e-6
        assert jnp.allclose(projected[0, 0, 1:, 0], field[0, 0, 1:, 0])

    def test_incompatible_field_is_annihilated(self):
        # The same profile against a PEC wall, where Ex is odd: nothing survives except the unpaired
        # outermost sample, so the residual is close to one.
        field = self._mode([[0.0, 7.0, 9.0, 7.0], [0.0] * 4, [0.0] * 4])
        projected, residual = project_onto_parity(field, "E", {1: -1})
        assert residual > 0.9
        assert jnp.allclose(projected[0, 0, 2, 0], 0.0)  # on-plane odd sample must vanish

    def test_odd_on_plane_component_is_zeroed_on_the_plane(self):
        field = self._mode([[1.0, 2.0, 3.0, 4.0], [0.0] * 4, [0.0] * 4])
        projected, _ = project_onto_parity(field, "E", {1: -1})
        assert jnp.allclose(projected[0, 0, 2, 0], 0.0)

    def test_zero_field_has_zero_residual(self):
        field = jnp.zeros((3, 1, 4, 1), dtype=jnp.float32)
        _projected, residual = project_onto_parity(field, "H", {1: 1})
        assert residual == 0.0
