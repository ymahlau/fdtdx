"""Unit tests for the low-level mirror-symmetry primitives."""

import jax
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
    mirror_pairs_on_plane,
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


class TestMirrorPairsOnPlane:
    """The mirror index map depends on the wall type, not only on the Yee offsets."""

    def test_electric_plane_uses_the_yee_offsets(self):
        for field_type in ("E", "H"):
            for axis in range(3):
                for component in range(3):
                    assert mirror_pairs_on_plane(field_type, component, axis, -1) == component_sits_on_plane(
                        field_type, component, axis
                    )

    def test_magnetic_plane_is_always_a_plain_flip(self):
        # A magnetic plane sits half a cell below the reduced domain, so nothing is sampled on it.
        for field_type in ("E", "H"):
            for axis in range(3):
                for component in range(3):
                    assert mirror_pairs_on_plane(field_type, component, axis, 1) is False


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

    def test_symmetric_field_is_untouched_across_a_magnetic_plane(self):
        # Ex is tangential to a y-wall and even under PMC. A magnetic plane sits half a cell below the
        # reduced domain, so its mirror map is the plain flip and a compatible profile is the one that
        # is symmetric about the array centre: f[0] == f[3], f[1] == f[2].
        field = self._mode([[7.0, 9.0, 9.0, 7.0], [0.0] * 4, [0.0] * 4])
        projected, residual = project_onto_parity(field, "E", {1: 1})
        assert residual < 1e-6
        assert jnp.allclose(projected[0, 0, :, 0], field[0, 0, :, 0])

    def test_symmetric_field_is_untouched_across_an_electric_plane(self):
        # Ez is tangential to a y-wall and odd under PEC, and it is sampled *on* an electric plane
        # (index 2 of 4), so a compatible profile vanishes there and satisfies f[1] == -f[3].
        field = self._mode([[0.0] * 4, [0.0] * 4, [0.0, -7.0, 0.0, 7.0]])
        projected, residual = project_onto_parity(field, "E", {1: -1})
        assert residual < 1e-6
        assert jnp.allclose(projected[2, 0, 1:, 0], field[2, 0, 1:, 0])

    def test_magnetic_plane_map_differs_from_the_electric_one(self):
        # The same profile cannot be compatible with both: across an electric plane the pairing is
        # m±j about the plane row, across a magnetic plane it is the plain flip.
        field = self._mode([[0.0, 7.0, 9.0, 7.0], [0.0] * 4, [0.0] * 4])
        _projected, residual = project_onto_parity(field, "E", {1: 1})
        assert residual > 0.1, "an m±j-symmetric profile is not plain-flip symmetric"

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

    def test_residual_is_a_jax_scalar_and_survives_tracing(self):
        # A mode source or mode-overlap detector that overlaps a Device solves its mode inside
        # apply_params, which callers trace (jax.jit around an optimization step). Concretizing the
        # residual there would raise ConcretizationTypeError, so it must stay a JAX scalar.
        field = self._mode([[1.0, 2.0, 3.0, 4.0], [0.0] * 4, [0.0] * 4])
        projected, residual = project_onto_parity(field, "E", {1: -1})
        assert isinstance(residual, jax.Array) and residual.shape == ()

        def traced(f):
            return project_onto_parity(f, "E", {1: -1})

        jit_projected, jit_residual = jax.jit(traced)(field)
        assert jnp.allclose(jit_projected, projected)
        assert float(jit_residual) == pytest.approx(float(residual), abs=1e-6)
