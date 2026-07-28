"""Unit tests for the closed-surface Poynting flux detector.

Covers the pure reduction primitive ``net_poynting_flux_through_box`` (constant
field -> zero net flux, linear field -> analytic divergence, per-cell area
weighting, thin-axis cancellation, axis selection) and the
``ClosedSurfacePoyntingFluxDetector`` end to end -- including a genuinely
non-uniform rectilinear grid where cell areas differ across a single face.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from fdtdx.core.physics.metrics import net_poynting_flux_through_box
from fdtdx.objects.detectors.poynting_flux import ClosedSurfacePoyntingFluxDetector, PoyntingFluxDetector

_KEY = jax.random.PRNGKey(0)


def _unit_area_weights(shape):
    """Per-axis area weights of 1 per cell, broadcastable to each Poynting component."""
    nx, ny, nz = shape
    return (
        jnp.ones((1, ny, nz)),
        jnp.ones((nx, 1, nz)),
        jnp.ones((nx, ny, 1)),
    )


def _x_ramp(shape):
    """Field varying as the x cell index, broadcast over the box."""
    nx = shape[0]
    return jnp.broadcast_to(jnp.arange(nx, dtype=jnp.float32).reshape(nx, 1, 1), shape)


class TestNetFluxPrimitive:
    def test_constant_field_is_divergence_free(self):
        shape = (4, 5, 6)
        pf = jnp.ones((3, *shape), dtype=jnp.float32)
        net = net_poynting_flux_through_box(pf, (0, 1, 2), _unit_area_weights(shape))
        assert abs(float(net)) < 1e-5

    def test_linear_field_matches_divergence(self):
        shape = (4, 3, 2)
        pf = jnp.stack([_x_ramp(shape), jnp.zeros(shape), jnp.zeros(shape)])
        net = net_poynting_flux_through_box(pf, (0, 1, 2), _unit_area_weights(shape))
        # Only the x-faces differ: max face S_x = (nx-1), min face S_x = 0.
        expected = (shape[0] - 1) * shape[1] * shape[2]
        assert abs(float(net) - expected) < 1e-4

    def test_per_cell_area_weights_are_summed(self):
        shape = (3, 2, 2)
        pf = jnp.stack([_x_ramp(shape), jnp.zeros(shape), jnp.zeros(shape)])
        # Non-uniform per-(y,z)-cell areas on the x-face.
        area_x = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32).reshape(1, 2, 2)
        aw = (area_x, jnp.ones((3, 1, 2)), jnp.ones((3, 2, 1)))
        net = net_poynting_flux_through_box(pf, (0,), aw)
        expected = (shape[0] - 1) * float(area_x.sum())
        assert abs(float(net) - expected) < 1e-4

    def test_thin_axis_faces_cancel(self):
        shape = (4, 4, 1)
        pf = jnp.stack([_x_ramp(shape), jnp.ones(shape), jnp.ones(shape)])
        aw = _unit_area_weights(shape)
        with_z = net_poynting_flux_through_box(pf, (0, 1, 2), aw)
        without_z = net_poynting_flux_through_box(pf, (0, 1), aw)
        assert abs(float(with_z) - float(without_z)) < 1e-6

    def test_axis_selection(self):
        shape = (3, 3, 3)
        # Divergence on both x and y.
        pf = jnp.stack([_x_ramp(shape), jnp.swapaxes(_x_ramp(shape), 0, 1), jnp.zeros(shape)])
        aw = _unit_area_weights(shape)
        only_x = net_poynting_flux_through_box(pf, (0,), aw)
        both = net_poynting_flux_through_box(pf, (0, 1), aw)
        assert abs(float(only_x) - (shape[0] - 1) * shape[1] * shape[2]) < 1e-4
        assert float(both) > float(only_x)  # the y faces add positive divergence


def _place(config, slice_tuple, **kwargs):
    det = ClosedSurfacePoyntingFluxDetector(name="box", **kwargs)
    return det.place_on_grid(grid_slice_tuple=slice_tuple, config=config, key=_KEY)


def _fields_for_Sx(ramp, shape):
    """E/H such that the Poynting vector is S = (ramp, 0, 0)."""
    E = jnp.stack([jnp.zeros(shape), ramp, jnp.zeros(shape)])  # E_y
    H = jnp.stack([jnp.zeros(shape), jnp.zeros(shape), jnp.ones(shape)])  # H_z
    return E, H


class TestClosedSurfaceDetector:
    def test_uniform_grid_constant_field_net_zero(self):
        config = SimulationConfig(
            time=20e-15, grid=UniformGrid(spacing=1e-7), backend="cpu", dtype=jnp.float32, gradient_config=None
        )
        placed = _place(config, ((0, 4), (0, 4), (0, 4)))
        shape = (4, 4, 4)
        E, H = _fields_for_Sx(jnp.ones(shape), shape)  # constant S_x
        state = placed.init_state()
        new_state = placed.update(jnp.asarray(0), E, H, state, jnp.ones((1, *shape)), 1.0)
        assert abs(float(new_state["poynting_flux"][0, 0])) < 1e-5

    def test_nonuniform_grid_per_cell_area_end_to_end(self):
        # Deliberately unequal cell widths (at a physical nm scale so the sim has
        # time steps), so areas differ across a single face.
        s = 1e-7
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0, 10.0]) * s,  # nx=4
            y_edges=jnp.asarray([0.0, 1.0, 3.0, 4.0]) * s,  # ny=3, widths 1,2,1
            z_edges=jnp.asarray([0.0, 2.0, 3.0]) * s,  # nz=2, widths 2,1
        )
        config = SimulationConfig(time=20e-15, grid=grid, backend="cpu", dtype=jnp.float32, gradient_config=None)
        shape = (4, 3, 2)
        placed = _place(config, ((0, 4), (0, 3), (0, 2)))
        E, H = _fields_for_Sx(_x_ramp(shape), shape)  # S_x = x cell index
        state = placed.init_state()
        new_state = placed.update(jnp.asarray(0), E, H, state, jnp.ones((1, *shape)), 1.0)
        net = float(new_state["poynting_flux"][0, 0])
        # Net = (S_x@max - S_x@min) * sum(dy_j * dz_k) = (nx-1) * A_x, using per-cell areas.
        dy = jnp.diff(grid.y_edges)
        dz = jnp.diff(grid.z_edges)
        area_x = float((dy[:, None] * dz[None, :]).sum())
        assert net == pytest.approx((shape[0] - 1) * area_x, rel=1e-4)

    def test_orientation_inward_negates(self):
        config = SimulationConfig(
            time=20e-15, grid=UniformGrid(spacing=1e-7), backend="cpu", dtype=jnp.float32, gradient_config=None
        )
        shape = (4, 3, 3)
        E, H = _fields_for_Sx(_x_ramp(shape), shape)
        out = _place(config, ((0, 4), (0, 3), (0, 3)), orientation="outward")
        inw = _place(config, ((0, 4), (0, 3), (0, 3)), orientation="inward")
        net_out = float(
            out.update(jnp.asarray(0), E, H, out.init_state(), jnp.ones((1, *shape)), 1.0)["poynting_flux"][0, 0]
        )
        net_in = float(
            inw.update(jnp.asarray(0), E, H, inw.init_state(), jnp.ones((1, *shape)), 1.0)["poynting_flux"][0, 0]
        )
        # (nx-1) * ny * nz * spacing**2 is the outward net for S_x = x cell index.
        expected = (shape[0] - 1) * shape[1] * shape[2] * (1e-7) ** 2
        assert net_out == pytest.approx(expected, rel=1e-4)
        assert net_in == pytest.approx(-net_out, rel=1e-5)

    def test_matches_poynting_flux_detector_convention(self):
        """Same normalization as PoyntingFluxDetector: same Poynting vector and physical
        face area, no extra scaling. A one-sided field makes the box's single active face
        equal a plane PoyntingFluxDetector reading of that face."""
        config = SimulationConfig(
            time=20e-15, grid=UniformGrid(spacing=1e-7), backend="cpu", dtype=jnp.float32, gradient_config=None
        )
        ny, nz = 3, 4
        c = 2.0
        # Box of depth 2 along x: S_x = 0 on the min layer, S_x = c on the max layer.
        ey = jnp.stack([jnp.zeros((ny, nz)), c * jnp.ones((ny, nz))], axis=0)  # E_y per x-layer
        zeros2 = jnp.zeros((2, ny, nz))
        E = jnp.stack([zeros2, ey, zeros2])
        H = jnp.stack([zeros2, zeros2, jnp.ones((2, ny, nz))])  # H_z = 1 -> S_x = E_y

        box = _place(config, ((0, 2), (0, ny), (0, nz)), axes=(0,))
        net = float(
            box.update(jnp.asarray(0), E, H, box.init_state(), jnp.ones((1, 2, ny, nz)), 1.0)["poynting_flux"][0, 0]
        )

        # Plane detector reading the max-x face (S_x = c there).
        plane = PoyntingFluxDetector(name="plane", direction="+", fixed_propagation_axis=0).place_on_grid(
            grid_slice_tuple=((1, 2), (0, ny), (0, nz)), config=config, key=_KEY
        )
        p_plane = float(
            plane.update(jnp.asarray(0), E[:, 1:2], H[:, 1:2], plane.init_state(), jnp.ones((1, 1, ny, nz)), 1.0)[
                "poynting_flux"
            ][0, 0]
        )
        # min face has S_x = 0, so the box net equals the plane detector's max-face reading.
        assert net == pytest.approx(p_plane, rel=1e-6)

    def test_default_active_axes_skips_thin_axis(self):
        config = SimulationConfig(
            time=20e-15, grid=UniformGrid(spacing=1e-7), backend="cpu", dtype=jnp.float32, gradient_config=None
        )
        placed = _place(config, ((0, 5), (0, 5), (0, 1)))  # thin z
        assert placed._resolve_active_axes() == (0, 1)

    def test_explicit_axes(self):
        config = SimulationConfig(
            time=20e-15, grid=UniformGrid(spacing=1e-7), backend="cpu", dtype=jnp.float32, gradient_config=None
        )
        placed = _place(config, ((0, 4), (0, 4), (0, 4)), axes=(0,))
        assert placed._resolve_active_axes() == (0,)

    def test_state_shape_is_scalar(self):
        config = SimulationConfig(
            time=20e-15, grid=UniformGrid(spacing=1e-7), backend="cpu", dtype=jnp.float32, gradient_config=None
        )
        placed = _place(config, ((0, 4), (0, 4), (0, 4)))
        state = placed.init_state()
        assert state["poynting_flux"].shape[1:] == (1,)

    def test_invalid_orientation_raises(self):
        config = SimulationConfig(
            time=20e-15, grid=UniformGrid(spacing=1e-7), backend="cpu", dtype=jnp.float32, gradient_config=None
        )
        with pytest.raises(ValueError, match="orientation"):
            _place(config, ((0, 4), (0, 4), (0, 4)), orientation="sideways")

    def test_invalid_axes_raises(self):
        config = SimulationConfig(
            time=20e-15, grid=UniformGrid(spacing=1e-7), backend="cpu", dtype=jnp.float32, gradient_config=None
        )
        with pytest.raises(ValueError, match="axes"):
            _place(config, ((0, 4), (0, 4), (0, 4)), axes=(0, 3))
