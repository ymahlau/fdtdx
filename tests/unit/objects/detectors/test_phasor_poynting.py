"""Unit tests for the frequency-domain (phasor-based) Poynting flux detectors.

Covers ``PhasorPoyntingFluxDetector`` (single plane) and
``ClosedSurfacePhasorPoyntingFluxDetector`` (hollow closed surface). The
time-averaged flux ``<S> = 1/2 Re(E(w) x H*(w))`` is bilinear in the fields, so
these detectors accumulate phasors during the run and reduce afterwards via
``compute_poynting_flux`` / ``compute_net_flux``. Most tests construct a known
phasor *state* directly (independent of the DFT accumulation) so the reduction
math is checked exactly, plus a full-box parity check that guards the hollow
face-accounting reimplementation and a hollow-storage check.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from fdtdx.core.physics.metrics import net_poynting_flux_through_box
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.detectors.poynting_flux import (
    ClosedSurfacePhasorPoyntingFluxDetector,
    PhasorPoyntingFluxDetector,
    _resolve_face_area_weights,
)

_KEY = jax.random.PRNGKey(0)
_SPACING = 1e-7
_WC = [WaveCharacter(wavelength=1.55e-6)]
_WC3 = [WaveCharacter(wavelength=1.0e-6), WaveCharacter(wavelength=1.55e-6), WaveCharacter(frequency=3e14)]

# component order in the phasor stack: (Ex, Ey, Ez, Hx, Hy, Hz)
_EY, _HZ = 1, 5


def _uniform_config():
    return SimulationConfig(
        time=20e-15, grid=UniformGrid(spacing=_SPACING), backend="cpu", dtype=jnp.float32, gradient_config=None
    )


# --------------------------------------------------------------------------- #
# Single-plane detector
# --------------------------------------------------------------------------- #


def _place_plane(config, slice_tuple, **kwargs):
    det = PhasorPoyntingFluxDetector(
        name="plane", direction=kwargs.pop("direction", "+"), wave_characters=_WC, **kwargs
    )
    return det.place_on_grid(grid_slice_tuple=slice_tuple, config=config, key=_KEY)


def _plane_state_ey_hz(det, ey, hz):
    """A phasor state with constant E_y and H_z over the plane (all frequencies)."""
    ph = det.init_state()["phasor"]  # (1, num_freqs, 6, *plane)
    ph = ph.at[:, :, _EY].set(ey)
    ph = ph.at[:, :, _HZ].set(hz)
    return {"phasor": ph}


class TestPhasorPlaneDetector:
    def test_state_shape(self):
        det = _place_plane(_uniform_config(), ((0, 1), (0, 4), (0, 5)))
        state = det.init_state()
        # 1 latent time step, 1 frequency, 6 components, plane 1x4x5
        assert state["phasor"].shape == (1, 1, 6, 1, 4, 5)
        assert state["phasor"].dtype == jnp.complex64

    def test_constant_cw_flux(self):
        ny, nz = 4, 5
        det = _place_plane(_uniform_config(), ((0, 1), (0, ny), (0, nz)))
        ey, hz = 2.0, 1.0
        flux = det.compute_poynting_flux(_plane_state_ey_hz(det, ey, hz))
        # continuous scaling applies the 1/2 time-average factor
        expected = 0.5 * (ey * hz) * _SPACING**2 * ny * nz
        assert flux.shape == (1,)
        assert float(flux[0]) == pytest.approx(expected, rel=1e-5)

    def test_direction_negative_flips_sign(self):
        ny, nz = 4, 5
        cfg = _uniform_config()
        pos = _place_plane(cfg, ((0, 1), (0, ny), (0, nz)), direction="+")
        neg = _place_plane(cfg, ((0, 1), (0, ny), (0, nz)), direction="-")
        f_pos = float(pos.compute_poynting_flux(_plane_state_ey_hz(pos, 2.0, 1.0))[0])
        f_neg = float(neg.compute_poynting_flux(_plane_state_ey_hz(neg, 2.0, 1.0))[0])
        assert f_neg == pytest.approx(-f_pos, rel=1e-6)

    def test_multi_wavelength_shape(self):
        det = PhasorPoyntingFluxDetector(name="plane", direction="+", wave_characters=_WC3)
        det = det.place_on_grid(grid_slice_tuple=((0, 1), (0, 4), (0, 5)), config=_uniform_config(), key=_KEY)
        ph = det.init_state()["phasor"]
        ph = ph.at[:, :, _EY].set(2.0).at[:, :, _HZ].set(1.0)
        flux = det.compute_poynting_flux({"phasor": ph})
        assert flux.shape == (3,)
        # identical phasors across frequencies -> identical flux
        assert jnp.allclose(flux, flux[0])

    def test_pulse_mode_skips_half_factor(self):
        ny, nz = 4, 5
        det = _place_plane(_uniform_config(), ((0, 1), (0, ny), (0, nz)), scaling_mode="pulse")
        flux = det.compute_poynting_flux(_plane_state_ey_hz(det, 2.0, 1.0))
        # pulse mode returns the raw bilinear product without the 1/2 average factor
        expected = (2.0 * 1.0) * _SPACING**2 * ny * nz
        assert float(flux[0]) == pytest.approx(expected, rel=1e-5)

    def test_nonuniform_grid_area_weighting(self):
        s = _SPACING
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0]) * s,  # thin x (propagation axis)
            y_edges=jnp.asarray([0.0, 1.0, 3.0, 4.0]) * s,  # widths 1, 2, 1
            z_edges=jnp.asarray([0.0, 2.0, 3.0]) * s,  # widths 2, 1
        )
        config = SimulationConfig(time=20e-15, grid=grid, backend="cpu", dtype=jnp.float32, gradient_config=None)
        det = _place_plane(config, ((0, 1), (0, 3), (0, 2)))
        ey, hz = 2.0, 1.0
        flux = det.compute_poynting_flux(_plane_state_ey_hz(det, ey, hz))
        dy = jnp.diff(grid.y_edges)
        dz = jnp.diff(grid.z_edges)
        area = float((dy[:, None] * dz[None, :]).sum())  # exact per-cell transverse area sum
        assert float(flux[0]) == pytest.approx(0.5 * ey * hz * area, rel=1e-5)


# --------------------------------------------------------------------------- #
# Closed-surface (hollow) detector
# --------------------------------------------------------------------------- #


def _place_box(config, slice_tuple, **kwargs):
    det = ClosedSurfacePhasorPoyntingFluxDetector(name="box", wave_characters=_WC, **kwargs)
    return det.place_on_grid(grid_slice_tuple=slice_tuple, config=config, key=_KEY)


def _hollow_state_from_full(box, full):
    """Build the per-face detector state by slicing a full-box phasor array.

    Args:
        full: complex phasors of shape ``(num_freqs, 6, nx, ny, nz)``.
    """
    state = box.init_state()
    for a in box._resolve_active_axes():
        for side in ("min", "max"):
            sl = [slice(None)] * full.ndim
            sl[a + 2] = slice(0, 1) if side == "min" else slice(-1, None)
            state[f"phasor_axis{a}_{side}"] = full[tuple(sl)][None, ...]
    return state


def _ref_net_flux(full, config, box, active_axes):
    """Reference net flux via the tested full-box primitive on the same phasors."""
    E, H = full[:, :3], full[:, 3:]
    pv = 0.5 * jnp.cross(E, jnp.conj(H), axisa=1, axisb=1, axisc=1).real  # (num_freqs, 3, nx, ny, nz)
    aw = tuple(_resolve_face_area_weights(config, box.grid_slice_tuple, a, jnp.float32) for a in range(3))
    return jnp.stack([net_poynting_flux_through_box(pv[f], active_axes, aw) for f in range(full.shape[0])])


class TestClosedSurfacePhasorDetector:
    def test_hollow_state_stores_only_faces(self):
        box = _place_box(_uniform_config(), ((0, 6), (0, 6), (0, 6)))
        state = box.init_state()
        # exactly two keys per active axis, and nothing that spans the full box interior
        assert set(state.keys()) == {f"phasor_axis{a}_{s}" for a in (0, 1, 2) for s in ("min", "max")}
        for key, arr in state.items():
            spatial = arr.shape[3:]  # (nx-ish, ny-ish, nz-ish)
            # a genuine face has exactly one collapsed (size-one) spatial dimension
            assert sum(d == 1 for d in spatial) == 1, f"{key} is not a hollow face: {arr.shape}"

    def test_default_active_axes_skips_thin_axis(self):
        box = _place_box(_uniform_config(), ((0, 5), (0, 5), (0, 1)))  # thin z
        assert box._resolve_active_axes() == (0, 1)
        assert "phasor_axis2_min" not in box.init_state()

    def test_explicit_axes_limits_stored_faces(self):
        box = _place_box(_uniform_config(), ((0, 4), (0, 4), (0, 4)), axes=(0,))
        assert set(box.init_state().keys()) == {"phasor_axis0_min", "phasor_axis0_max"}

    def test_constant_field_net_zero(self):
        box = _place_box(_uniform_config(), ((0, 4), (0, 4), (0, 4)))
        shape = (1, 6, 4, 4, 4)
        full = jnp.zeros(shape, dtype=jnp.complex64)
        full = full.at[:, _EY].set(2.0).at[:, _HZ].set(1.0)  # constant S_x, zero elsewhere
        net = box.compute_net_flux(_hollow_state_from_full(box, full))
        assert abs(float(net[0])) < 1e-18

    def test_one_sided_matches_plane_detector(self):
        cfg = _uniform_config()
        ny, nz = 3, 4
        # depth-2 box along x: S_x = 0 on the min layer, S_x = c on the max layer.
        c = 2.0
        full = jnp.zeros((1, 6, 2, ny, nz), dtype=jnp.complex64)
        full = full.at[:, _EY, 1].set(c).at[:, _HZ].set(1.0)  # E_y only on the max-x layer, H_z = 1 everywhere
        box = _place_box(cfg, ((0, 2), (0, ny), (0, nz)), axes=(0,))
        net = float(box.compute_net_flux(_hollow_state_from_full(box, full))[0])

        plane = _place_plane(cfg, ((1, 2), (0, ny), (0, nz)), fixed_propagation_axis=0)
        ph = plane.init_state()["phasor"]
        ph = ph.at[:, :, _EY].set(c).at[:, :, _HZ].set(1.0)
        p_plane = float(plane.compute_poynting_flux({"phasor": ph})[0])
        assert net == pytest.approx(p_plane, rel=1e-6)

    def test_orientation_inward_negates(self):
        cfg = _uniform_config()
        full = jnp.zeros((1, 6, 4, 4, 4), dtype=jnp.complex64)
        # a divergent field: S_x grows along x -> nonzero outward net
        ramp = jnp.arange(4, dtype=jnp.float32).reshape(1, 4, 1, 1)
        full = full.at[:, _EY].set(ramp).at[:, _HZ].set(1.0)
        out = _place_box(cfg, ((0, 4), (0, 4), (0, 4)), orientation="outward")
        inw = _place_box(cfg, ((0, 4), (0, 4), (0, 4)), orientation="inward")
        net_out = float(out.compute_net_flux(_hollow_state_from_full(out, full))[0])
        net_in = float(inw.compute_net_flux(_hollow_state_from_full(inw, full))[0])
        assert abs(net_out) > 0
        assert net_in == pytest.approx(-net_out, rel=1e-6)

    def test_parity_with_full_box_uniform(self):
        cfg = _uniform_config()
        box = _place_box(cfg, ((0, 4), (0, 5), (0, 3)))
        active = box._resolve_active_axes()
        k1, k2 = jax.random.split(_KEY)
        full = jax.random.normal(k1, (1, 6, 4, 5, 3)) + 1j * jax.random.normal(k2, (1, 6, 4, 5, 3))
        full = full.astype(jnp.complex64)
        net = box.compute_net_flux(_hollow_state_from_full(box, full))
        ref = _ref_net_flux(full, cfg, box, active)
        assert jnp.allclose(net, ref, rtol=1e-4, atol=1e-6)

    def test_parity_with_full_box_nonuniform(self):
        s = _SPACING
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0, 10.0]) * s,  # nx=4
            y_edges=jnp.asarray([0.0, 1.0, 3.0, 4.0]) * s,  # ny=3
            z_edges=jnp.asarray([0.0, 2.0, 3.0]) * s,  # nz=2
        )
        cfg = SimulationConfig(time=20e-15, grid=grid, backend="cpu", dtype=jnp.float32, gradient_config=None)
        box = _place_box(cfg, ((0, 4), (0, 3), (0, 2)))
        active = box._resolve_active_axes()
        k1, k2 = jax.random.split(jax.random.PRNGKey(7))
        full = jax.random.normal(k1, (1, 6, 4, 3, 2)) + 1j * jax.random.normal(k2, (1, 6, 4, 3, 2))
        full = full.astype(jnp.complex64)
        net = box.compute_net_flux(_hollow_state_from_full(box, full))
        ref = _ref_net_flux(full, cfg, box, active)
        assert jnp.allclose(net, ref, rtol=1e-4, atol=1e-6)

    def test_update_accumulates_into_faces(self):
        box = _place_box(_uniform_config(), ((0, 4), (0, 4), (0, 4)))
        shape = (4, 4, 4)
        E = jnp.stack([jnp.zeros(shape), jnp.ones(shape), jnp.zeros(shape)])
        H = jnp.stack([jnp.zeros(shape), jnp.zeros(shape), jnp.ones(shape)])
        s1 = box.update(jnp.asarray(0), E, H, box.init_state(), jnp.ones((1, *shape)), 1.0)
        s2 = box.update(jnp.asarray(0), E, H, s1, jnp.ones((1, *shape)), 1.0)
        # two updates at the same time step exactly double every stored face
        for key in s1:
            assert jnp.allclose(s2[key], 2 * s1[key], atol=1e-6)

    def test_invalid_orientation_raises(self):
        with pytest.raises(ValueError, match="orientation"):
            _place_box(_uniform_config(), ((0, 4), (0, 4), (0, 4)), orientation="sideways")

    def test_invalid_axes_raises(self):
        with pytest.raises(ValueError, match="axes"):
            _place_box(_uniform_config(), ((0, 4), (0, 4), (0, 4)), axes=(0, 3))
