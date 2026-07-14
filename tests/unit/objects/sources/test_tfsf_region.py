"""Unit / integration tests for objects/sources/tfsf_region.py.

Covers face enumeration, per-face descriptors populated by ``apply``, and the
cross-object placement validation (periodic/wrap axes must have periodic
boundaries on both sides and span the full domain; the propagation axis can
never be periodic).
"""

import jax
import jax.numpy as jnp
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.tfsf_region import TFSFPlaneSourceRegion


@pytest.fixture
def micro_config():
    return SimulationConfig(
        time=20e-15,
        grid=UniformGrid(spacing=100e-9),
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=None,
    )


def _make_region(propagation_axis=0, periodic_axes=(), grid_shape=(6, 6, 6)):
    return TFSFPlaneSourceRegion(
        name="src",
        partial_grid_shape=grid_shape,
        propagation_axis=propagation_axis,
        direction="+",
        wave_character=WaveCharacter(wavelength=1.0e-6),
        fixed_E_polarization_vector=(0, 1, 0),
        periodic_axes=periodic_axes,
    )


class TestActiveFaces:
    def test_default_box_has_six_faces(self):
        region = _make_region(propagation_axis=0)
        faces = region._active_faces()
        assert len(faces) == 6
        # both propagation caps always present
        assert (0, "-") in faces and (0, "+") in faces
        # both transverse axes confined
        assert (1, "-") in faces and (1, "+") in faces
        assert (2, "-") in faces and (2, "+") in faces

    def test_one_periodic_axis_drops_two_faces(self):
        region = _make_region(propagation_axis=0, periodic_axes=(2,))
        faces = region._active_faces()
        assert len(faces) == 4
        assert all(normal != 2 for normal, _ in faces)
        assert (0, "-") in faces and (0, "+") in faces
        assert (1, "-") in faces and (1, "+") in faces

    def test_two_periodic_axes_leaves_only_caps(self):
        region = _make_region(propagation_axis=0, periodic_axes=(1, 2))
        faces = region._active_faces()
        assert faces == [(0, "-"), (0, "+")]

    def test_propagation_axis_selects_transverse(self):
        region = _make_region(propagation_axis=2)
        assert region.horizontal_axis == 0
        assert region.vertical_axis == 1
        faces = region._active_faces()
        assert (2, "-") in faces and (2, "+") in faces


def _place(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    return fdtdx.place_objects(object_list=objects, config=config, constraints=constraints, key=key)


def _pml_only(volume):
    cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="pml", thickness=2)
    return fdtdx.boundary_objects_from_config(cfg, volume)


class TestApplyPopulatesFaces:
    def test_faces_and_signs(self, micro_config):
        volume = fdtdx.SimulationVolume(partial_real_shape=(1.2e-6, 1.2e-6, 1.2e-6))
        region = _make_region(propagation_axis=0, grid_shape=(6, 6, 6))
        bd, bc = _pml_only(volume)
        constraints = [region.place_at_center(volume), *bc]
        objects, arrays, params, _config, key = _place([volume, region, *bd.values()], constraints, micro_config)
        arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
        placed = next(o for o in objects.objects if isinstance(o, TFSFPlaneSourceRegion))

        assert len(placed._face_normal_axes) == 6
        assert len(placed._face_incident_E) == 6
        assert len(placed._face_incident_H) == 6
        # min faces +1, max faces -1
        for i, (normal, sign) in enumerate(zip(placed._face_normal_axes, placed._face_signs)):
            lo, hi = placed._face_slice_tuples[i][normal]
            assert hi - lo == 1  # face is one cell thick along its normal
            box_lo, box_hi = placed.grid_slice_tuple[normal]
            if lo == box_lo:
                assert sign == 1
            else:
                assert lo == box_hi - 1 and sign == -1
        # no dispersion -> no H filters
        assert all(f is None for f in placed._face_H_filter)


class TestValidatePlacement:
    def test_propagation_axis_periodic_raises(self, micro_config):
        volume = fdtdx.SimulationVolume(partial_real_shape=(1.2e-6, 1.2e-6, 1.2e-6))
        region = _make_region(propagation_axis=0, periodic_axes=(0,), grid_shape=(6, 6, 6))
        bd, bc = _pml_only(volume)
        constraints = [region.place_at_center(volume), *bc]
        with pytest.raises(ValueError, match="propagation axis"):
            _place([volume, region, *bd.values()], constraints, micro_config)

    def test_periodic_axis_without_periodic_boundary_raises(self, micro_config):
        volume = fdtdx.SimulationVolume(partial_real_shape=(1.2e-6, 1.2e-6, 1.2e-6))
        # axis 1 marked periodic but boundaries are PML -> error. Box spans axis 1.
        region = _make_region(propagation_axis=0, periodic_axes=(1,), grid_shape=(6, None, 6))
        bd, bc = _pml_only(volume)
        constraints = [
            region.place_at_center(volume),
            region.same_size(volume, axes=(1,)),
            *bc,
        ]
        with pytest.raises(ValueError, match=r"periodic \(Bloch\) boundaries on both"):
            _place([volume, region, *bd.values()], constraints, micro_config)

    def test_periodic_axis_not_spanning_domain_raises(self, micro_config):
        volume = fdtdx.SimulationVolume(partial_real_shape=(1.2e-6, 1.2e-6, 1.2e-6))
        # axis 1 periodic boundaries present, but box does NOT span axis 1 -> error.
        region = _make_region(propagation_axis=0, periodic_axes=(1,), grid_shape=(6, 4, 6))
        cfg = fdtdx.BoundaryConfig.from_uniform_bound(
            boundary_type="pml", thickness=2, override_types={"min_y": "periodic", "max_y": "periodic"}
        )
        bd, bc = fdtdx.boundary_objects_from_config(cfg, volume)
        constraints = [region.place_at_center(volume), *bc]
        with pytest.raises(ValueError, match="span the full simulation domain"):
            _place([volume, region, *bd.values()], constraints, micro_config)

    def test_periodic_axis_phase_shifted_bloch_raises(self, micro_config):
        volume = fdtdx.SimulationVolume(partial_real_shape=(1.2e-6, 1.2e-6, 1.2e-6))
        region = _make_region(propagation_axis=0, periodic_axes=(1,), grid_shape=(6, None, 6))
        cfg = fdtdx.BoundaryConfig.from_uniform_bound(
            boundary_type="pml", thickness=2, override_types={"min_y": "bloch", "max_y": "bloch"}
        )
        cfg = cfg.aset("bloch_vector", (0.0, 0.5, 0.0))
        bd, bc = fdtdx.boundary_objects_from_config(cfg, volume)
        constraints = [region.place_at_center(volume), region.same_size(volume, axes=(1,)), *bc]
        with pytest.raises(ValueError, match="phase-shifted Bloch"):
            _place([volume, region, *bd.values()], constraints, micro_config)

    def test_valid_periodic_placement_succeeds(self, micro_config):
        volume = fdtdx.SimulationVolume(partial_real_shape=(1.2e-6, 1.2e-6, 1.2e-6))
        region = _make_region(propagation_axis=0, periodic_axes=(1, 2), grid_shape=(6, None, None))
        cfg = fdtdx.BoundaryConfig.from_uniform_bound(
            boundary_type="pml",
            thickness=2,
            override_types={f: "periodic" for f in ("min_y", "max_y", "min_z", "max_z")},
        )
        bd, bc = fdtdx.boundary_objects_from_config(cfg, volume)
        constraints = [
            region.place_at_center(volume),
            region.same_size(volume, axes=(1, 2)),
            *bc,
        ]
        # Should not raise.
        objects, _arrays, _params, _config, _key = _place([volume, region, *bd.values()], constraints, micro_config)
        placed = next(o for o in objects.objects if isinstance(o, TFSFPlaneSourceRegion))
        assert placed._active_faces() == [(0, "-"), (0, "+")]


@pytest.fixture
def placed_vacuum_region(micro_config):
    """A fully-confined vacuum box, placed and applied, plus the domain shape."""
    volume = fdtdx.SimulationVolume(partial_real_shape=(1.2e-6, 1.2e-6, 1.2e-6))
    region = _make_region(propagation_axis=0, grid_shape=(6, 6, 6))
    bd, bc = _pml_only(volume)
    constraints = [region.place_at_center(volume), *bc]
    objects, arrays, params, config, key = _place([volume, region, *bd.values()], constraints, micro_config)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    placed = next(o for o in objects.objects if isinstance(o, TFSFPlaneSourceRegion))
    return placed, config.grid.shape


class TestInjectionBranches:
    def test_update_E_fully_anisotropic_runs(self, placed_vacuum_region):
        region, shape = placed_vacuum_region
        E = jnp.zeros((3, *shape), dtype=jnp.float32)
        inv_eps9 = jnp.ones((9, *shape), dtype=jnp.float32)
        out = region.update_E(E, inv_eps9, 1.0, jnp.asarray(5), inverse=False)
        assert out.shape == E.shape
        assert jnp.all(jnp.isfinite(out))
        assert not jnp.allclose(out, E)  # something was injected

    def test_update_H_fully_anisotropic_runs(self, placed_vacuum_region):
        region, shape = placed_vacuum_region
        H = jnp.zeros((3, *shape), dtype=jnp.float32)
        inv_mu9 = jnp.ones((9, *shape), dtype=jnp.float32)
        out = region.update_H(H, jnp.ones((1, *shape)), inv_mu9, jnp.asarray(5), inverse=False)
        assert out.shape == H.shape
        assert jnp.all(jnp.isfinite(out))
        assert not jnp.allclose(out, H)

    def test_forward_then_inverse_is_identity(self, placed_vacuum_region):
        """The reverse (adjoint) update must exactly undo the forward injection."""
        region, shape = placed_vacuum_region
        inv_eps = jnp.ones((1, *shape), dtype=jnp.float32)
        E0 = jax.random.normal(jax.random.PRNGKey(1), (3, *shape), dtype=jnp.float32)
        ts = jnp.asarray(7)
        E_fwd = region.update_E(E0, inv_eps, 1.0, ts, inverse=False)
        E_back = region.update_E(E_fwd, inv_eps, 1.0, ts, inverse=True)
        assert jnp.allclose(E0, E_back, atol=1e-6)

        H0 = jax.random.normal(jax.random.PRNGKey(2), (3, *shape), dtype=jnp.float32)
        H_fwd = region.update_H(H0, inv_eps, 1.0, ts, inverse=False)
        H_back = region.update_H(H_fwd, inv_eps, 1.0, ts, inverse=True)
        assert jnp.allclose(H0, H_back, atol=1e-6)
