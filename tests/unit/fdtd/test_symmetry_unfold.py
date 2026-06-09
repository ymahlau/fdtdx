"""Unit tests for the mirror-symmetry parity table and array unfolding helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.fdtd.symmetry import (
    _poynting_parity,
    _unfold_one_detector,
    field_component_parity,
    unfold_array,
    unfold_fields,
    unfold_source_mode,
)


class _FakeModeSource:
    """Minimal stand-in exposing the attributes unfold_source_mode reads (no tidy3d/placement)."""

    def __init__(self, mode_E, mode_H, grid_shape, name="modesrc"):
        self._E = mode_E
        self._H = mode_H
        self.grid_shape = grid_shape
        self.name = name


class TestFieldComponentParity:
    """The PEC/PMC mirror parity truth table (see module docstring of fdtd/symmetry.py)."""

    def test_pec_table(self):
        # PEC (wall=-1): tangential E zeroed -> E normal even, E tangential odd; H is opposite.
        for axis in range(3):
            for comp in range(3):
                normal = comp == axis
                assert field_component_parity("E", comp, axis, -1) == (1 if normal else -1)
                assert field_component_parity("H", comp, axis, -1) == (-1 if normal else 1)

    def test_pmc_table(self):
        # PMC (wall=+1): tangential H zeroed -> H normal even, H tangential odd; E is opposite.
        for axis in range(3):
            for comp in range(3):
                normal = comp == axis
                assert field_component_parity("E", comp, axis, 1) == (-1 if normal else 1)
                assert field_component_parity("H", comp, axis, 1) == (1 if normal else -1)

    def test_matches_quarter_domain_reference(self):
        # Ground truth from tests/.../test_quarter_domain_symmetry.py:
        #   PEC at y -> Ey (normal) even;  PMC at z -> Ey (tangential) even.
        assert field_component_parity("E", 1, 1, -1) == 1  # Ey across PEC y-plane
        assert field_component_parity("E", 1, 2, 1) == 1  # Ey across PMC z-plane

    def test_invalid_wall(self):
        with pytest.raises(ValueError):
            field_component_parity("E", 0, 0, 0)


class TestUnfoldFields:
    """unfold_fields reconstructs a full (3, Nx, Ny, Nz) field with correct parity."""

    def _half(self, seed=0, nx=2, nh=4, nz=2):
        rng = np.random.default_rng(seed)
        return jnp.asarray(rng.standard_normal((3, nx, nh, nz)), dtype=jnp.float32)

    def test_pec_y_even_and_odd_components(self):
        half = self._half()
        symmetry = (0, -1, 0)  # PEC mirror normal to y
        full = unfold_fields(half, symmetry, "E")

        # shape doubles along y (array axis 2); kept upper half reproduces the input.
        assert full.shape == (3, 2, 8, 2)
        assert jnp.allclose(full[:, :, 4:, :], half)

        # Ey is normal to the PEC y-plane -> even about the center.
        assert jnp.allclose(full[1], jnp.flip(full[1], axis=1))
        # Ex, Ez are tangential -> odd about the center.
        assert jnp.allclose(full[0], -jnp.flip(full[0], axis=1))
        assert jnp.allclose(full[2], -jnp.flip(full[2], axis=1))

    def test_pmc_z_parities_for_H(self):
        half = self._half(seed=1)
        symmetry = (0, 0, 1)  # PMC mirror normal to z
        full = unfold_fields(half, symmetry, "H")

        assert full.shape == (3, 2, 4, 4)
        # After indexing a component, spatial axes are (x=0, y=1, z=2); mirror about z is axis 2.
        # Hz normal to PMC z-plane -> even; Hx, Hy tangential -> odd.
        assert jnp.allclose(full[2], jnp.flip(full[2], axis=2))
        assert jnp.allclose(full[0], -jnp.flip(full[0], axis=2))
        assert jnp.allclose(full[1], -jnp.flip(full[1], axis=2))

    def test_two_axis_shape(self):
        half = self._half(nx=3, nh=4, nz=5)
        full = unfold_fields(half, (0, -1, 1), "E")
        assert full.shape == (3, 3, 8, 10)

    def test_no_symmetry_raises(self):
        half = self._half()
        with pytest.raises(ValueError):
            unfold_fields(half, (0, 0, 0), "E")

    def test_invalid_field_type(self):
        half = self._half()
        with pytest.raises(ValueError):
            unfold_fields(half, (0, -1, 0), "B")  # type: ignore[arg-type]


class TestUnfoldArray:
    """Generic spatial mirror-concat used by detector unfolding."""

    def test_scalar_even_mirror(self):
        rng = np.random.default_rng(2)
        arr = jnp.asarray(rng.standard_normal((1, 5, 4, 6)), dtype=jnp.float32)  # (T, nx, ny, nz)
        out = unfold_array(arr, (0, -1, 0), spatial_axes=(1, 2, 3))
        assert out.shape == (1, 5, 8, 6)
        # parity +1 -> even about the center along y.
        assert jnp.allclose(out, jnp.flip(out, axis=2))

    def test_per_component_signs(self):
        arr = jnp.ones((1, 2, 1, 1), dtype=jnp.float32)  # (T, ncomp=2, ny=1, nz=1)
        signs = {1: jnp.asarray([1.0, -1.0]).reshape((1, 2, 1, 1))}
        out = unfold_array(arr, (0, -1, 0), spatial_axes=(0, 2, 3), signs=signs)
        # axis y is array-axis 2; component 0 even (+1), component 1 odd (-1).
        assert out.shape == (1, 2, 2, 1)
        assert jnp.allclose(out[0, 0], jnp.asarray([[1.0], [1.0]]))
        assert jnp.allclose(out[0, 1], jnp.asarray([[-1.0], [1.0]]))

    def test_no_symmetry_raises(self):
        arr = jnp.ones((1, 2, 2, 2), dtype=jnp.float32)
        with pytest.raises(ValueError):
            unfold_array(arr, (0, 0, 0), spatial_axes=(1, 2, 3))


class TestPerDetectorPostProcessing:
    """Per-detector unfolding done purely from the stored reduced output (no in-loop work)."""

    def test_reduce_volume_phasor_even_kept_odd_zeroed(self):
        # PEC mirror normal to y: Ey (normal) is even -> mean unchanged; Ex (tangential) odd -> 0.
        det = fdtdx.PhasorDetector(
            name="rv",
            wave_characters=(fdtdx.WaveCharacter(wavelength=1e-6),),
            components=("Ex", "Ey"),
            reduce_volume=True,
            plot=False,
        )
        state = {"phasor": jnp.asarray([[[3.0 + 1j, 5.0 - 2j]]], dtype=jnp.complex64)}  # (1, 1, 2)
        out = _unfold_one_detector(det, state, (0, -1, 0), count=1)["phasor"]
        assert jnp.allclose(out[0, 0, 0], 0.0)  # Ex tangential -> 0
        assert jnp.allclose(out[0, 0, 1], 5.0 - 2j)  # Ey normal -> unchanged

    def test_poynting_parity_values(self):
        # PEC normal to y: S_x even (+1), S_y odd (-1), S_z even (+1).
        assert _poynting_parity(0, 1, -1) == 1
        assert _poynting_parity(1, 1, -1) == -1
        assert _poynting_parity(2, 1, -1) == 1

    def test_poynting_keep_all_summed_scaled_per_component(self):
        det = fdtdx.PoyntingFluxDetector(
            name="pf",
            direction="+",
            reduce_volume=True,
            keep_all_components=True,
            fixed_propagation_axis=0,
            plot=False,
        )
        state = {"poynting_flux": jnp.asarray([[2.0, 3.0, 4.0]], dtype=jnp.float32)}  # (T=1, 3)
        out = _unfold_one_detector(det, state, (0, -1, 0), count=1)["poynting_flux"]
        # S_x even -> x2, S_y odd -> x0, S_z even -> x2.
        assert jnp.allclose(out[0], jnp.asarray([4.0, 0.0, 8.0]))

    def test_poynting_scalar_summed_doubles(self):
        det = fdtdx.PoyntingFluxDetector(
            name="pf2",
            direction="+",
            reduce_volume=True,
            keep_all_components=False,
            fixed_propagation_axis=0,
            plot=False,
        )
        state = {"poynting_flux": jnp.asarray([[7.0]], dtype=jnp.float32)}  # (T=1, 1)
        # Two symmetric transverse planes (y PEC, z PMC); S_x even on both -> x4.
        out = _unfold_one_detector(det, state, (0, -1, 1), count=2)["poynting_flux"]
        assert jnp.allclose(out[0, 0], 28.0)

    def test_energy_as_slices_mirrors_in_plane_axes(self):
        det = fdtdx.EnergyDetector(name="en", as_slices=True, plot=False)
        rng = np.random.default_rng(7)
        state = {
            "XY Plane": jnp.asarray(rng.standard_normal((1, 3, 4)), dtype=jnp.float32),  # (T, nx, ny)
            "XZ Plane": jnp.asarray(rng.standard_normal((1, 3, 5)), dtype=jnp.float32),  # (T, nx, nz)
            "YZ Plane": jnp.asarray(rng.standard_normal((1, 4, 5)), dtype=jnp.float32),  # (T, ny, nz)
        }
        out = _unfold_one_detector(det, state, (0, -1, 0), count=1)
        # Only y is symmetric: XY doubles its y axis, YZ doubles its y axis, XZ is unchanged.
        assert out["XY Plane"].shape == (1, 3, 8)
        assert out["YZ Plane"].shape == (1, 8, 5)
        assert out["XZ Plane"].shape == (1, 3, 5)
        # Energy is even -> the doubled axis is mirror-symmetric.
        assert jnp.allclose(out["XY Plane"], jnp.flip(out["XY Plane"], axis=2))

    def test_diffractive_raises(self):
        from fdtdx.objects.detectors.diffractive import DiffractiveDetector

        d = DiffractiveDetector(name="diff", frequencies=(3e14,), direction="+", plot=False)
        with pytest.raises(NotImplementedError):
            _unfold_one_detector(d, {"diffractive": jnp.zeros((1, 1, 1), dtype=jnp.complex64)}, (0, -1, 0), 1)


class TestUnfoldSourceMode:
    """unfold_source_mode reconstructs the full-domain mode profile from a source's stored _E/_H."""

    def _src(self, seed=0, ny=4, nz=2):
        rng = np.random.default_rng(seed)
        # (3, Nx=1 on propagation axis x, ny, nz)
        mode_E = jnp.asarray(rng.standard_normal((3, 1, ny, nz)), dtype=jnp.float32)
        mode_H = jnp.asarray(rng.standard_normal((3, 1, ny, nz)), dtype=jnp.float32)
        return _FakeModeSource(mode_E, mode_H, grid_shape=(1, ny, nz))

    def test_matches_unfold_fields_on_transverse_axes(self):
        src = self._src()
        cfg = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=1e-7), time=1e-15, symmetry=(0, -1, 1))
        e_full, h_full = unfold_source_mode(src, cfg)
        assert e_full.shape == (3, 1, 8, 4) and h_full.shape == (3, 1, 8, 4)
        assert jnp.allclose(e_full, unfold_fields(src._E, (0, -1, 1), "E"))
        assert jnp.allclose(h_full, unfold_fields(src._H, (0, -1, 1), "H"))

    def test_propagation_axis_symmetry_is_ignored(self):
        src = self._src()  # propagation axis = x (0)
        cfg = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=1e-7), time=1e-15, symmetry=(-1, -1, 0))
        e_full, _ = unfold_source_mode(src, cfg)
        # x is the propagation axis -> masked out; only y is unfolded.
        assert e_full.shape == (3, 1, 8, 2)
        assert jnp.allclose(e_full, unfold_fields(src._E, (0, -1, 0), "E"))

    def test_mode_not_computed_raises(self):
        src = _FakeModeSource(None, None, grid_shape=(1, 4, 2))
        cfg = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=1e-7), time=1e-15, symmetry=(0, -1, 0))
        with pytest.raises(ValueError, match="no computed mode profile"):
            unfold_source_mode(src, cfg)

    def test_no_transverse_symmetry_raises(self):
        src = self._src()  # propagation axis x
        cfg = fdtdx.SimulationConfig(
            grid=fdtdx.UniformGrid(spacing=1e-7), time=1e-15, symmetry=(-1, 0, 0)
        )  # only the prop axis
        with pytest.raises(ValueError, match="No transverse symmetry"):
            unfold_source_mode(src, cfg)
