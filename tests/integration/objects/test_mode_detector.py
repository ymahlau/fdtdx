"""Integration tests for ModeOverlapDetector with multiple wave characters.

Geometry: Si/SiO2 waveguide (800 nm Si core, SiO2 cladding, z-propagation).
Covers output shapes, field finiteness, guided-mode neff bounds, self-overlap
normalization, and JIT compatibility of the multi-frequency mode solver.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.core.physics.modes import compute_modes_tracked
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.initialization import apply_params
from fdtdx.materials import Material
from fdtdx.objects.detectors.mode import ModeOverlapDetector

# Minimum transverse cross-section for the ARPACK mode solver is 8x8 cells.
_RESOLUTION = 200e-9
_NCELLS_T = 8
_PML = 3
_TOTAL_T = _NCELLS_T + 2 * _PML  # 14 cells per transverse axis

_EPS_SI = fdtdx.constants.relative_permittivity_silicon  # 12.25, n≈3.5
_EPS_SIO2 = fdtdx.constants.relative_permittivity_silica  # 2.25,  n≈1.5
_CORE_SIZE = 4 * _RESOLUTION  # 800 nm — 4 cells, well-guided at 1550 nm

_WC_1200 = WaveCharacter(wavelength=1.20e-6)
_WC_1300 = WaveCharacter(wavelength=1.30e-6)
_WC_1450 = WaveCharacter(wavelength=1.45e-6)
_WC_1550 = WaveCharacter(wavelength=1.55e-6)


def _make_det_fixture(wave_characters):
    """ModeOverlapDetector over a Si/SiO2 waveguide: SiO2 cladding, 800x800 nm Si core,
    propagation along z.  The guided structure produces non-uniform field profiles
    that expose normalization errors invisible in uniform-medium geometries.
    """
    total_t = _TOTAL_T * _RESOLUTION
    total_z = (1 + 2 * _PML) * _RESOLUTION

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(total_t, total_t, total_z),
        material=Material(),
    )
    bound_cfg = fdtdx.BoundaryConfig(
        thickness_grid_minx=_PML,
        thickness_grid_maxx=_PML,
        thickness_grid_miny=_PML,
        thickness_grid_maxy=_PML,
        thickness_grid_minz=_PML,
        thickness_grid_maxz=_PML,
    )
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)

    # SiO2 cladding fills the entire volume
    cladding = fdtdx.UniformMaterialObject(
        name="cladding",
        partial_real_shape=(None, None, None),
        material=fdtdx.Material(permittivity=_EPS_SIO2),
    )

    # Si core centred in the transverse plane, spanning full z extent
    core = fdtdx.UniformMaterialObject(
        name="core",
        partial_real_shape=(_CORE_SIZE, _CORE_SIZE, None),
        material=fdtdx.Material(permittivity=_EPS_SI),
    )

    det = ModeOverlapDetector(
        name="det",
        wave_characters=wave_characters,
        direction="+",
        filter_pol=None,
        partial_grid_shape=(None, None, 1),
    )
    config = fdtdx.SimulationConfig(
        time=1e-14,
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
    )
    objects = [volume, cladding, core, det, *list(bound_dict.values())]
    constraints = [
        *bound_constraints,
        *cladding.same_position_and_size(volume),
        core.same_size(volume, axes=(2,)),
        core.place_at_center(volume, axes=(0, 1, 2)),
        det.same_size(volume, axes=(0, 1)),
        det.place_at_center(volume, axes=(0, 1, 2)),
    ]
    key = jax.random.PRNGKey(0)
    obj_container, arrays, _, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays = fdtdx.extend_material_to_pml(objects=obj_container, arrays=arrays)
    _, obj_container, _ = apply_params(arrays, obj_container, {}, key)
    return obj_container["det"]


@pytest.fixture(scope="module")
def two_freq_det():
    """Si/SiO2 waveguide: 2-wave-character detector (1550 nm + 1300 nm, z-plane)."""
    return _make_det_fixture((_WC_1550, _WC_1300))


@pytest.fixture(scope="module")
def four_freq_det():
    """Si/SiO2 waveguide: 4-wave-character detector (1200/1300/1450/1550 nm, z-plane)."""
    return _make_det_fixture((_WC_1200, _WC_1300, _WC_1450, _WC_1550))


@pytest.mark.parametrize("det_fixture,n_freqs", [("two_freq_det", 2), ("four_freq_det", 4)])
class TestModeOverlapDetectorApply:
    """End-to-end tests for ModeOverlapDetector.apply() across 2 and 4 wave characters."""

    def test_output_shapes(self, det_fixture, n_freqs, request):
        """mode_E, mode_H have shape (n_freqs, 3, *spatial); mode_neff has shape (n_freqs,)."""
        det = request.getfixturevalue(det_fixture)
        assert det._mode_E.ndim == 5
        assert det._mode_E.shape[0] == n_freqs
        assert det._mode_E.shape[1] == 3
        assert det._mode_E.shape == det._mode_H.shape
        assert det._mode_neff.shape == (n_freqs,)

    def test_fields_finite_and_neff_in_guided_range(self, det_fixture, n_freqs, request):
        """All mode field components are finite and neff is in (n_clad, n_core) at every frequency.

        For a guided mode in Si/SiO2, real(neff) must lie strictly between the cladding
        index (sqrt(2.25)~1.5) and the core index (sqrt(12.25)~3.5).  Failure here
        indicates wrong permittivity assignment or a solver returning a radiation mode.
        """
        det = request.getfixturevalue(det_fixture)
        assert jnp.all(jnp.isfinite(det._mode_E))
        assert jnp.all(jnp.isfinite(det._mode_H))
        neffs = np.real(np.array(det._mode_neff))
        n_clad = np.sqrt(_EPS_SIO2)  # ~1.5
        n_core = np.sqrt(_EPS_SI)  # ~3.5
        assert np.all(neffs > n_clad), f"neff below cladding index {n_clad:.3f}: {neffs}"
        assert np.all(neffs < n_core), f"neff above core index {n_core:.3f}: {neffs}"

    def test_compute_overlap_shape_and_dtype(self, det_fixture, n_freqs, request):
        """compute_overlap() on a zero phasor returns complex array of shape (n_freqs,)."""
        det = request.getfixturevalue(det_fixture)
        state = det.init_state()
        result = det.compute_overlap(state=state)
        assert result.shape == (n_freqs,)
        assert jnp.iscomplexobj(result)

    def test_self_overlap_equals_one(self, det_fixture, n_freqs, request):
        """Self-overlap of each mode field is ~1.0 at every frequency.

        A unit-power mode fed back as its own phasor must satisfy |overlap| = 1
        by the bidirectional formula.  Catches incorrect normalization factors,
        conjugation errors, or wrong area weights.
        """
        det = request.getfixturevalue(det_fixture)
        state = det.init_state()
        phasor = state["phasor"]
        for i in range(n_freqs):
            phasor = phasor.at[0, i, :3].set(det._mode_E[i])
            phasor = phasor.at[0, i, 3:].set(det._mode_H[i])
        state = {"phasor": phasor}
        result = det.compute_overlap(state=state)
        for i in range(n_freqs):
            mag = float(jnp.abs(result[i]))
            assert mag == pytest.approx(1.0, abs=0.05), f"Self-overlap at freq {i} = {mag:.4f}, expected ~= 1.0"


class TestComputeModesTrackedJitCompatibility:
    """``compute_modes_tracked`` is callable from inside ``jax.jit``."""

    def test_compute_modes_tracked_callable_under_jit(self):
        """``jax.jit`` traces through ``compute_modes_tracked`` without raising."""
        frequencies = [
            _WC_1550.get_frequency(),
            _WC_1300.get_frequency(),
        ]
        # Minimal 8x8 SiO2 cross-section, z-propagation (last dim == 1).
        inv_eps_stack = jnp.ones((2, 1, 8, 8, 1), dtype=jnp.float32) / _EPS_SIO2

        @jax.jit
        def fn(stack: jax.Array) -> jax.Array:
            mode_Es, _mode_Hs, _neffs = compute_modes_tracked(
                frequencies=frequencies,
                inv_permittivities_stack=stack,
                inv_permeabilities=1.0,
                resolution=_RESOLUTION,
            )
            return mode_Es

        result = fn(inv_eps_stack)
        assert result.shape == (2, 3, 8, 8, 1)
