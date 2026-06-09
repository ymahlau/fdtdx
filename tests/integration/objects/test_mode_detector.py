"""Integration tests for ModeOverlapDetector with multiple wave characters.

Exercises multi-frequency mode solving through the real tidy3d mode solver with a
uniform air permittivity slice.  Verifies output shapes, finiteness, positive neff,
and self-overlap normalization without mocking any external dependencies.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.initialization import apply_params
from fdtdx.materials import Material
from fdtdx.objects.detectors.mode import ModeOverlapDetector

# Minimum cross-section for tidy3d's ARPACK solver is 8x8 cells.
_RESOLUTION = 200e-9
_NCELLS_T = 8
_PML = 3
_TOTAL_T = _NCELLS_T + 2 * _PML  # 14 cells per transverse axis

_WC_1200 = WaveCharacter(wavelength=1.20e-6)
_WC_1300 = WaveCharacter(wavelength=1.30e-6)
_WC_1450 = WaveCharacter(wavelength=1.45e-6)
_WC_1550 = WaveCharacter(wavelength=1.55e-6)


def _make_det_fixture(wave_characters):
    """Build and apply a ModeOverlapDetector with the given wave characters."""
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
    objects = [volume, det, *list(bound_dict.values())]
    constraints = [
        *bound_constraints,
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
    """Uniform-air domain: 2-wave-character detector (1550 nm + 1300 nm, z-plane)."""
    return _make_det_fixture((_WC_1550, _WC_1300))


@pytest.fixture(scope="module")
def four_freq_det():
    """Uniform-air domain: 4-wave-character detector (1200/1300/1450/1550 nm, z-plane)."""
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

    def test_fields_finite_and_neff_positive(self, det_fixture, n_freqs, request):
        """All mode field components are finite and real(neff) > 0 at every frequency."""
        det = request.getfixturevalue(det_fixture)
        assert jnp.all(jnp.isfinite(det._mode_E))
        assert jnp.all(jnp.isfinite(det._mode_H))
        neffs = np.real(np.array(det._mode_neff))
        assert np.all(neffs > 0), f"Non-positive neff: {neffs}"

    def test_compute_overlap_shape_and_dtype(self, det_fixture, n_freqs, request):
        """compute_overlap() on a zero phasor returns complex array of shape (n_freqs,)."""
        det = request.getfixturevalue(det_fixture)
        state = det.init_state()
        result = det.compute_overlap(state=state)
        assert result.shape == (n_freqs,)
        assert jnp.iscomplexobj(result)

    def test_self_overlap_equals_one(self, det_fixture, n_freqs, request):
        """Feeding each mode back as its own phasor gives |overlap| ≈ 1 at every frequency.

        Validates Poynting-flux normalization end-to-end through the real tidy3d solver:
        a unit-power mode must have self-overlap = 1 by the bidirectional formula.
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
            assert mag == pytest.approx(1.0, abs=0.05), f"Self-overlap at freq {i} = {mag:.4f}, expected ≈ 1.0"
