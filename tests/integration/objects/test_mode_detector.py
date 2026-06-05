"""Integration tests for ModeOverlapDetector with multiple wave characters.

Exercises multi-frequency mode solving through the real tidy3d mode solver with a
uniform air permittivity slice.  Verifies output shapes, finiteness, neff
plausibility, and self-overlap normalization without mocking any external dependencies.
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

_WC_1550 = WaveCharacter(wavelength=1.55e-6)
_WC_1300 = WaveCharacter(wavelength=1.30e-6)


@pytest.fixture(scope="module")
def two_freq_det():
    """Uniform-air domain with a 2-wave-character ModeOverlapDetector (z-plane).

    Calls apply_params so compute_mode runs twice (once per frequency) through the
    real tidy3d ARPACK solver.  Fixture is module-scoped so the expensive solve runs once.
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

    det = ModeOverlapDetector(
        name="det",
        wave_characters=(_WC_1550, _WC_1300),
        direction="+",
        filter_pol=None,
        partial_grid_shape=(None, None, 1),  # z-plane → propagation_axis=2
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


class TestMultiFreqDetectorApply:
    """End-to-end tests for ModeOverlapDetector.apply() with two wave characters."""

    def test_mode_E_shape(self, two_freq_det):
        """_mode_E has shape (num_freqs, 3, transverse_x, transverse_y, 1)."""
        assert two_freq_det._mode_E.shape == (2, 3, _TOTAL_T, _TOTAL_T, 1)

    def test_mode_H_shape(self, two_freq_det):
        """_mode_H has shape (num_freqs, 3, transverse_x, transverse_y, 1)."""
        assert two_freq_det._mode_H.shape == (2, 3, _TOTAL_T, _TOTAL_T, 1)

    def test_neff_shape(self, two_freq_det):
        """_mode_neff has shape (num_freqs,)."""
        assert two_freq_det._mode_neff.shape == (2,)

    def test_mode_fields_finite(self, two_freq_det):
        """All mode field components are finite (no NaN/Inf)."""
        assert jnp.all(jnp.isfinite(two_freq_det._mode_E))
        assert jnp.all(jnp.isfinite(two_freq_det._mode_H))

    def test_neff_finite(self, two_freq_det):
        """All effective indices are finite."""
        assert jnp.all(jnp.isfinite(two_freq_det._mode_neff))

    def test_neff_positive_real_part(self, two_freq_det):
        """real(neff) > 0 for both frequencies (physical mode)."""
        neffs = np.real(np.array(two_freq_det._mode_neff))
        assert np.all(neffs > 0), f"Non-positive neff: {neffs}"

    def test_neff_continuity(self, two_freq_det):
        """Neff values at 1550 nm and 1300 nm stay close — no mode hopping."""
        neffs = np.real(np.array(two_freq_det._mode_neff))
        diff = abs(neffs[0] - neffs[1])
        assert diff < 0.2, (
            f"neff jumped between 1550 nm and 1300 nm: {neffs[0]:.4f} vs {neffs[1]:.4f} (diff={diff:.4f})"
        )

    def test_freq_count_matches_wave_characters(self, two_freq_det):
        """One mode (E, H, neff) returned per wave character."""
        assert two_freq_det._mode_E.shape[0] == 2
        assert two_freq_det._mode_H.shape[0] == 2
        assert two_freq_det._mode_neff.shape[0] == 2

    def test_compute_overlap_returns_correct_shape(self, two_freq_det):
        """compute_overlap() on a zero-phasor state returns shape (2,)."""
        state = two_freq_det.init_state()
        result = two_freq_det.compute_overlap(state=state)
        assert result.shape == (2,)
        assert jnp.iscomplexobj(result)

    def test_self_overlap_equals_one(self, two_freq_det):
        """Feeding the mode fields back as the phasor gives |overlap|≈1 at freq-0.

        This validates Poynting-flux normalization: a normalised mode has unit
        self-overlap when bidirectional_mode_overlap is divided by 4.
        """
        state = two_freq_det.init_state()
        # Fill phasor[0, 0, :3] with mode_E[0] and phasor[0, 0, 3:] with mode_H[0]
        phasor = state["phasor"]
        phasor = phasor.at[0, 0, :3].set(two_freq_det._mode_E[0])
        phasor = phasor.at[0, 0, 3:].set(two_freq_det._mode_H[0])
        state = {"phasor": phasor}
        result = two_freq_det.compute_overlap(state=state)
        # air domain: neff≈1, self-overlap should be real and ≈1
        assert jnp.abs(result[0]) == pytest.approx(1.0, abs=0.05), (
            f"Self-overlap at freq-0 is {float(jnp.abs(result[0])):.4f}, expected ≈1.0"
        )

    def test_neff_monotone_normal_dispersion(self, two_freq_det):
        """In air (non-dispersive), neff is the same at both frequencies.

        For a dispersive Si/SiO2 waveguide neff would increase at shorter wavelength
        (normal dispersion), but here we just verify the values are physically plausible
        and self-consistent (both equal ≈1.0 for air).
        """
        neffs = np.real(np.array(two_freq_det._mode_neff))
        # In air: neff ≈ 1 at all wavelengths
        assert np.all(neffs > 0.9), f"neffs below 0.9 in air: {neffs}"
        assert np.all(neffs < 1.1), f"neffs above 1.1 in air: {neffs}"
