"""Tests for objects/detectors/mode.py - ModeOverlapDetector."""

import jax.numpy as jnp
import pytest

from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.detectors.mode import ModeOverlapDetector


@pytest.fixture
def single_frequency():
    """Single optical frequency wave character."""
    return [WaveCharacter(wavelength=1e-6)]


@pytest.fixture
def two_frequencies():
    """Two frequency wave characters."""
    return [WaveCharacter(wavelength=1e-6), WaveCharacter(wavelength=1.5e-6)]


class TestModeOverlapDetectorDefaults:
    """Tests for ModeOverlapDetector default configuration."""

    def test_components_always_all_six(self, single_frequency):
        """components is always all six EM components (fixed, non-configurable)."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        assert det.components == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

    def test_plot_always_false(self, single_frequency):
        """plot is always False (scalar overlap is not plotable)."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        assert det.plot is False

    def test_mode_index_default_zero(self, single_frequency):
        """mode_index defaults to 0 (fundamental mode)."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        assert det.mode_index == 0

    def test_filter_pol_default_none(self, single_frequency):
        """filter_pol defaults to None (no polarization filtering)."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        assert det.filter_pol is None

    def test_direction_plus_stored(self, single_frequency):
        """Positive propagation direction is stored."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        assert det.direction == "+"

    def test_direction_minus_stored(self, single_frequency):
        """Negative propagation direction is stored."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="-")
        assert det.direction == "-"

    def test_custom_mode_index(self, single_frequency):
        """Custom mode index is stored."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+", mode_index=2)
        assert det.mode_index == 2

    def test_filter_pol_te(self, single_frequency):
        """TE polarization filter is stored."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+", filter_pol="te")
        assert det.filter_pol == "te"

    def test_filter_pol_tm(self, single_frequency):
        """TM polarization filter is stored."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+", filter_pol="tm")
        assert det.filter_pol == "tm"


class TestModeOverlapDetectorPropagationAxis:
    """Tests for the propagation_axis property."""

    def test_propagation_axis_z(self, simulation_config, plane_grid_slice, random_key, single_frequency):
        """Plane perpendicular to z-axis → propagation_axis == 2."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        assert det.propagation_axis == 2

    def test_propagation_axis_x(self, simulation_config, random_key, single_frequency):
        """Plane perpendicular to x-axis → propagation_axis == 0."""
        x_plane = ((0, 1), (0, 8), (0, 8))
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(x_plane, simulation_config, random_key)
        assert det.propagation_axis == 0

    def test_propagation_axis_y(self, simulation_config, random_key, single_frequency):
        """Plane perpendicular to y-axis → propagation_axis == 1."""
        y_plane = ((0, 8), (0, 1), (0, 8))
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(y_plane, simulation_config, random_key)
        assert det.propagation_axis == 1

    def test_non_plane_shape_raises(self, simulation_config, small_grid_slice, random_key, single_frequency):
        """Volume detector shape raises with message about invalid shape."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(small_grid_slice, simulation_config, random_key)
        with pytest.raises(Exception, match="Invalid ModeOverlapDetector shape"):
            _ = det.propagation_axis


class TestModeOverlapDetectorPlaceOnGrid:
    """Tests for place_on_grid constraints."""

    def test_single_wave_char_succeeds(self, simulation_config, plane_grid_slice, random_key, single_frequency):
        """Single wave character placement succeeds."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        placed = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        assert placed is not None

    def test_multiple_wave_chars_raises_not_implemented(
        self, simulation_config, plane_grid_slice, random_key, two_frequencies
    ):
        """Multiple wave characters raise NotImplementedError (not yet supported)."""
        det = ModeOverlapDetector(wave_characters=two_frequencies, direction="+")
        with pytest.raises(NotImplementedError):
            det.place_on_grid(plane_grid_slice, simulation_config, random_key)


class TestModeOverlapDetectorComputeOverlap:
    """Tests for compute_overlap and compute_overlap_to_mode."""

    def test_compute_overlap_without_apply_raises(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        single_frequency,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """compute_overlap() raises if apply() has not been called first."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()
        state = det.update(
            jnp.array(0),
            constant_E_field,
            constant_H_field,
            state,
            inv_permittivity,
            inv_permeability,
        )

        with pytest.raises(Exception, match="Need to call apply"):
            det.compute_overlap(state)

    def test_compute_overlap_to_mode_returns_complex(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        single_frequency,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """compute_overlap_to_mode returns a complex scalar."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()
        state = det.update(
            jnp.array(0),
            constant_E_field,
            constant_H_field,
            state,
            inv_permittivity,
            inv_permeability,
        )

        # Mode fields must match spatial shape of the detector slice
        mode_E = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64) * 0.5
        mode_H = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64) * 0.5

        overlap = det.compute_overlap_to_mode(state=state, mode_E=mode_E, mode_H=mode_H)

        assert jnp.iscomplexobj(overlap)

    def test_compute_overlap_to_mode_zero_phasor_zero_overlap(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        single_frequency,
    ):
        """Zero phasor state (no field updates) gives zero mode overlap."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()  # All-zero phasors

        mode_E = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64)
        mode_H = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64)

        overlap = det.compute_overlap_to_mode(state=state, mode_E=mode_E, mode_H=mode_H)

        assert jnp.isclose(jnp.abs(overlap), 0.0)

    def test_compute_overlap_to_mode_nonzero_fields_nonzero_overlap(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        single_frequency,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Non-zero phasor fields with non-parallel mode fields give non-zero overlap.

        With constant_E_field = [1,1,1] and constant_H_field = [0.5,0.5,0.5],
        the phasors have equal components. To get a non-zero cross product we
        break the symmetry: mode_E has only Ex=1, mode_H has only Hy=1.
        Then cross(mode_E, conj(pH))_z = Ex*conj(pHy) - Ey*conj(pHx) = 1*0.5 ≠ 0.
        """
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()
        state = det.update(
            jnp.array(0),
            constant_E_field,
            constant_H_field,
            state,
            inv_permittivity,
            inv_permeability,
        )

        # Asymmetric mode fields: Ex-only source, Hy-only mode → non-parallel
        mode_E = jnp.zeros((3, 8, 8, 1), dtype=jnp.complex64).at[0].set(1.0)
        mode_H = jnp.zeros((3, 8, 8, 1), dtype=jnp.complex64).at[1].set(1.0)

        overlap = det.compute_overlap_to_mode(state=state, mode_E=mode_E, mode_H=mode_H)

        assert jnp.abs(overlap) > 0

    def test_compute_overlap_to_mode_zero_mode_zero_overlap(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        single_frequency,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Zero mode fields give zero overlap regardless of phasor values."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()
        state = det.update(
            jnp.array(0),
            constant_E_field,
            constant_H_field,
            state,
            inv_permittivity,
            inv_permeability,
        )

        mode_E = jnp.zeros((3, 8, 8, 1), dtype=jnp.complex64)
        mode_H = jnp.zeros((3, 8, 8, 1), dtype=jnp.complex64)

        overlap = det.compute_overlap_to_mode(state=state, mode_E=mode_E, mode_H=mode_H)

        assert jnp.isclose(jnp.abs(overlap), 0.0)

    def test_compute_overlap_to_mode_scaled_quarter(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        single_frequency,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Overlap coefficient includes the 1/4 normalization factor."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()
        state = det.update(
            jnp.array(0),
            constant_E_field,
            constant_H_field,
            state,
            inv_permittivity,
            inv_permeability,
        )

        # Asymmetric mode fields (Ex-only / Hy-only) so cross product is non-trivial
        mode_E = jnp.zeros((3, 8, 8, 1), dtype=jnp.complex64).at[0].set(1.0)
        mode_H = jnp.zeros((3, 8, 8, 1), dtype=jnp.complex64).at[1].set(1.0)

        # Manually replicate the formula to verify the 1/4 factor
        phasors = state["phasor"]
        phasors_E = phasors[0, 0, :3]
        phasors_H = phasors[0, 0, 3:]
        E_cross_H = jnp.cross(mode_E, jnp.conj(phasors_H), axis=0)[det.propagation_axis]
        E_cross_H_back = jnp.cross(jnp.conj(phasors_E), mode_H, axis=0)[det.propagation_axis]
        expected = jnp.sum(E_cross_H + E_cross_H_back) / 4.0

        overlap = det.compute_overlap_to_mode(state=state, mode_E=mode_E, mode_H=mode_H)

        assert jnp.isclose(overlap, expected)


class TestModeOverlapDetectorComputeOverlapPath:
    """Tests for compute_overlap() – the stored-mode path (via aset)."""

    def test_compute_overlap_without_apply_raises(
        self, single_frequency, simulation_config, plane_grid_slice, random_key
    ):
        """compute_overlap() raises when mode fields were never set (no apply() call)."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()

        with pytest.raises(Exception, match="apply"):
            det.compute_overlap(state=state)

    def test_compute_overlap_with_stored_mode_fields(
        self,
        single_frequency,
        simulation_config,
        plane_grid_slice,
        random_key,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """compute_overlap() uses stored _mode_E/_mode_H (the apply() success path)."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)

        # Manually set mode fields (simulating what apply() would do)
        mode_E = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64) * 0.5
        mode_H = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64) * 0.5
        det = det.aset("_mode_E", mode_E, create_new_ok=True)
        det = det.aset("_mode_H", mode_H, create_new_ok=True)

        state = det.init_state()
        state = det.update(jnp.array(0), constant_E_field, constant_H_field, state, inv_permittivity, inv_permeability)

        # compute_overlap() should work and give same result as compute_overlap_to_mode()
        result_stored = det.compute_overlap(state=state)
        result_explicit = det.compute_overlap_to_mode(state=state, mode_E=mode_E, mode_H=mode_H)

        assert jnp.iscomplexobj(result_stored)
        assert jnp.isclose(result_stored, result_explicit)
