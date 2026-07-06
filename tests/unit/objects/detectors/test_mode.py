"""Tests for objects/detectors/mode.py - ModeOverlapDetector."""

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.constants import c
from fdtdx.core.grid import RectilinearGrid
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

    def test_bend_radius_default_none(self, single_frequency):
        """bend_radius defaults to None (straight waveguide)."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        assert det.bend_radius is None

    def test_bend_axis_default_none(self, single_frequency):
        """bend_axis defaults to None (straight waveguide)."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        assert det.bend_axis is None

    def test_bend_radius_stored(self, single_frequency):
        """Custom bend_radius value is stored."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+", bend_radius=5e-6, bend_axis=0)
        assert det.bend_radius == 5e-6

    def test_bend_axis_stored(self, single_frequency):
        """Custom bend_axis value is stored."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+", bend_radius=5e-6, bend_axis=0)
        assert det.bend_axis == 0


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

    def test_multiple_wave_chars_placement_succeeds(
        self, simulation_config, plane_grid_slice, random_key, two_frequencies
    ):
        """Multiple wave characters placement succeeds."""
        det = ModeOverlapDetector(wave_characters=two_frequencies, direction="+")
        placed = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        assert placed is not None


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
        expected = jnp.sum((E_cross_H + E_cross_H_back) * det._face_area_weights()) / 4.0

        overlap = det.compute_overlap_to_mode(state=state, mode_E=mode_E, mode_H=mode_H)

        assert jnp.isclose(overlap, expected)

    def test_compute_overlap_invariant_to_absolute_face_area(self, random_key, single_frequency):
        """A constant overlap integrand only considers the relative detector face area."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0]),
            y_edges=jnp.asarray([0.0, 3.0, 7.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )
        config = SimulationConfig(time=1e-8, grid=grid, backend="cpu")
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(((0, 2), (0, 2), (0, 1)), config, random_key)

        state = det.init_state()
        phasor = jnp.zeros_like(state["phasor"]).at[0, 0, 4].set(1.0)
        state = {"phasor": phasor}
        mode_E = jnp.zeros((3, 2, 2, 1), dtype=jnp.complex64).at[0].set(1.0)
        mode_H = jnp.zeros((3, 2, 2, 1), dtype=jnp.complex64)

        overlap = det.compute_overlap_to_mode(state=state, mode_E=mode_E, mode_H=mode_H)

        assert jnp.allclose(overlap, jnp.asarray(1.0, dtype=jnp.complex64))

    def test_transverse_edge_coordinates_follow_detector_slice(self, random_key, single_frequency):
        """Mode solving receives the physical edge arrays for the transverse detector axes."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0]),
            y_edges=jnp.asarray([0.0, 3.0, 7.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )
        config = SimulationConfig(time=1e-8, grid=grid, backend="cpu")
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(((0, 2), (0, 2), (0, 1)), config, random_key)

        x_edges, y_edges = det._transverse_edge_coordinates()

        assert jnp.allclose(x_edges, jnp.asarray([0.0, 1.0, 3.0]))
        assert jnp.allclose(y_edges, jnp.asarray([0.0, 3.0, 7.0]))

    @patch("fdtdx.objects.detectors.mode.compute_mode")
    def test_apply_passes_nonuniform_mode_coordinates(self, mock_compute_mode, random_key, single_frequency):
        """Mode detector apply should not require a scalar grid spacing on stretched grids."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0]),
            y_edges=jnp.asarray([0.0, 3.0, 7.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )
        config = SimulationConfig(time=1e-8, grid=grid, backend="cpu")
        mock_compute_mode.return_value = (
            jnp.ones((3, 2, 2, 1), dtype=jnp.complex64),
            jnp.ones((3, 2, 2, 1), dtype=jnp.complex64),
            jnp.asarray(1.5 + 0j, dtype=jnp.complex64),
        )
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(((0, 2), (0, 2), (0, 1)), config, random_key)

        det.apply(random_key, jnp.ones((1, 2, 2, 1), dtype=jnp.float32), 1.0)

        kwargs = mock_compute_mode.call_args.kwargs
        assert kwargs["resolution"] == grid.min_spacing
        assert kwargs["transverse_coords"] is not None


class TestModeOverlapDetectorComputeOverlapPath:
    """Tests for compute_overlap() - the stored-mode path (via aset)."""

    @staticmethod
    def _layered_waveguide_detector(mode_index, random_key):
        """Build an asymmetric layered waveguide with frequency-dependent mode ordering.

        The unequal upper and lower high-index layers support higher-order mode
        families with different dispersion. Their effective-index curves cross
        over this sweep, so sorting each frequency only by effective index swaps
        the modes stored at indices 5 and 6.
        """
        spacing = 0.1e-6
        transverse_edges = jnp.arange(-5e-6, 5e-6 + spacing / 2, spacing)
        grid = RectilinearGrid(
            x_edges=transverse_edges,
            y_edges=jnp.asarray([0.0, spacing]),
            z_edges=transverse_edges,
        )
        config = SimulationConfig(time=1e-12, grid=grid, backend="cpu")

        center_wavelength = 2.0e-6
        frequency_center = c / center_wavelength
        frequency_width = frequency_center / 3.0
        frequencies = np.linspace(
            frequency_center - frequency_width,
            frequency_center + frequency_width,
            7,
        )
        det = ModeOverlapDetector(
            wave_characters=[WaveCharacter(frequency=float(frequency)) for frequency in frequencies],
            direction="+",
            mode_index=mode_index,
        )
        det = det.place_on_grid(((0, 100), (0, 1), (0, 100)), config, random_key)

        centers_um = 0.5 * (np.asarray(transverse_edges[:-1]) + np.asarray(transverse_edges[1:])) / 1e-6
        x, z = np.meshgrid(centers_um, centers_um, indexing="ij")
        permittivity = np.ones_like(x)
        permittivity[(np.abs(x) <= 0.75) & (np.abs(z) <= 0.5)] = 4.0
        permittivity[(np.abs(x) <= 0.75) & (z >= -1.3) & (z <= -0.3)] = 8.0
        permittivity[(np.abs(x) <= 0.75) & (z >= 0.3) & (z <= 1.1)] = 8.0
        inv_permittivity = jnp.asarray(1.0 / permittivity, dtype=jnp.float32)[None, :, None, :]

        return det.apply(random_key, inv_permittivity, 1.0)

    @staticmethod
    def _minimum_consecutive_mode_overlap(det):
        consecutive_overlaps = []
        for mode_a, mode_b in zip(det._mode_E[:-1], det._mode_E[1:], strict=True):
            mode_a = mode_a.ravel()
            mode_b = mode_b.ravel()
            overlap = jnp.abs(jnp.vdot(mode_a, mode_b)) / jnp.sqrt(
                jnp.real(jnp.vdot(mode_a, mode_a)) * jnp.real(jnp.vdot(mode_b, mode_b))
            )
            consecutive_overlaps.append(overlap)

        return jnp.min(jnp.asarray(consecutive_overlaps))

    def test_multifrequency_apply_keeps_fundamental_mode_identity(self, random_key):
        """The real fundamental mode remains continuous across the frequency sweep."""
        det = self._layered_waveguide_detector(mode_index=0, random_key=random_key)

        assert self._minimum_consecutive_mode_overlap(det) > 0.9

    @pytest.mark.xfail(
        strict=True,
        reason="ModeOverlapDetector solves each frequency independently and does not track physical mode identity",
    )
    def test_multifrequency_apply_tracks_higher_order_mode_identity(self, random_key):
        """The real higher-order mode requires overlap tracking across frequencies."""
        det = self._layered_waveguide_detector(mode_index=5, random_key=random_key)

        assert self._minimum_consecutive_mode_overlap(det) > 0.9

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

        # Manually set mode fields (simulating what apply() would do).
        # Shape is (N, 3, *spatial) where N=len(wave_characters)=1.
        mode_E = jnp.ones((1, 3, 8, 8, 1), dtype=jnp.complex64) * 0.5
        mode_H = jnp.ones((1, 3, 8, 8, 1), dtype=jnp.complex64) * 0.5
        det = det.aset("_mode_E", mode_E, create_new_ok=True)
        det = det.aset("_mode_H", mode_H, create_new_ok=True)

        state = det.init_state()
        state = det.update(jnp.array(0), constant_E_field, constant_H_field, state, inv_permittivity, inv_permeability)

        # compute_overlap() returns (N,); compute_overlap_to_mode returns a scalar.
        result_stored = det.compute_overlap(state=state)
        result_explicit = det.compute_overlap_to_mode(state=state, mode_E=mode_E[0], mode_H=mode_H[0])

        assert jnp.iscomplexobj(result_stored)
        assert result_stored.shape == (1,)
        assert jnp.isclose(result_stored[0], result_explicit)


class TestModeOverlapDetectorBentWaveguide:
    """Tests for bent waveguide mode parameters (bend_radius / bend_axis)."""

    def test_only_bend_radius_raises(self, single_frequency, simulation_config, plane_grid_slice, random_key):
        """Setting bend_radius without bend_axis raises ValueError."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+", bend_radius=5e-6)
        with pytest.raises(ValueError, match="both be set or both be None"):
            det.place_on_grid(plane_grid_slice, simulation_config, random_key)

    def test_only_bend_axis_raises(self, single_frequency, simulation_config, plane_grid_slice, random_key):
        """Setting bend_axis without bend_radius raises ValueError."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+", bend_axis=0)
        with pytest.raises(ValueError, match="both be set or both be None"):
            det.place_on_grid(plane_grid_slice, simulation_config, random_key)

    def test_bend_axis_equals_propagation_axis_raises(
        self, single_frequency, simulation_config, plane_grid_slice, random_key
    ):
        """bend_axis matching the propagation axis (z=2 for plane_grid_slice) raises ValueError."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+", bend_radius=5e-6, bend_axis=2)
        with pytest.raises(ValueError, match="must differ from the propagation axis"):
            det.place_on_grid(plane_grid_slice, simulation_config, random_key)

    def test_valid_bend_params_x_axis(self, single_frequency, simulation_config, plane_grid_slice, random_key):
        """Valid bend_radius + bend_axis=0 (x-axis, transverse to z-propagation) succeeds."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+", bend_radius=5e-6, bend_axis=0)
        placed = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        assert placed.bend_radius == 5e-6
        assert placed.bend_axis == 0

    def test_valid_bend_params_y_axis(self, single_frequency, simulation_config, plane_grid_slice, random_key):
        """Valid bend_radius + bend_axis=1 (y-axis, transverse to z-propagation) succeeds."""
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+", bend_radius=10e-6, bend_axis=1)
        placed = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        assert placed.bend_radius == 10e-6
        assert placed.bend_axis == 1

    def test_valid_bend_params_x_propagation(self, single_frequency, simulation_config, random_key):
        """Valid bend setup for x-propagating waveguide: bend_axis must be 1 or 2."""
        x_plane = ((0, 1), (0, 8), (0, 8))
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+", bend_radius=5e-6, bend_axis=2)
        placed = det.place_on_grid(x_plane, simulation_config, random_key)
        assert placed.propagation_axis == 0
        assert placed.bend_axis == 2

    def test_bend_axis_same_as_x_propagation_raises(self, single_frequency, simulation_config, random_key):
        """bend_axis=0 for x-propagating waveguide (propagation_axis=0) raises ValueError."""
        x_plane = ((0, 1), (0, 8), (0, 8))
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+", bend_radius=5e-6, bend_axis=0)
        with pytest.raises(ValueError, match="must differ from the propagation axis"):
            det.place_on_grid(x_plane, simulation_config, random_key)
