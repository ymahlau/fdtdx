"""Tests for objects/detectors/mode.py - ModeOverlapDetector."""

from unittest.mock import patch

import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import RectilinearGrid
from fdtdx.core.physics.metrics import bidirectional_mode_overlap
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.detectors.mode import ModeOverlapDetector


def _make_mock_mode(shape=(3, 8, 8, 1), value=1.0, dtype=jnp.complex64):
    """Return (mode_E, mode_H, neff) with constant field arrays."""
    mode_E = jnp.ones(shape, dtype=dtype) * value
    mode_H = jnp.ones(shape, dtype=dtype) * value
    neff = jnp.asarray(1.5 + 0j, dtype=dtype)
    return mode_E, mode_H, neff


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

    def test_multiple_wave_chars_succeeds(self, simulation_config, plane_grid_slice, random_key, two_frequencies):
        """place_on_grid succeeds with multiple wave characters."""
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

    def test_compute_overlap_to_mode_matches_formula(
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
        """Overlap coefficient matches bidirectional_mode_overlap directly (1/4 is inside the formula)."""
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

        phasors = state["phasor"]
        phasors_E = phasors[0, 0, :3]
        phasors_H = phasors[0, 0, 3:]
        expected = bidirectional_mode_overlap(
            mode_E=mode_E,
            mode_H=mode_H,
            sim_E=phasors_E,
            sim_H=phasors_H,
            propagation_axis=det.propagation_axis,
            area_EuHv=det._cached_area_EuHv,
            area_EvHu=det._cached_area_EvHu,
        )

        overlap = det.compute_overlap_to_mode(state=state, mode_E=mode_E, mode_H=mode_H)

        assert jnp.isclose(overlap, expected)

    def test_compute_overlap_integrates_nonuniform_face_area(self, random_key, single_frequency):
        """Overlap uses Yee-staggered area weights (primal_u x dual_v) for the Ex/Hy term.

        Grid: x in [0,1,3] (widths 1, 2), y in [0,3,7] (widths 3, 4).
        Dual y-widths: [1.5, 3.5] (distance between adjacent y-cell centers).
        area_EuHv = outer([1, 2], [1.5, 3.5]) -> sum = 15.
        bidirectional_mode_overlap includes 0.25: 0.25 * 15 = 3.75.
        """
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

        assert jnp.allclose(overlap, jnp.asarray(15.0 / 4.0, dtype=jnp.complex64))

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
            jnp.ones((1, 3, 2, 2, 1), dtype=jnp.complex64),
            jnp.ones((1, 3, 2, 2, 1), dtype=jnp.complex64),
            jnp.zeros((1,), dtype=jnp.complex64),
        )
        det = ModeOverlapDetector(wave_characters=single_frequency, direction="+")
        det = det.place_on_grid(((0, 2), (0, 2), (0, 1)), config, random_key)

        det.apply(random_key, jnp.ones((1, 2, 2, 1), dtype=jnp.float32), 1.0)

        kwargs = mock_compute_mode.call_args.kwargs
        assert kwargs["resolution"] == grid.min_spacing
        assert kwargs["transverse_coords"] is not None


class TestModeOverlapDetectorComputeOverlapPath:
    """Tests for compute_overlap() - the stored-mode path (via aset)."""

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

        # stacked along a leading freq axis: (num_freqs, 3, *spatial)
        single_mode_E = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64) * 0.5
        single_mode_H = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64) * 0.5
        stacked_mode_E = jnp.stack([single_mode_E])  # (1, 3, 8, 8, 1)
        stacked_mode_H = jnp.stack([single_mode_H])  # (1, 3, 8, 8, 1)
        det = det.aset("_mode_E", stacked_mode_E, create_new_ok=True)
        det = det.aset("_mode_H", stacked_mode_H, create_new_ok=True)

        state = det.init_state()
        state = det.update(jnp.array(0), constant_E_field, constant_H_field, state, inv_permittivity, inv_permeability)

        # compute_overlap() returns (num_freqs,) array; compare element to direct call
        result_stored = det.compute_overlap(state=state)
        result_explicit = det.compute_overlap_to_mode(state=state, mode_E=single_mode_E, mode_H=single_mode_H)

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


class TestModeOverlapDetectorMultiFrequency:
    """Tests for multi-frequency support in ModeOverlapDetector.

    apply() delegates all frequency solves to compute_mode (a single
    jax.pure_callback), which is fully compatible with jax.jit.  Field-overlap
    continuity tracking happens inside the callback where numpy arrays are
    concrete, so it never breaks JAX tracing.
    """

    def _mock_compute(self, n: int, shape=(3, 8, 8, 1)) -> tuple:
        """Return value for a mocked compute_mode with n frequencies."""
        return (
            jnp.ones((n, *shape), dtype=jnp.complex64),
            jnp.ones((n, *shape), dtype=jnp.complex64),
            jnp.zeros((n,), dtype=jnp.complex64),
        )

    @patch("fdtdx.objects.detectors.mode.compute_mode")
    def test_apply_calls_compute_mode_once(
        self, mock_compute, simulation_config, plane_grid_slice, random_key, two_frequencies
    ):
        """apply() calls compute_mode exactly once regardless of frequency count."""
        mock_compute.return_value = self._mock_compute(2)
        det = ModeOverlapDetector(wave_characters=two_frequencies, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det.apply(random_key, jnp.ones((1, 8, 8, 1), dtype=jnp.float32), 1.0)
        assert mock_compute.call_count == 1

    @patch("fdtdx.objects.detectors.mode.compute_mode")
    def test_apply_passes_frequency_list(
        self, mock_compute, simulation_config, plane_grid_slice, random_key, two_frequencies
    ):
        """apply() passes all wave-character frequencies to compute_mode."""
        mock_compute.return_value = self._mock_compute(2)
        det = ModeOverlapDetector(wave_characters=two_frequencies, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det.apply(random_key, jnp.ones((1, 8, 8, 1), dtype=jnp.float32), 1.0)
        kwargs = mock_compute.call_args.kwargs
        expected = [wc.get_frequency() for wc in two_frequencies]
        assert kwargs["frequencies"] == expected

    @patch("fdtdx.objects.detectors.mode.compute_mode")
    def test_apply_stores_stacked_mode_fields(
        self, mock_compute, simulation_config, plane_grid_slice, random_key, two_frequencies
    ):
        """_mode_E and _mode_H have shape (num_freqs, 3, *spatial) after apply()."""
        mock_compute.return_value = self._mock_compute(2)
        det = ModeOverlapDetector(wave_characters=two_frequencies, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((1, 8, 8, 1), dtype=jnp.float32), 1.0)
        assert det._mode_E.shape == (2, 3, 8, 8, 1)
        assert det._mode_H.shape == (2, 3, 8, 8, 1)

    @patch("fdtdx.objects.detectors.mode.compute_mode")
    def test_freq0_overlap_independent_of_freq1_phasor(
        self, mock_compute, simulation_config, plane_grid_slice, random_key, two_frequencies
    ):
        """Populating only the freq-1 phasor slot leaves the freq-0 overlap at zero."""
        mock_compute.return_value = self._mock_compute(2)
        det = ModeOverlapDetector(wave_characters=two_frequencies, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((1, 8, 8, 1), dtype=jnp.float32), 1.0)
        state = det.init_state()
        phasor = state["phasor"].at[0, 1, :].set(1.0)
        state = {"phasor": phasor}
        result = det.compute_overlap(state=state)
        assert jnp.isclose(jnp.abs(result[0]), 0.0)

    @patch("fdtdx.objects.detectors.mode.compute_mode")
    def test_freq1_overlap_independent_of_freq0_phasor(
        self, mock_compute, simulation_config, plane_grid_slice, random_key, two_frequencies
    ):
        """Populating only the freq-0 phasor slot leaves the freq-1 overlap at zero."""
        mock_compute.return_value = self._mock_compute(2)
        det = ModeOverlapDetector(wave_characters=two_frequencies, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((1, 8, 8, 1), dtype=jnp.float32), 1.0)
        state = det.init_state()
        phasor = state["phasor"].at[0, 0, :].set(1.0)
        state = {"phasor": phasor}
        result = det.compute_overlap(state=state)
        assert jnp.isclose(jnp.abs(result[1]), 0.0)

    @patch("fdtdx.objects.detectors.mode.compute_mode")
    def test_mode_frequency_correspondence(
        self, mock_compute, simulation_config, plane_grid_slice, random_key, two_frequencies
    ):
        """Each frequency's overlap uses its own mode, not the other frequency's mode.

        Strategy: give each frequency a distinct mode_E (Ex=1 for freq-0, Ex=2 for freq-1).
        Set both phasor slots identically. The overlap magnitudes should differ proportionally.
        """
        mode_E_f0 = jnp.zeros((3, 8, 8, 1), dtype=jnp.complex64).at[0].set(1.0)
        mode_H_f0 = jnp.zeros((3, 8, 8, 1), dtype=jnp.complex64).at[1].set(1.0)
        mode_E_f1 = jnp.zeros((3, 8, 8, 1), dtype=jnp.complex64).at[0].set(2.0)
        mode_H_f1 = jnp.zeros((3, 8, 8, 1), dtype=jnp.complex64).at[1].set(2.0)
        mock_compute.return_value = (
            jnp.stack([mode_E_f0, mode_E_f1]),
            jnp.stack([mode_H_f0, mode_H_f1]),
            jnp.array([1.5 + 0j, 1.5 + 0j], dtype=jnp.complex64),
        )

        det = ModeOverlapDetector(wave_characters=two_frequencies, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((1, 8, 8, 1), dtype=jnp.float32), 1.0)

        state = det.init_state()
        phasor = state["phasor"].at[0, :, 0].set(1.0).at[0, :, 4].set(0.5)
        state = {"phasor": phasor}

        result = det.compute_overlap(state=state)

        # freq-1 mode has Ex=2, Hy=2 vs Ex=1, Hy=1 for freq-0 — doubling both fields → 2x overlap
        ratio = jnp.abs(result[1]) / jnp.abs(result[0])
        assert jnp.isclose(ratio, 2.0, atol=0.01)
