"""Integration tests for fdtdx.utils.sparams.

Tests cover setup_sparams_simulation (no FDTD time-stepping), the
determine_input_norm_detector_name helper, the fast ValueError guard
inside calculate_sparam, and the multi-source source-disabling path that
exercises the aset call on line 269 of sparams.py.  Full S-parameter physics
are tested separately in tests/simulation/physics/test_sparams.py.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.utils.sparams import (
    PortSpec,
    _make_port_shape,
    calculate_sparam,
    calculate_sparams,
    determine_input_norm_detector_name,
    setup_sparams_simulation,
)

# ---------------------------------------------------------------------------
# Shared domain constants – tiny grid for speed
# ---------------------------------------------------------------------------

_RESOLUTION = 200e-9  # 200 nm
_DOMAIN_SIZE = (1e-6, 400e-9, 400e-9)  # 5 × 2 × 2 core cells
_PML_LAYERS = 3  # total grid: 11 × 8 × 8 = 704 cells
_WAVELENGTH = 1.55e-6
_MAX_TIME = 100e-15

# Port positioned well inside the background volume (background-relative coords)
_PORT_CENTER_IN = (1.1e-6, 0.8e-6, 0.8e-6)
_PORT_CENTER_OUT = (1.65e-6, 0.8e-6, 0.8e-6)
_PORT_WIDTH = 0.8e-6
_PORT_HEIGHT = 0.8e-6


def _make_setup(**kwargs):
    """Thin wrapper around setup_sparams_simulation with shared defaults."""
    defaults = dict(
        polygons=[],
        wavelength=_WAVELENGTH,
        resolution=_RESOLUTION,
        max_time=_MAX_TIME,
        domain_size=_DOMAIN_SIZE,
        pml_layers=_PML_LAYERS,
        key=jax.random.PRNGKey(0),
    )
    defaults.update(kwargs)
    return setup_sparams_simulation(**defaults)


# ---------------------------------------------------------------------------
# Session-scoped fixtures – computed once per test session for speed
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def minimal_setup():
    """1 named input port (SrcA) + 1 named output port (DetB)."""
    return _make_setup(
        input_ports=[
            PortSpec(
                center=_PORT_CENTER_IN,
                axis=0,
                direction="+",
                width=_PORT_WIDTH,
                height=_PORT_HEIGHT,
                name="SrcA",
            )
        ],
        output_ports=[
            PortSpec(
                center=_PORT_CENTER_OUT,
                axis=0,
                direction="+",
                width=_PORT_WIDTH,
                height=_PORT_HEIGHT,
                name="DetB",
            )
        ],
    )


@pytest.fixture(scope="session")
def two_port_setup():
    """2 named input ports (P1, P2) + 2 named output ports (P3, P4)."""
    xs = [0.88e-6, 1.1e-6, 1.32e-6, 1.54e-6]
    cy, cz = 0.8e-6, 0.8e-6
    names = ["P1", "P2", "P3", "P4"]
    ports = [
        PortSpec(center=(x, cy, cz), axis=0, direction="+", width=_PORT_WIDTH, height=_PORT_HEIGHT, name=n)
        for x, n in zip(xs, names)
    ]
    return _make_setup(input_ports=ports[:2], output_ports=ports[2:])


@pytest.fixture(scope="session")
def no_source_setup():
    """0 input ports + 1 output port (DetOnly).

    Used to test the calculate_sparam ValueError guard without triggering the
    latent multi-source aset bug or any FDTD execution.
    """
    return _make_setup(
        input_ports=[],
        output_ports=[
            PortSpec(
                center=_PORT_CENTER_OUT,
                axis=0,
                direction="+",
                width=_PORT_WIDTH,
                height=_PORT_HEIGHT,
                name="DetOnly",
            )
        ],
    )


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class TestSetupSparamsReturnTypes:
    def test_returns_object_container(self, minimal_setup):
        objs, arrays, config = minimal_setup
        assert isinstance(objs, ObjectContainer)

    def test_returns_array_container(self, minimal_setup):
        objs, arrays, config = minimal_setup
        assert isinstance(arrays, ArrayContainer)

    def test_returns_simulation_config(self, minimal_setup):
        objs, arrays, config = minimal_setup
        assert isinstance(config, SimulationConfig)

    def test_config_resolution_preserved(self, minimal_setup):
        objs, arrays, config = minimal_setup
        assert config.resolution == _RESOLUTION


# ---------------------------------------------------------------------------
# Grid sizing
# ---------------------------------------------------------------------------


class TestSetupSparamsGridSizing:
    def test_grid_shape_all_axes(self, minimal_setup):
        objs, arrays, config = minimal_setup
        pml_cells = _PML_LAYERS
        expected = tuple(round((_DOMAIN_SIZE[i] + 2 * pml_cells * _RESOLUTION) / _RESOLUTION) for i in range(3))
        assert objs.volume.grid_shape == expected  # (11, 8, 8)

    def test_pml_layers_scale_grid(self):
        def _grid(pml):
            objs, _, _ = _make_setup(input_ports=[], output_ports=[], pml_layers=pml)
            return objs.volume.grid_shape

        g3 = _grid(3)
        g5 = _grid(5)
        for i in range(3):
            assert g5[i] - g3[i] == 4  # 2 extra cells per face × 2 faces


# ---------------------------------------------------------------------------
# Object counts
# ---------------------------------------------------------------------------


class TestSetupSparamsObjectCounts:
    def test_six_pml_objects_always_created(self, minimal_setup):
        objs, arrays, config = minimal_setup
        assert len(objs.pml_objects) == 6

    def test_all_pml_objects_are_pml_type(self, minimal_setup):
        objs, arrays, config = minimal_setup
        for pml in objs.pml_objects:
            assert isinstance(pml, PerfectlyMatchedLayer)

    def test_one_input_one_output_total_objects(self, minimal_setup):
        objs, arrays, config = minimal_setup
        # 1 background + 6 PML + 1 source + 1 norm_det + 1 output_det = 10
        assert len(objs.object_list) == 10

    def test_source_count_matches_input_ports(self, two_port_setup):
        objs, arrays, config = two_port_setup
        assert len(objs.sources) == 2

    def test_detector_count_equals_input_plus_output(self, two_port_setup):
        objs, arrays, config = two_port_setup
        # 2 norm detectors (one per input) + 2 output detectors = 4
        assert len(objs.detectors) == 4


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


class TestSetupSparamsNaming:
    def test_named_source_name_preserved(self, minimal_setup):
        objs, arrays, config = minimal_setup
        assert "SrcA" in [s.name for s in objs.sources]

    def test_named_output_detector_name_preserved(self, minimal_setup):
        objs, arrays, config = minimal_setup
        assert "DetB" in [d.name for d in objs.detectors]

    def test_input_norm_detector_name_pattern(self, minimal_setup):
        objs, arrays, config = minimal_setup
        assert "SrcA_input_normalization" in [d.name for d in objs.detectors]

    def test_unnamed_input_gets_source_i_name(self):
        objs, _, _ = _make_setup(
            input_ports=[
                PortSpec(center=_PORT_CENTER_IN, axis=0, direction="+", width=_PORT_WIDTH, height=_PORT_HEIGHT),
                PortSpec(center=_PORT_CENTER_OUT, axis=0, direction="+", width=_PORT_WIDTH, height=_PORT_HEIGHT),
            ],
            output_ports=[],
        )
        source_names = [s.name for s in objs.sources]
        assert "Source_0" in source_names
        assert "Source_1" in source_names

    def test_unnamed_output_gets_detector_i_name(self):
        objs, _, _ = _make_setup(
            input_ports=[],
            output_ports=[
                PortSpec(center=_PORT_CENTER_IN, axis=0, direction="+", width=_PORT_WIDTH, height=_PORT_HEIGHT),
                PortSpec(center=_PORT_CENTER_OUT, axis=0, direction="+", width=_PORT_WIDTH, height=_PORT_HEIGHT),
            ],
        )
        det_names = [d.name for d in objs.detectors]
        assert "Detector_0" in det_names
        assert "Detector_1" in det_names

    def test_two_port_norm_detector_names(self, two_port_setup):
        objs, arrays, config = two_port_setup
        det_names = [d.name for d in objs.detectors]
        assert "P1_input_normalization" in det_names
        assert "P2_input_normalization" in det_names


# ---------------------------------------------------------------------------
# Port attributes
# ---------------------------------------------------------------------------


class TestSetupSparamsPortAttributes:
    def _single_source_setup(self, **port_kwargs):
        port = PortSpec(center=_PORT_CENTER_IN, axis=0, width=_PORT_WIDTH, height=_PORT_HEIGHT, **port_kwargs)
        objs, _, _ = _make_setup(input_ports=[port], output_ports=[])
        return objs

    def test_source_direction_preserved(self):
        objs = self._single_source_setup(direction="-", name="S")
        assert objs["S"].direction == "-"

    def test_source_mode_index_preserved(self):
        objs = self._single_source_setup(direction="+", name="S", mode_index=2)
        assert objs["S"].mode_index == 2

    def test_source_filter_pol_te(self):
        objs = self._single_source_setup(direction="+", name="S", filter_pol="te")
        assert objs["S"].filter_pol == "te"

    def test_source_filter_pol_none(self):
        objs = self._single_source_setup(direction="+", name="S", filter_pol=None)
        assert objs["S"].filter_pol is None

    def test_source_shape_from_make_port_shape(self):
        w, h = 0.4e-6, 0.6e-6
        port = PortSpec(center=_PORT_CENTER_IN, axis=0, direction="+", width=w, height=h, name="S")
        objs, _, _ = _make_setup(input_ports=[port], output_ports=[])
        expected = _make_port_shape(axis=0, resolution=_RESOLUTION, width=w, height=h)
        assert objs["S"].partial_real_shape == expected

    def test_norm_detector_same_shape_as_source(self, minimal_setup):
        objs, arrays, config = minimal_setup
        assert objs["SrcA_input_normalization"].partial_real_shape == objs["SrcA"].partial_real_shape


# ---------------------------------------------------------------------------
# ArrayContainer
# ---------------------------------------------------------------------------


class TestSetupSparamsArrayContainer:
    def test_e_field_shape_matches_grid(self, minimal_setup):
        objs, arrays, config = minimal_setup
        expected = (3,) + objs.volume.grid_shape
        assert arrays.E.shape == expected

    def test_h_field_shape_matches_grid(self, minimal_setup):
        objs, arrays, config = minimal_setup
        expected = (3,) + objs.volume.grid_shape
        assert arrays.H.shape == expected

    def test_detector_states_keys_match_detectors(self, minimal_setup):
        objs, arrays, config = minimal_setup
        expected = {d.name for d in objs.detectors}
        assert set(arrays.detector_states.keys()) == expected

    def test_fields_initialized_to_zero(self, minimal_setup):
        objs, arrays, config = minimal_setup
        assert jnp.all(arrays.E == 0)
        assert jnp.all(arrays.H == 0)

    def test_inv_permittivities_positive(self, minimal_setup):
        objs, arrays, config = minimal_setup
        assert jnp.all(arrays.inv_permittivities > 0)


# ---------------------------------------------------------------------------
# determine_input_norm_detector_name
# ---------------------------------------------------------------------------


class TestDetermineInputNormDetectorName:
    """Detectors in two_port_setup:
    P1_input_normalization, P2_input_normalization, P3, P4
    """

    def test_finds_p1_norm_detector(self, two_port_setup):
        objs, arrays, config = two_port_setup
        assert determine_input_norm_detector_name("P1", objs) == "P1_input_normalization"

    def test_finds_p2_norm_detector(self, two_port_setup):
        objs, arrays, config = two_port_setup
        assert determine_input_norm_detector_name("P2", objs) == "P2_input_normalization"

    def test_partial_suffix_unique_match(self, two_port_setup):
        # "P1_input" is a substring of "P1_input_normalization" only
        objs, arrays, config = two_port_setup
        assert determine_input_norm_detector_name("P1_input", objs) == "P1_input_normalization"

    def test_no_match_raises_exception(self, two_port_setup):
        objs, arrays, config = two_port_setup
        with pytest.raises(Exception, match="Cannot find"):
            determine_input_norm_detector_name("nonexistent_xyz_987", objs)

    def test_ambiguous_match_raises_exception(self, two_port_setup):
        # "_input_normalization" is a substring of both P1 and P2 norm detectors
        objs, arrays, config = two_port_setup
        with pytest.raises(Exception, match="Cannot uniquely determine"):
            determine_input_norm_detector_name("_input_normalization", objs)

    def test_output_only_detector_matched_by_name(self, two_port_setup):
        # Confirms that all ModeOverlapDetectors (not just norm ones) are searched
        objs, arrays, config = two_port_setup
        assert determine_input_norm_detector_name("P3", objs) == "P3"

    def test_returns_string(self, two_port_setup):
        objs, arrays, config = two_port_setup
        result = determine_input_norm_detector_name("P1", objs)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# calculate_sparam – fast ValueError guard (no FDTD run)
# ---------------------------------------------------------------------------


class TestCalculateSparamValidation:
    """calculate_sparam raises ValueError before any FDTD execution when the
    requested input_port_name does not match any source in the container.
    The no_source_setup fixture has zero sources, so the guard fires immediately.
    """

    def test_raises_value_error_for_unknown_port(self, no_source_setup):
        objs, arrays, config = no_source_setup
        with pytest.raises(ValueError, match="does not exist"):
            calculate_sparam(objs, arrays, config, input_port_name="NONEXISTENT")

    def test_error_message_contains_port_name(self, no_source_setup):
        objs, arrays, config = no_source_setup
        bad_name = "totally_missing_port_xyz"
        with pytest.raises(ValueError, match=bad_name):
            calculate_sparam(objs, arrays, config, input_port_name=bad_name)


# ---------------------------------------------------------------------------
# calculate_sparam – multi-source path (exercises the aset line)
# ---------------------------------------------------------------------------

# Dedicated constants for the multi-source FDTD fixture.
# The domain must be large enough in the transverse plane for the tidy3d
# ARPACK mode solver to converge (≥ 8×8 = 64 grid points per cross-section).
# Using a cubic 2 µm core (10 cells/side) with 8 µm-wide ports gives 8×8
# cross-sections, well above the ARPACK minimum.
_MC_RESOLUTION = 200e-9  # 200 nm – same resolution, larger domain
_MC_DOMAIN = (2e-6, 2e-6, 2e-6)  # 10×10×10 core cells
_MC_PML = 3  # total grid: 16×16×16 = 4096 cells
_MC_PORT_W = 1.6e-6  # 8-cell cross-section in y
_MC_PORT_H = 1.6e-6  # 8-cell cross-section in z
# dt ≈ 0.385 fs → 5 fs gives ≈ 13 time steps
_MC_MAX_TIME = 5e-15

# pml_thickness = 3 × 200 nm = 600 nm; background total = 3.2 µm per axis
# Port centres placed at y/z midpoint of background (1.6 µm) and at
# increasing x positions within the 3.2 µm background extent.
_MC_CY = 1.6e-6
_MC_CZ = 1.6e-6
_MC_XS = [1.1e-6, 1.6e-6, 2.1e-6, 2.6e-6]  # P1, P2, P3, P4


@pytest.fixture(scope="session")
def two_source_sparam_result():
    """Run calculate_sparam with 2 active sources.

    The source-disabling loop in calculate_sparam reaches the
    ``objects.aset("object_list->[N]->switch->is_always_off", True)`` line
    for every source that is NOT the requested input port.  With two sources
    (P1 active, P2 disabled) that line is executed once, which is the minimum
    needed to cover the fix.
    """
    names = ["P1", "P2", "P3", "P4"]
    ports = [
        PortSpec(
            center=(_MC_XS[i], _MC_CY, _MC_CZ),
            axis=0,
            direction="+",
            width=_MC_PORT_W,
            height=_MC_PORT_H,
            name=n,
        )
        for i, n in enumerate(names)
    ]
    objs, arrays, config = setup_sparams_simulation(
        polygons=[],
        input_ports=ports[:2],
        output_ports=ports[2:],
        wavelength=_WAVELENGTH,
        resolution=_MC_RESOLUTION,
        max_time=_MC_MAX_TIME,
        domain_size=_MC_DOMAIN,
        pml_layers=_MC_PML,
        key=jax.random.PRNGKey(0),
    )
    result, states = calculate_sparam(
        objs,
        arrays,
        config,
        input_port_name="P1",
        show_progress=False,
    )
    return result, states, objs


class TestCalculateSparamMultiSource:
    """Verify the source-disabling aset path works when 2+ sources are present.

    Each test uses the ``two_source_sparam_result`` session fixture which runs
    a minimal FDTD simulation (≈13 time steps, 704 cells) to ensure the aset
    line is exercised and the function returns a valid result.
    """

    def test_returns_without_error(self, two_source_sparam_result):
        # Simply constructing the fixture confirms no exception was raised
        result, states, objs = two_source_sparam_result
        assert result is not None

    def test_result_is_dict(self, two_source_sparam_result):
        result, states, objs = two_source_sparam_result
        assert isinstance(result, dict)

    def test_states_is_dict(self, two_source_sparam_result):
        result, states, objs = two_source_sparam_result
        assert isinstance(states, dict)

    def test_result_keys_are_two_tuples(self, two_source_sparam_result):
        result, states, objs = two_source_sparam_result
        for key in result:
            assert isinstance(key, tuple) and len(key) == 2

    def test_input_port_name_in_all_keys(self, two_source_sparam_result):
        result, states, objs = two_source_sparam_result
        for det_name, src_name in result:
            assert src_name == "P1"

    def test_all_detectors_appear_in_result(self, two_source_sparam_result):
        result, states, objs = two_source_sparam_result
        expected_dets = {d.name for d in objs.detectors}
        result_dets = {det_name for det_name, _ in result}
        assert result_dets == expected_dets

    def test_detector_states_returned_for_all_detectors(self, two_source_sparam_result):
        result, states, objs = two_source_sparam_result
        for det in objs.detectors:
            assert det.name in states

    def test_s_params_are_finite(self, two_source_sparam_result):
        import jax.numpy as jnp

        result, states, objs = two_source_sparam_result
        for (det_name, _), s_param in result.items():
            assert jnp.isfinite(jnp.abs(s_param)), f"S-param for {det_name!r} is not finite"


# ---------------------------------------------------------------------------
# calculate_sparams – multi-port wrapper
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sparams_wrapper_setup():
    """Simulation scene with 2 input ports (P1, P2) and 2 output ports (P3, P4).

    Reuses the same multi-source constants as two_source_sparam_result.
    """
    names = ["P1", "P2", "P3", "P4"]
    ports = [
        PortSpec(
            center=(_MC_XS[i], _MC_CY, _MC_CZ),
            axis=0,
            direction="+",
            width=_MC_PORT_W,
            height=_MC_PORT_H,
            name=n,
        )
        for i, n in enumerate(names)
    ]
    return setup_sparams_simulation(
        polygons=[],
        input_ports=ports[:2],
        output_ports=ports[2:],
        wavelength=_WAVELENGTH,
        resolution=_MC_RESOLUTION,
        max_time=_MC_MAX_TIME,
        domain_size=_MC_DOMAIN,
        pml_layers=_MC_PML,
        key=jax.random.PRNGKey(0),
    )


@pytest.fixture(scope="session")
def sparams_wrapper_result(sparams_wrapper_setup):
    """Call calculate_sparams with both input ports and return_detector_states=True."""
    objs, arrays, config = sparams_wrapper_setup
    result, states = calculate_sparams(
        objs,
        arrays,
        config,
        input_port_names=["P1", "P2"],
        show_progress=False,
        return_detector_states=True,
    )
    return result, states, objs


class TestCalculateSparamsWrapper:
    def test_returns_tuple(self, sparams_wrapper_result):
        result = sparams_wrapper_result[:2]
        assert isinstance(result, tuple) and len(result) == 2

    def test_merged_dict_contains_p1(self, sparams_wrapper_result):
        result, states, objs = sparams_wrapper_result
        src_names = {src_name for _, src_name in result}
        assert "P1" in src_names

    def test_merged_dict_contains_p2(self, sparams_wrapper_result):
        result, states, objs = sparams_wrapper_result
        src_names = {src_name for _, src_name in result}
        assert "P2" in src_names

    def test_merged_dict_key_count(self, sparams_wrapper_result):
        result, states, objs = sparams_wrapper_result
        n_detectors = len(objs.detectors)
        n_ports = 2
        assert len(result) == n_detectors * n_ports

    def test_detector_states_list_length(self, sparams_wrapper_result):
        result, states, objs = sparams_wrapper_result
        assert len(states) == 2

    def test_detector_states_are_dicts(self, sparams_wrapper_result):
        result, states, objs = sparams_wrapper_result
        for s in states:
            assert isinstance(s, dict)

    def test_no_detector_states_by_default(self, sparams_wrapper_setup):
        objs, arrays, config = sparams_wrapper_setup
        _, states = calculate_sparams(
            objs,
            arrays,
            config,
            input_port_names=["P1"],
            show_progress=False,
            return_detector_states=False,
        )
        assert states == []

    def test_empty_input_returns_empty_dict(self, sparams_wrapper_setup):
        objs, arrays, config = sparams_wrapper_setup
        result, states = calculate_sparams(
            objs,
            arrays,
            config,
            input_port_names=[],
            show_progress=False,
        )
        assert result == {}
        assert states == []

    def test_s_params_are_finite(self, sparams_wrapper_result):
        result, states, objs = sparams_wrapper_result
        for (det_name, _), s_param in result.items():
            assert jnp.isfinite(jnp.abs(s_param)), f"S-param for {det_name!r} is not finite"
