"""Tests for objects/detectors/field_projection.py - Field-projection angle detector."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx.objects.detectors.field_projection as field_projection_module
from fdtdx import FieldProjectionAngleDetector as ExportedFieldProjectionAngleDetector
from fdtdx import FieldProjectionCartesianDetector as ExportedFieldProjectionCartesianDetector
from fdtdx import FieldProjectionKSpaceDetector as ExportedFieldProjectionKSpaceDetector
from fdtdx import constants
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import QuasiUniformGrid, RectilinearGrid, UniformGrid
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.dispersion import DispersionModel, LorentzPole
from fdtdx.materials import Material
from fdtdx.objects.detectors.field_projection import (
    _PROJECTION_FIELD_KEYS,
    _PROJECTION_RESULT_KEYS,
    FieldProjectionAngleDetector,
    FieldProjectionCartesianDetector,
    FieldProjectionKSpaceDetector,
    _surface_axis_direction,
    _surface_state_key,
)


@pytest.fixture
def single_frequency():
    return [WaveCharacter(wavelength=0.689e-6)]


@pytest.fixture
def multiple_frequencies():
    return [WaveCharacter(wavelength=0.532e-6), WaveCharacter(wavelength=0.689e-6)]


UNIFORM_CONFIG = SimulationConfig(time=100e-15, grid=UniformGrid(spacing=50e-9), backend="cpu")
UNIFORM_CONFIG_80NM = SimulationConfig(time=100e-15, grid=UniformGrid(spacing=80e-9), backend="cpu")


def transverse_polarization(ux: float, uy: float) -> np.ndarray:
    radius = math.hypot(ux, uy)
    if radius < 1e-12:
        return np.asarray([1.0, 0.0, 0.0])
    return np.asarray([-uy / radius, ux / radius, 0.0])


def plane_wave_global_fields(
    theta_deg: float,
    phi_deg: float,
    refractive_index: float,
    impedance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    radial = np.asarray([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
    electric = transverse_polarization(radial[0], radial[1])
    magnetic = np.cross(radial, electric) / (impedance or constants.eta0 / refractive_index)
    return radial, electric, magnetic


def plane_wave_phasor_for_surface(
    detector: FieldProjectionAngleDetector,
    *,
    propagation_axis: int,
    local_axes: tuple[int, int, int],
    signs: tuple[int, int, int],
    local_coordinates: tuple[np.ndarray, np.ndarray, float],
    grid_shape: tuple[int, int, int],
    radial_global: np.ndarray,
    e_global: np.ndarray,
    h_global: np.ndarray,
    refractive_index: float,
    wave_character_index: int,
) -> jnp.ndarray:
    u_coords, v_coords, normal_offset = local_coordinates
    uu, vv = np.meshgrid(u_coords, v_coords, indexing="ij")
    wavelength = detector.wave_characters[wave_character_index].get_wavelength() / refractive_index
    radial_local = np.asarray([signs[i] * radial_global[axis] for i, axis in enumerate(local_axes)])
    phase = np.exp(
        1j * 2.0 * np.pi / wavelength * (radial_local[0] * uu + radial_local[1] * vv + radial_local[2] * normal_offset)
    )
    e_local = np.asarray([signs[i] * e_global[axis] for i, axis in enumerate(local_axes)])
    h_local = np.asarray([signs[i] * h_global[axis] for i, axis in enumerate(local_axes)])
    local_grids = (
        (e_local[:, None, None] * phase[None, :, :]) / constants.eta0,
        h_local[:, None, None] * phase[None, :, :],
    )

    phasor = np.zeros((len(detector.wave_characters), 6, *grid_shape), dtype=np.complex64)
    remaining_axes = [axis for axis in range(3) if axis != propagation_axis]
    inverse_spatial_order = np.argsort([remaining_axes.index(axis) for axis in local_axes[:2]])
    for local_component, global_axis in enumerate(local_axes):
        for field_offset, local_grid in zip((0, 3), local_grids):
            field = signs[local_component] * np.transpose(local_grid[local_component], inverse_spatial_order)
            phasor[wave_character_index, global_axis + field_offset] = np.expand_dims(field, axis=propagation_axis)
    return jnp.asarray(phasor[None, ...])


def make_detector_state_for_plane_wave(
    detector: FieldProjectionAngleDetector,
    *,
    theta_deg: float,
    phi_deg: float,
    refractive_index: float = 1.0,
    impedance: float | None = None,
    wave_character_index: int = 0,
) -> dict[str, jnp.ndarray]:
    radial, electric, magnetic = plane_wave_global_fields(theta_deg, phi_deg, refractive_index, impedance)
    return {
        "phasor": plane_wave_phasor_for_surface(
            detector,
            propagation_axis=detector.propagation_axis,
            local_axes=detector._local_axes()[0],
            signs=detector._local_axes()[1],
            local_coordinates=detector._local_coordinates(),
            grid_shape=detector.grid_shape,
            radial_global=radial,
            e_global=electric,
            h_global=magnetic,
            refractive_index=refractive_index,
            wave_character_index=wave_character_index,
        )
    }


def make_box_detector_state_for_plane_wave(
    detector: FieldProjectionAngleDetector,
    *,
    theta_deg: float,
    phi_deg: float,
    refractive_index: float = 1.0,
    impedance: float | None = None,
    wave_character_index: int = 0,
) -> dict[str, jnp.ndarray]:
    radial, electric, magnetic = plane_wave_global_fields(theta_deg, phi_deg, refractive_index, impedance)
    state = {}
    for surface in detector._included_box_surfaces():
        propagation_axis, direction = _surface_axis_direction(surface)
        state[_surface_state_key(surface)] = plane_wave_phasor_for_surface(
            detector,
            propagation_axis=propagation_axis,
            local_axes=detector._local_axes_for_surface(propagation_axis, direction)[0],
            signs=detector._local_axes_for_surface(propagation_axis, direction)[1],
            local_coordinates=detector._local_coordinates_for_surface(propagation_axis, direction),
            grid_shape=detector._surface_grid_shape(surface),
            radial_global=radial,
            e_global=electric,
            h_global=magnetic,
            refractive_index=refractive_index,
            wave_character_index=wave_character_index,
        )
    return state


def box_center(detector: FieldProjectionAngleDetector) -> tuple[float, float, float]:
    return tuple(float(0.5 * (detector._axis_centers(axis)[0] + detector._axis_centers(axis)[-1])) for axis in range(3))


def face_grid_slice(
    surface: str, shape: tuple[int, int, int]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    axis, direction = _surface_axis_direction(surface)
    ranges = [(0, shape[0]), (0, shape[1]), (0, shape[2])]
    ranges[axis] = (0, 1) if direction == "-" else (shape[axis] - 1, shape[axis])
    return tuple(ranges)


def passive_sqrt(value: complex) -> complex:
    root = complex(np.sqrt(complex(value)))
    if root.imag < 0.0 or (np.isclose(root.imag, 0.0) and root.real < 0.0):
        root = -root
    return root


def positive_impedance_sqrt(value: complex) -> complex:
    root = complex(np.sqrt(complex(value)))
    if root.real < 0.0 or (np.isclose(root.real, 0.0) and root.imag < 0.0):
        root = -root
    return root


def assert_power_peak_at(
    power: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    theta_deg: float,
    phi_deg: float,
) -> tuple[int, int]:
    peak_index = np.unravel_index(np.argmax(power), np.asarray(power).shape)
    measured_theta = float(np.rad2deg(theta[peak_index[0]]))
    measured_phi = float(np.rad2deg(phi[peak_index[1]]))
    phi_error = min(abs(measured_phi - phi_deg), abs(measured_phi - phi_deg - 360.0))
    assert abs(measured_theta - theta_deg) <= 1.0
    assert phi_error <= 1.0
    return peak_index


def assert_projected_fields_close(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
    keys: tuple[str, ...] = (*_PROJECTION_FIELD_KEYS, "power"),
) -> None:
    for key in keys:
        assert np.allclose(actual[key], expected[key], rtol=rtol, atol=atol)


def assert_project_all_matches_individuals(
    result: dict[str, np.ndarray],
    individuals: tuple[dict[str, np.ndarray], ...],
) -> None:
    expected_shape = (len(individuals), *np.asarray(individuals[0]["power"]).shape)
    for key in (*_PROJECTION_FIELD_KEYS, "power"):
        assert result[key].shape == expected_shape
        for index, individual in enumerate(individuals):
            assert np.allclose(result[key][index], individual[key])


def assert_frequency_peaks(
    result: dict[str, np.ndarray],
    theta: np.ndarray,
    phi: np.ndarray,
    expected_angles: tuple[tuple[float, float], ...],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        assert_power_peak_at(result["power"][index], theta, phi, theta_deg, phi_deg)
        for index, (theta_deg, phi_deg) in enumerate(expected_angles)
    )


def place_projection_detector(
    detector_class,
    wave_characters,
    random_key,
    grid_shape: tuple[int, int, int],
    *,
    config: SimulationConfig = UNIFORM_CONFIG,
    **kwargs,
):
    return detector_class(wave_characters=wave_characters, **kwargs).place_on_grid(
        tuple((0, size) for size in grid_shape),
        config,
        random_key,
    )


def combine_phasor_states(*states: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    return {"phasor": jnp.asarray(sum(np.asarray(state["phasor"]) for state in states))}


def combine_surface_states(*states: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    return {key: jnp.asarray(sum(np.asarray(state[key]) for state in states)) for key in states[0]}


def assert_project_all_matches_project(
    detector, state, coordinates: tuple[np.ndarray, np.ndarray]
) -> dict[str, np.ndarray]:
    result = detector.project_all(state, *coordinates)
    individual_results = [
        detector.project(state, *coordinates, wave_character_index=index)
        for index in range(len(detector.wave_characters))
    ]
    for key in _PROJECTION_RESULT_KEYS:
        for index, individual in enumerate(individual_results):
            assert np.allclose(result[key][index], individual[key])
    assert np.array_equal(result["wave_character_indices"], np.arange(len(individual_results)))
    return result


def sum_planar_face_projection_fields(
    box_detector: FieldProjectionAngleDetector,
    box_shape: tuple[int, int, int],
    random_key,
    theta: np.ndarray,
    phi: np.ndarray,
    theta_deg: float,
    phi_deg: float,
    *,
    config: SimulationConfig = UNIFORM_CONFIG,
    **detector_kwargs,
) -> dict[str, np.ndarray]:
    origin = box_center(box_detector)
    summed_fields = None
    for surface in box_detector._included_box_surfaces():
        _, direction = _surface_axis_direction(surface)
        planar_detector = FieldProjectionAngleDetector(
            wave_characters=box_detector.wave_characters,
            direction=direction,
            origin=origin,
            window_size=(0.0, 0.0),
            **detector_kwargs,
        )
        planar_detector = planar_detector.place_on_grid(face_grid_slice(surface, box_shape), config, random_key)
        planar_state = make_detector_state_for_plane_wave(planar_detector, theta_deg=theta_deg, phi_deg=phi_deg)
        planar_result = planar_detector.project(planar_state, theta, phi)
        if summed_fields is None:
            summed_fields = {key: np.zeros_like(planar_result[key]) for key in _PROJECTION_FIELD_KEYS}
        for key in _PROJECTION_FIELD_KEYS:
            summed_fields[key] = summed_fields[key] + planar_result[key]
    return summed_fields


class TestFieldProjectionAngleDetectorInit:
    def test_public_export(self):
        assert ExportedFieldProjectionAngleDetector is FieldProjectionAngleDetector
        assert ExportedFieldProjectionCartesianDetector is FieldProjectionCartesianDetector
        assert ExportedFieldProjectionKSpaceDetector is FieldProjectionKSpaceDetector

    def test_empty_wave_characters_raises(self):
        with pytest.raises(ValueError, match="wave_characters"):
            FieldProjectionAngleDetector(wave_characters=[], direction="+")

    def test_detector_interpolation_is_not_user_configurable(self, single_frequency):
        for argument in ["colocate", "use_colocated_integration", "exact_interpolation"]:
            with pytest.raises(TypeError, match=argument):
                FieldProjectionAngleDetector(
                    wave_characters=single_frequency,
                    direction="+",
                    **{argument: False},
                )

        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+")
        assert detector.exact_interpolation is True

    def test_invalid_angle_detector_parameters_raise(self, single_frequency):
        cases = [
            ("window_size", {"window_size": value})
            for value in [
                (0.1,),
                (-0.1, 0.0),
                (0.0, 1.1),
                (math.nan, 0.0),
                (math.inf, 0.0),
                ("bad", 0.0),
                ("0.1", 0.0),
                (True, 0.0),
            ]
        ]
        cases += [
            ("interval_space", {"interval_space": value})
            for value in [(1,), (1, 1), (0, 1, 1), (1.5, 1, 1), (True, 1, 1)]
        ]
        cases += [("direction", {"direction": "x"})]
        cases += [
            ("projection_distance", {"projection_distance": value})
            for value in [0.0, -1.0, math.nan, math.inf, "bad", "1.0", True]
        ]
        cases += [("far_field_approx", {"far_field_approx": value}) for value in [0, 1, "true", None]]
        cases += [
            ("exact_projection_batch_size", {"exact_projection_batch_size": value})
            for value in [0, -1, 1.5, True, "128"]
        ]
        cases += [
            ("origin", {"origin": value})
            for value in [
                (0.0, 0.0),
                (math.nan, 0.0, 0.0),
                (math.inf, 0.0, 0.0),
                ("bad", 0.0, 0.0),
                ("0.1", 0.0, 0.0),
                (True, 0.0, 0.0),
            ]
        ]
        for match, kwargs in cases:
            with pytest.raises(ValueError, match=match):
                FieldProjectionAngleDetector(**{"wave_characters": single_frequency, "direction": "+", **kwargs})

    def test_invalid_projection_medium_parameters_raise(self, single_frequency):
        cases = [
            (field_name, {field_name: value})
            for field_name in ["projection_medium_refractive_index", "projection_medium_impedance"]
            for value in [0.0, -1.0, math.nan, math.inf, "bad", "2.0", True, (1.0, 2.0), [0.0], [math.nan], [True]]
        ]
        cases += [
            ("window_size", {"window_size": (0.1 + 0.2j, 0.0)}),
            ("origin", {"origin": (0.0, 0.0 + 0.1j, 0.0)}),
            ("projection_medium_refractive_index", {"projection_medium_refractive_index": 1.0 + 0.0j}),
            ("projection_medium_impedance", {"projection_medium_impedance": constants.eta0 + 0.0j}),
        ]
        cases += [
            ("projection_medium", {"projection_medium": value})
            for value in [
                object(),
                Material(permittivity=(1.0, 2.0, 3.0)),
                Material(permittivity=(1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
                Material(permeability=(1.0, 1.2, 1.0)),
                Material(electric_conductivity=(0.0, 0.1, 0.0)),
                Material(magnetic_conductivity=(0.0, 0.0, 0.1)),
                Material(permittivity=0.0),
                Material(permeability=0.0),
                Material(permittivity=True),
                Material(permittivity=1.0 + 0.1j),
                Material(permittivity=math.inf),
                Material(electric_conductivity=-1.0),
                Material(magnetic_conductivity=-1.0),
            ]
        ]
        for match, kwargs in cases:
            with pytest.raises(ValueError, match=match):
                FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+", **kwargs)

    def test_projection_medium_rejects_manual_medium_parameters(self, single_frequency):
        for match, kwargs in [
            ("projection_medium_refractive_index", {"projection_medium_refractive_index": 1.5}),
            ("projection_medium_impedance", {"projection_medium_impedance": constants.eta0}),
        ]:
            with pytest.raises(ValueError, match=match):
                FieldProjectionAngleDetector(
                    wave_characters=single_frequency,
                    direction="+",
                    projection_medium=Material(permittivity=2.25),
                    **kwargs,
                )

    def test_invalid_coordinate_projection_axis_raises(self, single_frequency):
        for detector_class in [FieldProjectionCartesianDetector, FieldProjectionKSpaceDetector]:
            for projection_axis in [-1, 3, 0.5, True, "2"]:
                with pytest.raises(ValueError, match="projection_axis"):
                    detector_class(
                        wave_characters=single_frequency,
                        direction="+",
                        projection_axis=projection_axis,
                    )

    def test_project_rejects_invalid_coordinate_arrays(self, random_key, single_frequency):
        common = [
            ([[0.0]], [0.0], "one-dimensional"),
            ([0.0], [[0.0]], "one-dimensional"),
            ([], [0.0], "non-empty"),
            ([0.0], [], "non-empty"),
            ([math.nan], [0.0], "finite"),
            ([0.0], [math.inf], "finite"),
            ([True], [0.0], "finite numeric"),
        ]
        jax_finite = [
            (jnp.asarray([jnp.nan]), jnp.asarray([0.0]), "finite"),
            (jnp.asarray([0.0]), jnp.asarray([jnp.inf]), "finite"),
            (jnp.asarray([0.0 + 0.0j]), jnp.asarray([0.0]), "finite numeric"),
        ]
        detector_cases = [
            (
                FieldProjectionAngleDetector,
                [
                    (np.asarray(theta), np.asarray(phi), match)
                    for theta, phi, match in [
                        *common,
                        (["bad"], [0.0], "finite numeric"),
                        ([0.0], [False], "finite numeric"),
                        ([0.0], ["bad"], "finite numeric"),
                        ([-0.1], [0.0], r"\[0, pi\]"),
                        ([math.pi + 0.1], [0.0], r"\[0, pi\]"),
                    ]
                ]
                + [
                    (jnp.asarray([jnp.nan]), jnp.asarray([0.0]), "finite"),
                    (jnp.asarray([jnp.inf]), jnp.asarray([0.0]), "finite"),
                    (jnp.asarray([-0.1]), jnp.asarray([0.0]), r"\[0, pi\]"),
                    (jnp.asarray([jnp.pi + 0.1]), jnp.asarray([0.0]), r"\[0, pi\]"),
                    (jnp.asarray([0.1 + 0.0j]), jnp.asarray([0.0]), "finite numeric"),
                ],
            ),
            (
                FieldProjectionCartesianDetector,
                [
                    *common,
                    (["0.0"], [0.0], "finite numeric"),
                    (["bad"], [0.0], "finite numeric"),
                    (["0.1"], [0.0], "finite numeric"),
                    ([0.0], [False], "finite numeric"),
                    ([0.0], ["0.1"], "finite numeric"),
                    *jax_finite,
                ],
            ),
            (
                FieldProjectionKSpaceDetector,
                [
                    *common,
                    (["0.0"], [0.0], "finite numeric"),
                    (["bad"], [0.0], "finite numeric"),
                    ([1.1], [0.0], r"\[-1, 1\]"),
                    ([0.8], [0.8], r"ux\^2 \+ uy\^2"),
                    *jax_finite,
                    (jnp.asarray([1.1]), jnp.asarray([0.0]), r"\[-1, 1\]"),
                    (jnp.asarray([0.8]), jnp.asarray([0.8]), r"ux\^2 \+ uy\^2"),
                ],
            ),
        ]
        for detector_class, invalid_coordinates in detector_cases:
            detector = detector_class(wave_characters=single_frequency, direction="+")
            detector = detector.place_on_grid(((0, 8), (0, 8), (0, 1)), UNIFORM_CONFIG, random_key)
            state = make_detector_state_for_plane_wave(detector, theta_deg=0.0, phi_deg=0.0)
            for coordinate_0, coordinate_1, match in invalid_coordinates:
                with pytest.raises(ValueError, match=match):
                    detector.project(state, coordinate_0, coordinate_1)

    def test_jit_project_rejects_complex_coordinate_tracer(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+")
        detector = detector.place_on_grid(((0, 8), (0, 8), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(detector, theta_deg=0.0, phi_deg=0.0)

        def project_power(theta):
            return detector.project(state, theta, jnp.asarray([0.0]))["power"]

        with pytest.raises(ValueError, match="finite numeric"):
            jax.jit(project_power)(jnp.asarray([0.1 + 0.0j]))


class TestFieldProjectionAngleDetector:
    def test_shape_dtype_single_frequency(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+")
        detector = detector.place_on_grid(((0, 24), (0, 24), (0, 1)), UNIFORM_CONFIG, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert shape_dtype["phasor"].shape == (1, 6, 24, 24, 1)
        assert detector.propagation_axis == 2

    def test_box_shape_dtype_records_only_included_surfaces(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            exclude_surfaces=("z-",),
            interval_space=(2, 3, 4),
        )
        detector = detector.place_on_grid(((0, 5), (0, 6), (0, 7)), UNIFORM_CONFIG, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert set(shape_dtype) == {
            "phasor_x_minus",
            "phasor_x_plus",
            "phasor_y_minus",
            "phasor_y_plus",
            "phasor_z_plus",
        }
        assert shape_dtype["phasor_x_minus"].shape == (1, 6, 1, 6, 7)
        assert shape_dtype["phasor_x_plus"].shape == (1, 6, 1, 6, 7)
        assert shape_dtype["phasor_y_minus"].shape == (1, 6, 5, 1, 7)
        assert shape_dtype["phasor_z_plus"].shape == (1, 6, 5, 6, 1)

    def test_box_update_records_only_surface_phasors(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, exclude_surfaces=("x-", "z+"))
        detector = detector.place_on_grid(((0, 5), (0, 6), (0, 7)), UNIFORM_CONFIG, random_key)
        state = detector.init_state()
        base_field = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
        fields_e = jnp.asarray(np.stack([base_field + component * 1000.0 for component in range(3)]))
        fields_h = jnp.asarray(np.stack([base_field + (component + 3) * 1000.0 for component in range(3)]))

        updated = detector.update(
            time_step=jnp.asarray(0),
            E=fields_e,
            H=fields_h,
            state=state,
            inv_permittivity=jnp.ones((3, 5, 6, 7), dtype=jnp.float32),
            inv_permeability=1.0,
        )

        assert set(updated) == {"phasor_x_plus", "phasor_y_minus", "phasor_y_plus", "phasor_z_minus"}
        assert updated["phasor_x_plus"].shape == (1, 1, 6, 1, 6, 7)
        assert updated["phasor_y_minus"].shape == (1, 1, 6, 5, 1, 7)
        assert "phasor_x_minus" not in updated
        assert "phasor_z_plus" not in updated
        expected_fields = np.concatenate([np.asarray(fields_e), np.asarray(fields_h)], axis=0)
        static_scale = 2.0 / detector.num_time_steps_recorded
        for surface in ("x+", "y-", "y+", "z-"):
            axis, direction = _surface_axis_direction(surface)
            face_slices = [slice(None), slice(None), slice(None), slice(None)]
            face_slices[axis + 1] = slice(0, 1) if direction == "-" else slice(detector.grid_shape[axis] - 1, None)
            expected = expected_fields[tuple(face_slices)] * static_scale
            assert np.allclose(np.asarray(updated[_surface_state_key(surface)])[0, 0], expected)

    def test_surface_update_uses_single_phasor_state(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+")
        detector = detector.place_on_grid(((0, 5), (0, 6), (0, 1)), UNIFORM_CONFIG, random_key)
        state = detector.init_state()
        base_field = np.arange(5 * 6, dtype=np.float32).reshape(5, 6, 1)
        fields_e = jnp.asarray(np.stack([base_field + component * 100.0 for component in range(3)]))
        fields_h = jnp.asarray(np.stack([base_field + (component + 3) * 100.0 for component in range(3)]))

        updated = detector.update(
            time_step=jnp.asarray(0),
            E=fields_e,
            H=fields_h,
            state=state,
            inv_permittivity=jnp.ones((3, 5, 6, 1), dtype=jnp.float32),
            inv_permeability=1.0,
        )

        expected_fields = np.concatenate([np.asarray(fields_e), np.asarray(fields_h)], axis=0)
        static_scale = 2.0 / detector.num_time_steps_recorded
        assert set(updated) == {"phasor"}
        assert updated["phasor"].shape == (1, 1, 6, 5, 6, 1)
        assert np.allclose(np.asarray(updated["phasor"])[0, 0], expected_fields * static_scale)

    def test_box_inverse_update_subtracts_surface_phasors(self, random_key, single_frequency):
        forward_detector = FieldProjectionAngleDetector(wave_characters=single_frequency, exclude_surfaces=("z+",))
        inverse_detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            exclude_surfaces=("z+",),
            inverse=True,
        )
        grid_slice = ((0, 4), (0, 5), (0, 6))
        forward_detector = forward_detector.place_on_grid(grid_slice, UNIFORM_CONFIG, random_key)
        inverse_detector = inverse_detector.place_on_grid(grid_slice, UNIFORM_CONFIG, random_key)
        state = forward_detector.init_state()
        fields_e = jnp.ones((3, 4, 5, 6), dtype=jnp.float32)
        fields_h = 2.0 * jnp.ones((3, 4, 5, 6), dtype=jnp.float32)
        update_kwargs = dict(
            time_step=jnp.asarray(0),
            E=fields_e,
            H=fields_h,
            state=state,
            inv_permittivity=jnp.ones((3, 4, 5, 6), dtype=jnp.float32),
            inv_permeability=1.0,
        )

        forward = forward_detector.update(**update_kwargs)
        inverse = inverse_detector.update(**update_kwargs)

        for surface in forward:
            assert np.allclose(np.asarray(inverse[surface]), -np.asarray(forward[surface]))

    def test_box_pulse_update_does_not_apply_continuous_scaling(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            exclude_surfaces=("z+",),
            scaling_mode="pulse",
        )
        detector = detector.place_on_grid(((0, 4), (0, 5), (0, 6)), UNIFORM_CONFIG, random_key)
        state = detector.init_state()
        fields_e = jnp.ones((3, 4, 5, 6), dtype=jnp.float32)
        fields_h = 2.0 * jnp.ones((3, 4, 5, 6), dtype=jnp.float32)

        updated = detector.update(
            time_step=jnp.asarray(0),
            E=fields_e,
            H=fields_h,
            state=state,
            inv_permittivity=jnp.ones((3, 4, 5, 6), dtype=jnp.float32),
            inv_permeability=1.0,
        )

        expected_fields = np.concatenate([np.asarray(fields_e), np.asarray(fields_h)], axis=0)
        for surface in ("x-", "x+", "y-", "y+", "z-"):
            state_key = _surface_state_key(surface)
            assert state_key in updated
            face_slices = [slice(None), slice(None), slice(None), slice(None)]
            axis, direction = _surface_axis_direction(surface)
            face_slices[axis + 1] = slice(0, 1) if direction == "-" else slice(detector.grid_shape[axis] - 1, None)
            expected = expected_fields[tuple(face_slices)]
            assert np.allclose(np.asarray(updated[state_key])[0, 0], expected)

    def test_rejects_line_and_point_shapes(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+")

        with pytest.raises(Exception, match="Expected a single planar surface or a box volume"):
            detector.place_on_grid(((0, 1), (0, 4), (0, 1)), UNIFORM_CONFIG, random_key)
        with pytest.raises(Exception, match="Expected a single planar surface or a box volume"):
            detector.place_on_grid(((0, 1), (0, 1), (0, 1)), UNIFORM_CONFIG, random_key)

    def test_surface_placement_requires_direction(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency)

        with pytest.raises(ValueError, match="direction"):
            detector.place_on_grid(((0, 16), (0, 16), (0, 1)), UNIFORM_CONFIG, random_key)

    def test_box_placement_rejects_direction_window_and_surface_exclusion(self, random_key, single_frequency):
        with pytest.raises(ValueError, match="direction"):
            FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+").place_on_grid(
                ((0, 5), (0, 6), (0, 7)), UNIFORM_CONFIG, random_key
            )
        with pytest.raises(ValueError, match="window_size"):
            FieldProjectionAngleDetector(wave_characters=single_frequency, window_size=(0.1, 0.0)).place_on_grid(
                ((0, 5), (0, 6), (0, 7)), UNIFORM_CONFIG, random_key
            )
        with pytest.raises(ValueError, match="exclude_surfaces"):
            FieldProjectionAngleDetector(
                wave_characters=single_frequency, direction="+", exclude_surfaces=("z-",)
            ).place_on_grid(((0, 16), (0, 16), (0, 1)), UNIFORM_CONFIG, random_key)

    def test_box_detector_rejects_planar_axis_access(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency)
        detector = detector.place_on_grid(((0, 4), (0, 5), (0, 6)), UNIFORM_CONFIG, random_key)

        with pytest.raises(Exception, match="surface shape"):
            _ = detector.propagation_axis

    def test_invalid_exclude_surfaces_raises(self, single_frequency):
        for exclude_surfaces in [
            ("bad",),
            ("z-", "z-"),
            ("x-", "x+", "y-", "y+", "z-", "z+"),
            "z-",
            1,
        ]:
            with pytest.raises(ValueError, match="exclude_surfaces"):
                FieldProjectionAngleDetector(wave_characters=single_frequency, exclude_surfaces=exclude_surfaces)

    def test_project_rejects_invalid_angle_arrays(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+")
        detector = detector.place_on_grid(((0, 16), (0, 16), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(detector, theta_deg=0.0, phi_deg=0.0)

        for theta, phi, match in [
            ([[0.0]], [0.0], "one-dimensional"),
            ([0.0], [[0.0]], "one-dimensional"),
            ([], [0.0], "non-empty"),
            ([0.0], [], "non-empty"),
            ([math.nan], [0.0], "finite"),
            ([0.0], [math.inf], "finite"),
            ([True], [0.0], "finite numeric"),
            (["bad"], [0.0], "finite numeric"),
            ([0.0], [False], "finite numeric"),
            ([0.0], ["bad"], "finite numeric"),
            ([-0.1], [0.0], r"\[0, pi\]"),
            ([math.pi + 0.1], [0.0], r"\[0, pi\]"),
        ]:
            with pytest.raises(ValueError, match=match):
                detector.project(state, np.asarray(theta), np.asarray(phi))

    def test_project_rejects_invalid_wave_character_index(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+")
        detector = detector.place_on_grid(((0, 16), (0, 16), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(detector, theta_deg=0.0, phi_deg=0.0)

        for wave_character_index in [-1, 1, 0.5]:
            with pytest.raises(ValueError, match="wave_character_index"):
                detector.project(state, np.asarray([0.0]), np.asarray([0.0]), wave_character_index=wave_character_index)

    def test_project_rejects_missing_phasor_state(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+")
        detector = detector.place_on_grid(((0, 16), (0, 16), (0, 1)), UNIFORM_CONFIG, random_key)

        with pytest.raises(ValueError, match="phasor"):
            detector.project({}, np.asarray([0.0]), np.asarray([0.0]))

    def test_project_rejects_invalid_phasor_state_shape(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+")
        detector = detector.place_on_grid(((0, 16), (0, 16), (0, 1)), UNIFORM_CONFIG, random_key)

        for bad_shape in [
            (2, 1, 6, 16, 16, 1),
            (1, 2, 6, 16, 16, 1),
            (1, 1, 5, 16, 16, 1),
            (1, 1, 6, 15, 16, 1),
            (1, 1, 6, 16, 16),
        ]:
            state = {"phasor": jnp.zeros(bad_shape, dtype=jnp.complex64)}
            with pytest.raises(ValueError, match=r"state\['phasor'\] must have shape"):
                detector.project(state, np.asarray([0.0]), np.asarray([0.0]))

    def test_box_project_rejects_missing_surface_state(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, exclude_surfaces=("z+",))
        detector = detector.place_on_grid(((0, 4), (0, 5), (0, 6)), UNIFORM_CONFIG, random_key)
        state = make_box_detector_state_for_plane_wave(detector, theta_deg=0.0, phi_deg=0.0)
        missing_key = _surface_state_key("x-")
        incomplete_state = dict(state)
        incomplete_state.pop(missing_key)

        with pytest.raises(ValueError, match=missing_key):
            detector.project(incomplete_state, np.asarray([0.0]), np.asarray([0.0]))

    def test_box_project_rejects_invalid_surface_state_shape(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, exclude_surfaces=("z+",))
        detector = detector.place_on_grid(((0, 4), (0, 5), (0, 6)), UNIFORM_CONFIG, random_key)
        state = make_box_detector_state_for_plane_wave(detector, theta_deg=0.0, phi_deg=0.0)
        bad_key = _surface_state_key("x-")
        bad_state = dict(state)
        bad_state[bad_key] = state[bad_key][..., :-1]

        with pytest.raises(ValueError, match=rf"state\['{bad_key}'\] must have shape"):
            detector.project(bad_state, np.asarray([0.0]), np.asarray([0.0]))

    def test_project_z_plane_wave_peak(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            window_size=(0.0, 0.0),
        )
        detector = detector.place_on_grid(((0, 24), (0, 24), (0, 1)), UNIFORM_CONFIG, random_key)

        for theta_deg, phi_deg in [(15.0, 40.0), (35.0, 40.0)]:
            state = make_detector_state_for_plane_wave(detector, theta_deg=theta_deg, phi_deg=phi_deg)
            theta = np.deg2rad(np.asarray([theta_deg - 1.0, theta_deg, theta_deg + 1.0]))
            phi = np.deg2rad(np.asarray([phi_deg - 1.0, phi_deg, phi_deg + 1.0]))
            result = detector.project(state, theta, phi)
            assert_power_peak_at(result["power"], theta, phi, theta_deg, phi_deg)

    def test_box_single_face_matches_planar_projection(self, random_key, single_frequency):
        box_shape = (4, 5, 6)
        theta = np.deg2rad(np.asarray([19.0, 21.0, 23.0]))
        phi = np.deg2rad(np.asarray([31.0, 33.0, 35.0]))

        for surface in ("x-", "x+", "y-", "y+", "z-", "z+"):
            excluded_surfaces = tuple(
                candidate for candidate in ("x-", "x+", "y-", "y+", "z-", "z+") if candidate != surface
            )
            box_detector = FieldProjectionAngleDetector(
                wave_characters=single_frequency,
                exclude_surfaces=excluded_surfaces,
            )
            box_detector = box_detector.place_on_grid(
                ((0, box_shape[0]), (0, box_shape[1]), (0, box_shape[2])), UNIFORM_CONFIG, random_key
            )
            origin = box_center(box_detector)
            propagation_axis, direction = _surface_axis_direction(surface)
            planar_detector = FieldProjectionAngleDetector(
                wave_characters=single_frequency,
                direction=direction,
                origin=origin,
                window_size=(0.0, 0.0),
            )
            planar_detector = planar_detector.place_on_grid(
                face_grid_slice(surface, box_shape), UNIFORM_CONFIG, random_key
            )
            box_state = make_box_detector_state_for_plane_wave(box_detector, theta_deg=21.0, phi_deg=33.0)
            planar_state = make_detector_state_for_plane_wave(planar_detector, theta_deg=21.0, phi_deg=33.0)
            box_result = box_detector.project(box_state, theta, phi)
            planar_result = planar_detector.project(planar_state, theta, phi)

            assert box_result["surfaces"] == (surface,)
            assert planar_result["surfaces"] == (surface,)
            assert propagation_axis in [0, 1, 2]
            assert_projected_fields_close(
                box_result,
                planar_result,
                rtol=2e-6,
                atol=1e-8,
                keys=("Etheta", "Ephi", "Htheta", "Hphi", "power"),
            )

    def test_box_projection_is_coherent_sum_of_surfaces(self, random_key, single_frequency):
        cases = [
            dict(
                box_shape=(4, 5, 6),
                config=UNIFORM_CONFIG,
                angles=(24.0, 35.0),
                theta=[22.0, 24.0, 26.0],
                phi=[33.0, 35.0, 37.0],
                rtol=2e-6,
                atol=1e-8,
            ),
            dict(
                box_shape=(2, 2, 2),
                config=UNIFORM_CONFIG_80NM,
                angles=(25.0, 35.0),
                theta=[25.0],
                phi=[35.0],
                detector_kwargs={"projection_distance": 2.0e-6, "far_field_approx": False},
                rtol=1e-12,
                atol=1e-18,
            ),
        ]
        for case in cases:
            detector_kwargs = case.get("detector_kwargs", {})
            theta_deg, phi_deg = case["angles"]
            box_shape = case["box_shape"]
            full_box = FieldProjectionAngleDetector(
                wave_characters=single_frequency,
                exclude_surfaces=("x-", "y-", "y+", "z-"),
                **detector_kwargs,
            )
            full_box = full_box.place_on_grid(
                ((0, box_shape[0]), (0, box_shape[1]), (0, box_shape[2])), case["config"], random_key
            )
            theta = np.deg2rad(case["theta"])
            phi = np.deg2rad(case["phi"])
            full_state = make_box_detector_state_for_plane_wave(full_box, theta_deg=theta_deg, phi_deg=phi_deg)
            full_result = full_box.project(full_state, theta, phi)
            summed_fields = sum_planar_face_projection_fields(
                full_box,
                box_shape,
                random_key,
                theta,
                phi,
                theta_deg,
                phi_deg,
                config=case["config"],
                **detector_kwargs,
            )
            expected_power = 0.5 * np.real(
                summed_fields["Etheta"] * np.conj(summed_fields["Hphi"])
                - summed_fields["Ephi"] * np.conj(summed_fields["Htheta"])
            )

            assert full_result["surfaces"] == ("x+", "z+")
            for key in _PROJECTION_FIELD_KEYS:
                assert np.allclose(full_result[key], summed_fields[key], rtol=case["rtol"], atol=case["atol"])
            assert np.allclose(full_result["power"], expected_power, rtol=case["rtol"], atol=case["atol"])

    def test_box_projection_respects_exclude_surfaces(self, random_key, single_frequency):
        box_shape = (4, 5, 6)
        full_box = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            exclude_surfaces=("x-", "y-", "y+", "z-"),
        )
        full_box = full_box.place_on_grid(
            ((0, box_shape[0]), (0, box_shape[1]), (0, box_shape[2])), UNIFORM_CONFIG, random_key
        )
        origin = box_center(full_box)
        open_box = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            exclude_surfaces=("x-", "y-", "y+", "z-", "z+"),
        )
        open_box = open_box.place_on_grid(
            ((0, box_shape[0]), (0, box_shape[1]), (0, box_shape[2])), UNIFORM_CONFIG, random_key
        )
        z_plus_detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            origin=origin,
            window_size=(0.0, 0.0),
        )
        z_plus_detector = z_plus_detector.place_on_grid(face_grid_slice("z+", box_shape), UNIFORM_CONFIG, random_key)
        theta = np.deg2rad(np.asarray([21.0, 23.0, 25.0]))
        phi = np.deg2rad(np.asarray([29.0, 31.0, 33.0]))

        full_result = full_box.project(
            make_box_detector_state_for_plane_wave(full_box, theta_deg=23.0, phi_deg=31.0), theta, phi
        )
        open_result = open_box.project(
            make_box_detector_state_for_plane_wave(open_box, theta_deg=23.0, phi_deg=31.0), theta, phi
        )
        z_plus_result = z_plus_detector.project(
            make_detector_state_for_plane_wave(z_plus_detector, theta_deg=23.0, phi_deg=31.0),
            theta,
            phi,
        )

        expected_e_theta = full_result["Etheta"] - z_plus_result["Etheta"]
        expected_e_phi = full_result["Ephi"] - z_plus_result["Ephi"]
        expected_h_theta = full_result["Htheta"] - z_plus_result["Htheta"]
        expected_h_phi = full_result["Hphi"] - z_plus_result["Hphi"]
        expected_power = 0.5 * np.real(
            expected_e_theta * np.conj(expected_h_phi) - expected_e_phi * np.conj(expected_h_theta)
        )
        assert open_result["surfaces"] == ("x+",)
        assert np.allclose(open_result["Etheta"], expected_e_theta, rtol=2e-6, atol=1e-8)
        assert np.allclose(open_result["Ephi"], expected_e_phi, rtol=2e-6, atol=1e-8)
        assert np.allclose(open_result["power"], expected_power, rtol=2e-6, atol=1e-8)

    def test_interval_space_preserves_plane_wave_peak(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            interval_space=(3, 4, 1),
            window_size=(0.0, 0.0),
        )
        detector = detector.place_on_grid(((0, 24), (0, 24), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(detector, theta_deg=20.0, phi_deg=35.0)
        theta = np.deg2rad(np.asarray([19.0, 20.0, 21.0]))
        phi = np.deg2rad(np.asarray([34.0, 35.0, 36.0]))

        result = detector.project(state, theta, phi)

        assert_power_peak_at(result["power"], theta, phi, 20.0, 35.0)

    def test_window_size_tapers_planar_projection_power(self, random_key, single_frequency):
        grid_slice = ((0, 24), (0, 24), (0, 1))
        untapered = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            window_size=(0.0, 0.0),
        )
        tapered = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            window_size=(0.5, 0.5),
        )
        untapered = untapered.place_on_grid(grid_slice, UNIFORM_CONFIG, random_key)
        tapered = tapered.place_on_grid(grid_slice, UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(untapered, theta_deg=20.0, phi_deg=35.0)
        theta = np.deg2rad(np.asarray([20.0]))
        phi = np.deg2rad(np.asarray([35.0]))

        untapered_power = untapered.project(state, theta, phi)["power"][0, 0]
        tapered_power = tapered.project(state, theta, phi)["power"][0, 0]

        assert tapered_power > 0.0
        assert tapered_power < untapered_power

    def test_project_result_shapes_match_angle_grid(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+")
        detector = detector.place_on_grid(((0, 6), (0, 6), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(detector, theta_deg=12.0, phi_deg=25.0)
        theta = np.deg2rad(np.asarray([0.0, 12.0, 30.0]))
        phi = np.deg2rad(np.asarray([0.0, 25.0, 60.0]))

        result = detector.project(state, theta, phi)

        expected_shape = (theta.size, phi.size)
        for key in ["Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi", "power"]:
            assert result[key].shape == expected_shape
        assert np.all(result["Er"] == 0)
        assert np.all(result["Hr"] == 0)
        assert np.allclose(result["theta"], theta)
        assert np.allclose(result["phi"], phi)

    def test_exact_projection_raises_when_observation_overlaps_source_point(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_distance=50e-9,
            far_field_approx=False,
            window_size=(0.0, 0.0),
        )
        detector = detector.place_on_grid(((0, 3), (0, 3), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(detector, theta_deg=0.0, phi_deg=0.0)

        with pytest.raises(ValueError, match="observation points"):
            detector.project(state, np.asarray([math.pi / 2.0]), np.asarray([0.0]))

    def test_exact_projection_converges_to_far_field_approximation(self, random_key, single_frequency):
        projection_distance = 1.0e-3
        far_detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_distance=projection_distance,
            far_field_approx=True,
            window_size=(0.0, 0.0),
        )
        exact_detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_distance=projection_distance,
            far_field_approx=False,
            window_size=(0.0, 0.0),
        )
        far_detector = far_detector.place_on_grid(((0, 6), (0, 6), (0, 1)), UNIFORM_CONFIG, random_key)
        exact_detector = exact_detector.place_on_grid(((0, 6), (0, 6), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(far_detector, theta_deg=20.0, phi_deg=35.0)
        theta = np.deg2rad(np.asarray([20.0]))
        phi = np.deg2rad(np.asarray([35.0]))

        far_result = far_detector.project(state, theta, phi)
        exact_result = exact_detector.project(state, theta, phi)

        exact_transverse_e = np.hypot(abs(exact_result["Etheta"]), abs(exact_result["Ephi"]))
        exact_transverse_h = np.hypot(abs(exact_result["Htheta"]), abs(exact_result["Hphi"]))
        assert np.allclose(exact_result["power"], far_result["power"], rtol=1e-5, atol=1e-16)
        assert np.max(abs(exact_result["Er"]) / exact_transverse_e) < 1.0e-4
        assert np.max(abs(exact_result["Hr"]) / exact_transverse_h) < 1.0e-4

    def test_exact_projection_uses_custom_projection_medium_impedance(self, random_key, single_frequency):
        impedance = constants.eta0 / 2.5
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_distance=4.0e-6,
            far_field_approx=False,
            projection_medium_refractive_index=1.8,
            projection_medium_impedance=impedance,
            window_size=(0.0, 0.0),
        )
        detector = detector.place_on_grid(((0, 4), (0, 4), (0, 1)), UNIFORM_CONFIG_80NM, random_key)
        state = make_detector_state_for_plane_wave(
            detector,
            theta_deg=20.0,
            phi_deg=35.0,
            refractive_index=1.8,
            impedance=impedance,
        )

        result = detector.project(state, np.deg2rad([70.0]), np.deg2rad([90.0]))

        electric_transverse = np.hypot(abs(result["Etheta"][0, 0]), abs(result["Ephi"][0, 0]))
        magnetic_transverse = np.hypot(abs(result["Htheta"][0, 0]), abs(result["Hphi"][0, 0]))
        eps_complex, mu_complex = detector._projection_relative_permittivity_permeability(0)
        assert np.isclose(eps_complex, constants.eta0 * 1.8 / impedance)
        assert np.isclose(mu_complex, 1.8 * impedance / constants.eta0)
        assert result["far_field_approx"] is False
        assert result["projection_medium_impedance"] == impedance
        assert result["power"][0, 0] > 0.0
        assert abs(result["Er"][0, 0]) > 0.02 * electric_transverse
        assert abs(result["Hr"][0, 0]) > 0.01 * magnetic_transverse

    def test_exact_projection_chunked_matches_unchunked_projection(self, random_key, single_frequency):
        unchunked_detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_distance=2.0e-6,
            far_field_approx=False,
            exact_projection_batch_size=None,
            window_size=(0.0, 0.0),
        )
        chunked_detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_distance=2.0e-6,
            far_field_approx=False,
            exact_projection_batch_size=2,
            window_size=(0.0, 0.0),
        )
        unchunked_detector = unchunked_detector.place_on_grid(((0, 4), (0, 4), (0, 1)), UNIFORM_CONFIG, random_key)
        chunked_detector = chunked_detector.place_on_grid(((0, 4), (0, 4), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(unchunked_detector, theta_deg=20.0, phi_deg=35.0)
        theta = np.deg2rad(np.asarray([20.0, 28.0]))
        phi = np.deg2rad(np.asarray([35.0, 45.0]))

        unchunked_result = unchunked_detector.project(state, theta, phi)
        chunked_result = chunked_detector.project(state, theta, phi)

        assert_projected_fields_close(chunked_result, unchunked_result, rtol=5e-6, atol=1e-8)
        assert np.allclose(chunked_result["power"], unchunked_result["power"], rtol=5e-6, atol=1e-11)

    def test_project_far_field_planar_is_jittable_and_differentiable(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            window_size=(0.0, 0.0),
        )
        detector = detector.place_on_grid(((0, 8), (0, 8), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(detector, theta_deg=20.0, phi_deg=35.0)
        theta = jnp.deg2rad(jnp.asarray([20.0]))
        phi = jnp.deg2rad(jnp.asarray([35.0]))

        result = detector.project(state, theta, phi)
        assert isinstance(result["power"], jax.Array)

        phasor = state["phasor"]

        def loss(scale, theta, phi):
            scaled_state = {"phasor": phasor * scale}
            projected = detector.project(scaled_state, theta, phi)
            return jnp.real(jnp.sum(projected["power"]))

        eager_loss = loss(jnp.asarray(1.0), theta, phi)
        jit_loss = jax.jit(loss)(jnp.asarray(1.0), theta, phi)
        grad = jax.grad(loss)(jnp.asarray(1.0), theta, phi)

        assert jnp.all(jnp.isfinite(jit_loss))
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.allclose(jit_loss, eager_loss, rtol=1e-6, atol=1e-12)
        assert jnp.abs(grad) > 0.0

    def test_project_far_field_large_jax_coordinate_grid_is_jittable(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            window_size=(0.0, 0.0),
        )
        detector = detector.place_on_grid(((0, 8), (0, 8), (0, 1)), UNIFORM_CONFIG_80NM, random_key)
        state = make_detector_state_for_plane_wave(detector, theta_deg=20.0, phi_deg=35.0)
        theta = jnp.linspace(0.0, 0.8, 7)
        phi = jnp.linspace(0.0, 2 * jnp.pi, 13)

        def project_power(phasor):
            projected = detector.project({"phasor": phasor}, theta, phi)
            return projected["power"]

        eager = project_power(state["phasor"])
        jitted = jax.jit(project_power)(state["phasor"])

        assert isinstance(jitted, jax.Array)
        assert jitted.shape == (theta.size, phi.size)
        assert jnp.allclose(jitted, eager, rtol=1e-6, atol=1e-12)

    def test_project_jax_coordinate_arrays_do_not_use_numpy_asarray(
        self,
        monkeypatch,
        random_key,
        single_frequency,
    ):
        angle_detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            window_size=(0.0, 0.0),
        )
        cartesian_detector = FieldProjectionCartesianDetector(
            wave_characters=single_frequency,
            direction="+",
            window_size=(0.0, 0.0),
        )
        kspace_detector = FieldProjectionKSpaceDetector(
            wave_characters=single_frequency,
            direction="+",
            window_size=(0.0, 0.0),
        )
        grid_slice = ((0, 8), (0, 8), (0, 1))
        angle_detector = angle_detector.place_on_grid(grid_slice, UNIFORM_CONFIG_80NM, random_key)
        cartesian_detector = cartesian_detector.place_on_grid(grid_slice, UNIFORM_CONFIG_80NM, random_key)
        kspace_detector = kspace_detector.place_on_grid(grid_slice, UNIFORM_CONFIG_80NM, random_key)
        state = make_detector_state_for_plane_wave(angle_detector, theta_deg=20.0, phi_deg=35.0)

        original_asarray = field_projection_module.np.asarray

        def guarded_asarray(value, *args, **kwargs):
            if isinstance(value, jax.Array):
                raise AssertionError("JAX coordinate arrays should not be converted with numpy.asarray")
            return original_asarray(value, *args, **kwargs)

        monkeypatch.setattr(field_projection_module.np, "asarray", guarded_asarray)

        theta = jnp.deg2rad(jnp.asarray([18.0, 20.0, 22.0]))
        phi = jnp.deg2rad(jnp.asarray([32.0, 35.0, 38.0]))
        angle_result = angle_detector.project(state, theta, phi)
        assert isinstance(angle_result["power"], jax.Array)

        cartesian_result = cartesian_detector.project(state, jnp.asarray([0.1e-6]), jnp.asarray([0.05e-6]))
        assert isinstance(cartesian_result["power"], jax.Array)

        kspace_result = kspace_detector.project(state, jnp.asarray([0.10]), jnp.asarray([0.05]))
        assert isinstance(kspace_result["power"], jax.Array)

    def test_project_far_field_box_is_jittable_and_differentiable(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            exclude_surfaces=("x-", "x+", "y-", "y+", "z-"),
        )
        detector = detector.place_on_grid(((0, 2), (0, 2), (0, 2)), UNIFORM_CONFIG_80NM, random_key)
        state = make_box_detector_state_for_plane_wave(detector, theta_deg=25.0, phi_deg=35.0)
        theta = jnp.deg2rad(jnp.asarray([25.0]))
        phi = jnp.deg2rad(jnp.asarray([35.0]))

        result = detector.project(state, theta, phi)
        assert isinstance(result["power"], jax.Array)

        def loss(scale):
            scaled_state = {key: value * scale for key, value in state.items()}
            projected = detector.project(scaled_state, theta, phi)
            return jnp.real(jnp.sum(projected["power"]))

        eager_loss = loss(jnp.asarray(1.0))
        jit_loss = jax.jit(loss)(jnp.asarray(1.0))
        grad = jax.grad(loss)(jnp.asarray(1.0))

        assert jnp.all(jnp.isfinite(jit_loss))
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.allclose(jit_loss, eager_loss, rtol=1e-6, atol=1e-12)
        assert jnp.abs(grad) > 0.0

    def test_project_exact_planar_is_jittable_and_differentiable(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_distance=2.0e-6,
            far_field_approx=False,
            window_size=(0.0, 0.0),
        )
        detector = detector.place_on_grid(((0, 4), (0, 4), (0, 1)), UNIFORM_CONFIG_80NM, random_key)
        state = make_detector_state_for_plane_wave(detector, theta_deg=20.0, phi_deg=35.0)
        theta = jnp.deg2rad(jnp.asarray([20.0]))
        phi = jnp.deg2rad(jnp.asarray([35.0]))

        result = detector.project(state, theta, phi)
        assert isinstance(result["power"], jax.Array)

        phasor = state["phasor"]

        def loss(scale):
            scaled_state = {"phasor": phasor * scale}
            projected = detector.project(scaled_state, theta, phi)
            return jnp.real(jnp.sum(projected["power"]))

        eager_loss = loss(jnp.asarray(1.0))
        jit_loss = jax.jit(loss)(jnp.asarray(1.0))
        grad = jax.grad(loss)(jnp.asarray(1.0))

        assert jnp.all(jnp.isfinite(jit_loss))
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.allclose(jit_loss, eager_loss, rtol=1e-6, atol=1e-12)
        assert jnp.abs(grad) > 0.0

    @pytest.mark.parametrize("far_field_approx", [True, False])
    def test_cartesian_projection_matches_angle_projection_for_same_observation_point(
        self,
        random_key,
        single_frequency,
        far_field_approx,
    ):
        theta_value = math.radians(20.0)
        phi_value = math.radians(35.0)
        projection_plane_distance = 4.0e-6
        radial_distance = projection_plane_distance / math.cos(theta_value)
        x_value = projection_plane_distance * math.tan(theta_value) * math.cos(phi_value)
        y_value = projection_plane_distance * math.tan(theta_value) * math.sin(phi_value)
        angle_detector = place_projection_detector(
            FieldProjectionAngleDetector,
            single_frequency,
            random_key,
            (6, 6, 1),
            direction="+",
            projection_distance=radial_distance,
            far_field_approx=far_field_approx,
            window_size=(0.0, 0.0),
        )
        cartesian_detector = place_projection_detector(
            FieldProjectionCartesianDetector,
            single_frequency,
            random_key,
            (6, 6, 1),
            direction="+",
            projection_distance=projection_plane_distance,
            projection_axis=2,
            far_field_approx=far_field_approx,
            window_size=(0.0, 0.0),
        )
        state = make_detector_state_for_plane_wave(angle_detector, theta_deg=20.0, phi_deg=35.0)

        angle_result = angle_detector.project(state, np.asarray([theta_value]), np.asarray([phi_value]))
        cartesian_result = cartesian_detector.project(state, np.asarray([x_value]), np.asarray([y_value]))

        assert np.isclose(cartesian_result["theta"][0, 0], theta_value)
        assert np.isclose(cartesian_result["phi"][0, 0], phi_value)
        assert "radial_distance" not in cartesian_result
        assert "observation_points" not in cartesian_result
        assert cartesian_result["projection_axis"] == 2
        assert_projected_fields_close(cartesian_result, angle_result, rtol=1e-5, atol=5e-8)

    @pytest.mark.parametrize(
        ("projection_axis", "far_field_approx"),
        [
            (0, True),
            (1, True),
            (2, True),
            (2, False),
        ],
    )
    def test_kspace_projection_matches_angle_projection_for_same_direction(
        self,
        random_key,
        single_frequency,
        projection_axis,
        far_field_approx,
    ):
        ux = np.asarray([0.25])
        uy = np.asarray([0.10])
        projection_distance = 6.0e-6
        kspace_detector = place_projection_detector(
            FieldProjectionKSpaceDetector,
            single_frequency,
            random_key,
            (6, 6, 1),
            direction="+",
            projection_axis=projection_axis,
            projection_distance=projection_distance,
            far_field_approx=far_field_approx,
            window_size=(0.0, 0.0),
        )
        state = make_detector_state_for_plane_wave(kspace_detector, theta_deg=20.0, phi_deg=35.0)

        kspace_result = kspace_detector.project(state, ux, uy)
        angle_detector = place_projection_detector(
            FieldProjectionAngleDetector,
            single_frequency,
            random_key,
            (6, 6, 1),
            direction="+",
            projection_distance=projection_distance,
            far_field_approx=far_field_approx,
            window_size=(0.0, 0.0),
        )
        angle_result = angle_detector.project(
            state,
            np.asarray([kspace_result["theta"][0, 0]]),
            np.asarray([kspace_result["phi"][0, 0]]),
        )

        assert kspace_result["projection_axis"] == projection_axis
        assert_projected_fields_close(kspace_result, angle_result, rtol=1e-5, atol=5e-8)

    @pytest.mark.parametrize(
        ("detector_class", "coordinates"),
        [
            (FieldProjectionCartesianDetector, (jnp.asarray([0.1e-6]), jnp.asarray([0.05e-6]))),
            (FieldProjectionKSpaceDetector, (jnp.asarray([0.10]), jnp.asarray([0.05]))),
        ],
    )
    def test_coordinate_projection_is_jittable_and_differentiable(
        self,
        random_key,
        single_frequency,
        detector_class,
        coordinates,
    ):
        detector = place_projection_detector(
            detector_class,
            single_frequency,
            random_key,
            (4, 4, 1),
            config=UNIFORM_CONFIG_80NM,
            direction="+",
            projection_distance=2.0e-6,
            window_size=(0.0, 0.0),
        )
        state = make_detector_state_for_plane_wave(detector, theta_deg=20.0, phi_deg=35.0)
        first_coordinate, second_coordinate = coordinates

        result = detector.project(state, first_coordinate, second_coordinate)
        assert isinstance(result["power"], jax.Array)

        phasor = state["phasor"]

        def loss(scale, first_coordinate, second_coordinate):
            scaled_state = {"phasor": phasor * scale}
            projected = detector.project(scaled_state, first_coordinate, second_coordinate)
            return jnp.real(jnp.sum(projected["power"]))

        eager_loss = loss(jnp.asarray(1.0), first_coordinate, second_coordinate)
        jit_loss = jax.jit(loss)(jnp.asarray(1.0), first_coordinate, second_coordinate)
        grad = jax.grad(loss)(jnp.asarray(1.0), first_coordinate, second_coordinate)

        assert jnp.all(jnp.isfinite(jit_loss))
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.allclose(jit_loss, eager_loss, rtol=1e-6, atol=1e-12)
        assert jnp.abs(grad) > 0.0

    def test_cartesian_and_kspace_project_all_match_individual_projections(self, random_key, multiple_frequencies):
        cartesian_detector = place_projection_detector(
            FieldProjectionCartesianDetector,
            multiple_frequencies,
            random_key,
            (4, 4, 1),
            direction="+",
            projection_distance=4.0e-6,
            far_field_approx=False,
            window_size=(0.0, 0.0),
        )
        kspace_detector = place_projection_detector(
            FieldProjectionKSpaceDetector,
            multiple_frequencies,
            random_key,
            (4, 4, 1),
            direction="+",
            projection_distance=4.0e-6,
            far_field_approx=False,
            window_size=(0.0, 0.0),
        )
        state_0 = make_detector_state_for_plane_wave(
            cartesian_detector,
            theta_deg=18.0,
            phi_deg=26.0,
            wave_character_index=0,
        )
        state_1 = make_detector_state_for_plane_wave(
            cartesian_detector,
            theta_deg=31.0,
            phi_deg=43.0,
            wave_character_index=1,
        )
        state = combine_phasor_states(state_0, state_1)

        cartesian_all = assert_project_all_matches_project(
            cartesian_detector,
            state,
            (np.asarray([0.1e-6]), np.asarray([0.3e-6])),
        )
        assert_project_all_matches_project(kspace_detector, state, (np.asarray([0.1]), np.asarray([0.05])))
        assert "radial_distance" not in cartesian_all
        assert "observation_points" not in cartesian_all

    def test_project_result_includes_wave_metadata(self, random_key, single_frequency):
        refractive_index = 1.7
        impedance = constants.eta0 / 2.3
        projection_distance = 2.5
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_distance=projection_distance,
            projection_medium_refractive_index=refractive_index,
            projection_medium_impedance=impedance,
        )
        detector = detector.place_on_grid(((0, 8), (0, 8), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(
            detector,
            theta_deg=0.0,
            phi_deg=0.0,
            refractive_index=refractive_index,
            impedance=impedance,
        )

        result = detector.project(state, np.asarray([0.0]), np.asarray([0.0]))

        assert result["wave_character_index"] == 0
        assert np.isclose(result["frequency"], single_frequency[0].get_frequency())
        assert np.isclose(result["free_space_wavelength"], single_frequency[0].get_wavelength())
        assert np.isclose(result["projection_distance"], projection_distance)
        assert np.isclose(result["projection_medium_refractive_index"], refractive_index)
        assert np.isclose(result["projection_medium_impedance"], impedance)
        assert np.isclose(result["projection_wavelength"], single_frequency[0].get_wavelength() / refractive_index)

    def test_projection_distance_scales_fields_not_radiated_power(self, random_key, single_frequency):
        near_detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_distance=1.0,
            window_size=(0.0, 0.0),
        )
        far_detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_distance=2.0,
            window_size=(0.0, 0.0),
        )
        near_detector = near_detector.place_on_grid(((0, 8), (0, 8), (0, 1)), UNIFORM_CONFIG, random_key)
        far_detector = far_detector.place_on_grid(((0, 8), (0, 8), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(near_detector, theta_deg=20.0, phi_deg=35.0)
        theta = np.deg2rad(np.asarray([20.0]))
        phi = np.deg2rad(np.asarray([35.0]))

        near_result = near_detector.project(state, theta, phi)
        far_result = far_detector.project(state, theta, phi)

        peak_index = np.unravel_index(np.argmax(near_result["power"]), np.asarray(near_result["power"]).shape)
        assert np.isclose(
            abs(far_result["Etheta"][peak_index]) / abs(near_result["Etheta"][peak_index]),
            0.5,
            rtol=1e-6,
        )
        assert np.isclose(
            far_result["power"][peak_index] / near_result["power"][peak_index],
            0.25,
            rtol=1e-6,
        )

    def test_project_uses_selected_wave_character_index(self, random_key, multiple_frequencies):
        detector = FieldProjectionAngleDetector(
            wave_characters=multiple_frequencies,
            direction="+",
            window_size=(0.0, 0.0),
        )
        detector = detector.place_on_grid(((0, 24), (0, 24), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(
            detector,
            theta_deg=24.0,
            phi_deg=32.0,
            wave_character_index=1,
        )
        theta = np.deg2rad(np.asarray([23.0, 24.0, 25.0]))
        phi = np.deg2rad(np.asarray([31.0, 32.0, 33.0]))

        result = detector.project(state, theta, phi, wave_character_index=1)

        assert result["wave_character_index"] == 1
        assert_power_peak_at(result["power"], theta, phi, 24.0, 32.0)

    def test_project_all_matches_individual_frequency_projections(self, random_key, multiple_frequencies):
        detector = FieldProjectionAngleDetector(
            wave_characters=multiple_frequencies,
            direction="+",
            window_size=(0.0, 0.0),
        )
        detector = detector.place_on_grid(((0, 24), (0, 24), (0, 1)), UNIFORM_CONFIG, random_key)
        state_0 = make_detector_state_for_plane_wave(
            detector,
            theta_deg=18.0,
            phi_deg=26.0,
            wave_character_index=0,
        )
        state_1 = make_detector_state_for_plane_wave(
            detector,
            theta_deg=31.0,
            phi_deg=43.0,
            wave_character_index=1,
        )
        state = combine_phasor_states(state_0, state_1)
        theta = np.deg2rad(np.asarray([18.0, 31.0]))
        phi = np.deg2rad(np.asarray([26.0, 43.0]))

        result = detector.project_all(state, theta, phi)
        individual_0 = detector.project(state, theta, phi, wave_character_index=0)
        individual_1 = detector.project(state, theta, phi, wave_character_index=1)

        assert_project_all_matches_individuals(result, (individual_0, individual_1))
        assert np.allclose(result["theta"], theta)
        assert np.allclose(result["phi"], phi)
        assert np.array_equal(result["wave_character_indices"], np.asarray([0, 1]))
        assert np.allclose(result["frequencies"], [wc.get_frequency() for wc in multiple_frequencies])
        assert np.allclose(result["free_space_wavelengths"], [wc.get_wavelength() for wc in multiple_frequencies])
        assert np.isclose(result["projection_distance"], 1.0)
        assert np.allclose(result["projection_medium_refractive_indices"], np.ones(2))
        assert np.allclose(result["projection_medium_impedances"], constants.eta0 * np.ones(2))
        assert np.allclose(result["projection_wavelengths"], [wc.get_wavelength() for wc in multiple_frequencies])
        assert result["direction"] == "+"
        assert result["surfaces"] == ("z+",)

        assert_frequency_peaks(result, theta, phi, ((18.0, 26.0), (31.0, 43.0)))

    def test_project_all_handles_box_projection(self, random_key, multiple_frequencies):
        detector = FieldProjectionAngleDetector(
            wave_characters=multiple_frequencies,
            exclude_surfaces=("x-", "y-", "y+", "z-"),
        )
        detector = detector.place_on_grid(((0, 8), (0, 9), (0, 10)), UNIFORM_CONFIG, random_key)
        state_0 = make_box_detector_state_for_plane_wave(
            detector,
            theta_deg=18.0,
            phi_deg=26.0,
            wave_character_index=0,
        )
        state_1 = make_box_detector_state_for_plane_wave(
            detector,
            theta_deg=31.0,
            phi_deg=43.0,
            wave_character_index=1,
        )
        state = combine_surface_states(state_0, state_1)
        theta = np.deg2rad(np.asarray([18.0, 31.0]))
        phi = np.deg2rad(np.asarray([26.0, 43.0]))

        result = detector.project_all(state, theta, phi)
        individual_0 = detector.project(state, theta, phi, wave_character_index=0)
        individual_1 = detector.project(state, theta, phi, wave_character_index=1)

        assert_project_all_matches_individuals(result, (individual_0, individual_1))
        assert result["direction"] is None
        assert result["surfaces"] == ("x+", "z+")
        assert np.allclose(result["projection_wavelengths"], [wc.get_wavelength() for wc in multiple_frequencies])

    def test_project_uses_global_theta_phi_for_nonstandard_planes(self, random_key, single_frequency):
        cases = [
            dict(direction="+", grid_shape=(1, 24, 24), angles=(68.0, 22.0)),
            dict(direction="+", grid_shape=(24, 1, 24), angles=(64.0, 70.0)),
            dict(direction="-", grid_shape=(24, 24, 1), angles=(145.0, 35.0)),
        ]
        for case in cases:
            theta_deg, phi_deg = case["angles"]
            detector = place_projection_detector(
                FieldProjectionAngleDetector,
                single_frequency,
                random_key,
                case["grid_shape"],
                direction=case["direction"],
                window_size=(0.0, 0.0),
            )
            state = make_detector_state_for_plane_wave(detector, theta_deg=theta_deg, phi_deg=phi_deg)
            theta = np.deg2rad(np.asarray([theta_deg - 1.0, theta_deg, theta_deg + 1.0]))
            phi = np.deg2rad(np.asarray([phi_deg - 1.0, phi_deg, phi_deg + 1.0]))

            result = detector.project(state, theta, phi)

            assert_power_peak_at(result["power"], theta, phi, theta_deg, phi_deg)

    def test_projection_medium_parameters_set_phase_and_impedance(self, random_key, single_frequency):
        cases = [
            dict(
                detector_kwargs={"projection_medium_refractive_index": 2.0},
                refractive_index=2.0,
                impedance=constants.eta0 / 2.0,
                angles=(30.0, 25.0),
            ),
            dict(
                detector_kwargs={
                    "projection_medium_refractive_index": 2.0,
                    "projection_medium_impedance": constants.eta0 / 3.0,
                },
                refractive_index=2.0,
                impedance=constants.eta0 / 3.0,
                angles=(25.0, 30.0),
            ),
            dict(
                detector_kwargs={"projection_medium": Material(permittivity=4.0, permeability=2.25)},
                refractive_index=3.0,
                impedance=constants.eta0 * 0.75,
                angles=(30.0, 25.0),
                assert_metadata=True,
            ),
        ]
        for case in cases:
            theta_deg, phi_deg = case["angles"]
            detector = place_projection_detector(
                FieldProjectionAngleDetector,
                single_frequency,
                random_key,
                (24, 24, 1),
                direction="+",
                window_size=(0.0, 0.0),
                **case["detector_kwargs"],
            )
            state = make_detector_state_for_plane_wave(
                detector,
                theta_deg=theta_deg,
                phi_deg=phi_deg,
                refractive_index=case["refractive_index"],
                impedance=case["impedance"],
            )
            theta = np.deg2rad(np.asarray([theta_deg - 2.0, theta_deg, theta_deg + 2.0]))
            phi = np.deg2rad(np.asarray([phi_deg - 2.0, phi_deg, phi_deg + 2.0]))

            result = detector.project(state, theta, phi)

            peak = assert_power_peak_at(result["power"], theta, phi, theta_deg, phi_deg)
            assert np.allclose(result["Hphi"][peak], result["Etheta"][peak] / case["impedance"], rtol=1e-6, atol=1e-6)
            if case.get("assert_metadata"):
                assert np.isclose(result["projection_medium_refractive_index"], case["refractive_index"])
                assert np.isclose(result["projection_medium_impedance"], case["impedance"])
                assert np.isclose(
                    result["projection_wavenumber"],
                    2.0 * np.pi * case["refractive_index"] / single_frequency[0].get_wavelength(),
                )

    def test_per_frequency_projection_medium_parameters(self, random_key, multiple_frequencies):
        refractive_indices = (1.4, 2.1)
        impedances = (constants.eta0 / 2.5, constants.eta0 / 3.5)
        detector = FieldProjectionAngleDetector(
            wave_characters=multiple_frequencies,
            direction="+",
            projection_medium_refractive_index=refractive_indices,
            projection_medium_impedance=impedances,
            window_size=(0.0, 0.0),
        )
        detector = detector.place_on_grid(((0, 24), (0, 24), (0, 1)), UNIFORM_CONFIG, random_key)
        state_0 = make_detector_state_for_plane_wave(
            detector,
            theta_deg=18.0,
            phi_deg=24.0,
            refractive_index=refractive_indices[0],
            impedance=impedances[0],
            wave_character_index=0,
        )
        state_1 = make_detector_state_for_plane_wave(
            detector,
            theta_deg=32.0,
            phi_deg=42.0,
            refractive_index=refractive_indices[1],
            impedance=impedances[1],
            wave_character_index=1,
        )
        state = combine_phasor_states(state_0, state_1)
        theta = np.deg2rad(np.asarray([18.0, 32.0]))
        phi = np.deg2rad(np.asarray([24.0, 42.0]))

        result = detector.project_all(state, theta, phi)

        peak_0, peak_1 = assert_frequency_peaks(result, theta, phi, ((18.0, 24.0), (32.0, 42.0)))
        assert np.allclose(
            result["Hphi"][0][peak_0],
            result["Etheta"][0][peak_0] / impedances[0],
            rtol=1e-6,
            atol=1e-6,
        )
        assert np.allclose(
            result["Hphi"][1][peak_1],
            result["Etheta"][1][peak_1] / impedances[1],
            rtol=1e-6,
            atol=1e-6,
        )
        assert np.allclose(result["projection_medium_refractive_indices"], refractive_indices)
        assert np.allclose(result["projection_medium_impedances"], impedances)
        assert np.allclose(
            result["projection_wavelengths"],
            [
                wc.get_wavelength() / refractive_index
                for wc, refractive_index in zip(multiple_frequencies, refractive_indices)
            ],
        )

    def test_projection_medium_material_matches_scalar_medium(self, random_key, single_frequency):
        for far_field_approx, grid_shape, atol in [(True, (16, 16, 1), 1e-6), (False, (4, 4, 1), 1e-10)]:
            material_detector = FieldProjectionAngleDetector(
                wave_characters=single_frequency,
                direction="+",
                projection_medium=Material(permittivity=2.25),
                far_field_approx=far_field_approx,
                projection_distance=4.0e-6,
            )
            scalar_detector = FieldProjectionAngleDetector(
                wave_characters=single_frequency,
                direction="+",
                projection_medium_refractive_index=1.5,
                far_field_approx=far_field_approx,
                projection_distance=4.0e-6,
            )
            grid_slice = tuple((0, size) for size in grid_shape)
            material_detector = material_detector.place_on_grid(grid_slice, UNIFORM_CONFIG, random_key)
            scalar_detector = scalar_detector.place_on_grid(grid_slice, UNIFORM_CONFIG, random_key)
            state = make_detector_state_for_plane_wave(
                scalar_detector, theta_deg=25.0, phi_deg=35.0, refractive_index=1.5
            )
            theta = np.deg2rad(np.asarray([25.0]))
            phi = np.deg2rad(np.asarray([35.0]))

            material_result = material_detector.project(state, theta, phi)
            scalar_result = scalar_detector.project(state, theta, phi)

            assert_projected_fields_close(material_result, scalar_result, rtol=1e-6, atol=atol)
            assert material_result["far_field_approx"] is far_field_approx
            assert np.isclose(material_result["projection_medium_refractive_index"], 1.5)
            assert np.isclose(material_result["projection_medium_impedance"], constants.eta0 / 1.5)
            assert np.isclose(
                material_result["projection_wavenumber"],
                2.0 * np.pi * 1.5 / single_frequency[0].get_wavelength(),
            )
            assert np.isclose(material_result["projection_wavelength"], single_frequency[0].get_wavelength() / 1.5)

    def test_projection_medium_conductive_material_uses_complex_parameters(self, single_frequency):
        sigma = 100.0
        material = Material(permittivity=2.25, electric_conductivity=sigma)
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_medium=material,
            projection_distance=1.0e-6,
        )
        farther_detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_medium=material,
            projection_distance=2.0e-6,
        )
        omega = 2.0 * np.pi * single_frequency[0].get_frequency()
        eps_complex = 2.25 + 1j * sigma / (omega * constants.eps0)
        expected_index = passive_sqrt(eps_complex)
        expected_impedance = constants.eta0 * positive_impedance_sqrt(1.0 / eps_complex)
        refractive_index, impedance, wavenumber = detector._projection_parameters(0)

        assert np.isclose(refractive_index, expected_index)
        assert np.isclose(impedance, expected_impedance)
        assert np.isclose(
            wavenumber,
            2.0 * np.pi * expected_index / single_frequency[0].get_wavelength(),
        )
        assert abs(farther_detector._propagation_factor(0)) < 0.5 * abs(detector._propagation_factor(0))

    def test_projection_medium_dispersive_material_uses_permittivity_model(self, single_frequency):
        pole = LorentzPole(resonance_frequency=4.0e15, damping=1.0e13, delta_epsilon=0.5)
        dispersion = DispersionModel(poles=(pole,))
        material = Material(permittivity=2.0, dispersion=dispersion)
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            projection_medium=material,
        )
        omega = 2.0 * np.pi * single_frequency[0].get_frequency()
        expected_eps = dispersion.permittivity(omega, eps_inf=2.0)
        expected_index = passive_sqrt(expected_eps)
        expected_impedance = constants.eta0 * positive_impedance_sqrt(1.0 / expected_eps)
        refractive_index, impedance, wavenumber = detector._projection_parameters(0)

        assert np.isclose(refractive_index, expected_index)
        assert np.isclose(impedance, expected_impedance)
        assert np.isclose(
            wavenumber,
            2.0 * np.pi * expected_index / single_frequency[0].get_wavelength(),
        )

    def test_origin_normal_offset_changes_complex_phase_only(self, random_key, single_frequency):
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+", window_size=(0, 0))
        detector = detector.place_on_grid(((0, 16), (0, 16), (0, 1)), UNIFORM_CONFIG, random_key)
        shifted_origin = (0.0, 0.0, -0.2e-6)
        shifted = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            origin=shifted_origin,
            window_size=(0, 0),
        )
        shifted = shifted.place_on_grid(((0, 16), (0, 16), (0, 1)), UNIFORM_CONFIG, random_key)
        state = make_detector_state_for_plane_wave(detector, theta_deg=0.0, phi_deg=0.0)
        theta = np.asarray([0.0])
        phi = np.asarray([0.0])

        result = detector.project(state, theta, phi)
        shifted_result = shifted.project(state, theta, phi)

        default_origin_z = float(0.5 * (detector._axis_centers(2)[0] + detector._axis_centers(2)[-1]))
        wavenumber = 2.0 * np.pi / single_frequency[0].get_wavelength()
        expected_phase = np.exp(1j * wavenumber * (shifted_origin[2] - default_origin_z))
        assert np.isclose(shifted_result["Etheta"][0, 0] / result["Etheta"][0, 0], expected_phase, rtol=1e-6)
        assert np.isclose(abs(shifted_result["Etheta"][0, 0]), abs(result["Etheta"][0, 0]), rtol=1e-6)

    def test_rectilinear_grid_coordinates_are_physical_units(self, random_key, single_frequency):
        config = SimulationConfig(
            time=100e-15,
            grid=RectilinearGrid(
                x_edges=jnp.arange(65) * 40e-9,
                y_edges=jnp.arange(65) * 60e-9,
                z_edges=jnp.arange(2) * 80e-9,
            ),
            backend="cpu",
        )
        detector = FieldProjectionAngleDetector(wave_characters=single_frequency, direction="+")
        detector = detector.place_on_grid(((0, 24), (0, 24), (0, 1)), config, random_key)

        u_coords, v_coords = detector._local_transverse_coordinates()

        assert np.isclose(np.diff(u_coords).mean(), 40e-9)
        assert np.isclose(np.diff(v_coords).mean(), 60e-9)

    def test_resolved_quasi_uniform_grid_coordinates_are_physical_units(self, random_key, single_frequency):
        unresolved_config = SimulationConfig(
            time=100e-15,
            grid=QuasiUniformGrid(dx=40e-9, dy=60e-9, dz=80e-9),
            backend="cpu",
        )
        config = unresolved_config.aset("grid", unresolved_config.resolve_grid((8, 8, 2)))
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            window_size=(0.0, 0.0),
        )
        detector = detector.place_on_grid(((0, 4), (0, 4), (0, 1)), config, random_key)

        u_coords, v_coords = detector._local_transverse_coordinates()
        assert np.isclose(np.diff(u_coords).mean(), 40e-9)
        assert np.isclose(np.diff(v_coords).mean(), 60e-9)

    def test_nonuniform_rectilinear_grid_projection_uses_physical_coordinates(self, random_key, single_frequency):
        x_widths = 42e-9 * (1.0 + 0.12 * np.sin(np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)))
        y_widths = 58e-9 * (1.0 + 0.10 * np.cos(np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)))
        config = SimulationConfig(
            time=100e-15,
            grid=RectilinearGrid(
                x_edges=jnp.asarray(np.concatenate(([0.0], np.cumsum(x_widths)))),
                y_edges=jnp.asarray(np.concatenate(([0.0], np.cumsum(y_widths)))),
                z_edges=jnp.asarray([0.0, 75e-9]),
            ),
            backend="cpu",
        )
        detector = FieldProjectionAngleDetector(
            wave_characters=single_frequency,
            direction="+",
            window_size=(0.0, 0.0),
        )
        detector = detector.place_on_grid(((0, 24), (0, 24), (0, 1)), config, random_key)
        state = make_detector_state_for_plane_wave(detector, theta_deg=23.0, phi_deg=31.0)
        theta = np.deg2rad(np.asarray([22.0, 23.0, 24.0]))
        phi = np.deg2rad(np.asarray([30.0, 31.0, 32.0]))

        result = detector.project(state, theta, phi)

        u_coords, v_coords = detector._local_transverse_coordinates()
        assert np.ptp(np.diff(u_coords)) > 5e-9
        assert np.ptp(np.diff(v_coords)) > 5e-9
        assert_power_peak_at(result["power"], theta, phi, 23.0, 31.0)
