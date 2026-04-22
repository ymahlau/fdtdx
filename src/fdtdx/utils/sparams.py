from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jax

from fdtdx import DetectorState, GaussianPulseProfile, extend_material_to_pml
from fdtdx.config import SimulationConfig
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.initialization import apply_params, place_objects
from fdtdx.fdtd.stop_conditions import EnergyThresholdCondition
from fdtdx.fdtd.wrapper import run_fdtd
from fdtdx.materials import Material
from fdtdx.objects.boundaries.initialization import BoundaryConfig, boundary_objects_from_config
from fdtdx.objects.detectors.mode import ModeOverlapDetector
from fdtdx.objects.sources.mode import ModePlaneSource
from fdtdx.objects.static_material.polygon import ExtrudedPolygon
from fdtdx.objects.static_material.static import SimulationVolume


@dataclass
class PortSpec:
    """Specification for a simulation port (input source or output detector).

    Coordinates are expressed in the *core* coordinate system where the origin
    corresponds to the start of the simulation domain (excluding PML padding).

    Args:
        center: 3-D centre position ``(x, y, z)`` in metres, relative to the
            start of the core region.
        axis: Propagation axis - ``0`` for x, ``1`` for y, ``2`` for z.
        direction: Propagation direction along ``axis`` - ``'+'`` or ``'-'``.
        width: Cross-section extent (metres) along the first transverse axis.
        height: Cross-section extent (metres) along the second transverse axis.
        mode_index: Waveguide mode index (default 0 = fundamental mode).
        filter_pol: Polarisation filter - ``'te'``, ``'tm'``, or ``None``.
        name: Optional name for the source/detector object.
    """

    center: tuple[float, float, float]
    axis: int
    direction: Literal["+", "-"]
    width: float
    height: float
    mode_index: int = 0
    filter_pol: Literal["te", "tm"] | None = "te"
    name: str = ""


def _make_port_shape(axis: int, resolution: float, width: float, height: float) -> tuple[float, float, float]:
    """Return partial_real_shape with one-voxel thickness along the propagation axis."""
    transverse = [i for i in range(3) if i != axis]
    shape: list[float] = [resolution, resolution, resolution]
    shape[transverse[0]] = width
    shape[transverse[1]] = height
    return (shape[0], shape[1], shape[2])


def setup_sparams_simulation(
    polygons: list[tuple[ExtrudedPolygon, tuple[float, float, float]]],
    input_ports: list[PortSpec],
    output_ports: list[PortSpec],
    wavelength: float,
    resolution: float,
    max_time: float,
    domain_size: tuple[float, float, float],
    background_material: Material | None = None,
    pml_layers: int = 10,
    key: jax.Array | None = None,
) -> tuple[ObjectContainer, ArrayContainer, SimulationConfig]:
    """Set up an FDTD simulation scene for S-parameter extraction.

    Builds a fully initialised simulation scene containing:

    * A background :class:`~fdtdx.objects.static_material.static.SimulationVolume`
      surrounded by PML absorbing boundaries on all six sides.
    * Any GDS-derived :class:`~fdtdx.objects.static_material.polygon.ExtrudedPolygon`
      objects placed at their requested positions.
    * A :class:`~fdtdx.objects.sources.mode.ModePlaneSource` for every input
      port.
    * A :class:`~fdtdx.objects.detectors.mode.ModeOverlapDetector` for every
      output port.

    To compute the full S-matrix, call this function once per input port (each
    time with a single entry in *input_ports*) and collect the detector
    readings.

    Args:
        polygons: Pairs of ``(ExtrudedPolygon, center_offset)`` where
            ``center_offset`` is the 3-D centre of the polygon in the *core*
            coordinate system (metres, origin at the start of the core region).
            The polygon's ``partial_real_shape`` must be fully specified at
            construction time (no ``None`` entries).
        input_ports: Ports that receive a
            :class:`~fdtdx.objects.sources.mode.ModePlaneSource`.
        output_ports: Ports that receive a
            :class:`~fdtdx.objects.detectors.mode.ModeOverlapDetector`.
        wavelength: Free-space wavelength in metres.
        resolution: Spatial resolution (voxel size) in metres.
        max_time: Total simulation time in seconds.
        domain_size: Size of the *core* simulation region (excluding PML) as
            ``(Lx, Ly, Lz)`` in metres.
        background_material: Material filling the simulation volume.  Defaults
            to air (``Material()``).
        pml_layers: Number of PML grid cells added to every face.
        key: JAX random key used by :func:`~fdtdx.fdtd.initialization.place_objects`.
            Defaults to ``PRNGKey(0)`` when ``None``. Usually not necessary to specify
            since simulation is deterministic.

    Returns:
        A 3-tuple ``(objects, arrays, config)``, ready to pass to
        :func:`calculate_sparam`.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    if background_material is None:
        background_material = Material()

    pml_thickness = pml_layers * resolution
    total_size: tuple[float, float, float] = (
        domain_size[0] + 2.0 * pml_thickness,
        domain_size[1] + 2.0 * pml_thickness,
        domain_size[2] + 2.0 * pml_thickness,
    )

    config = SimulationConfig(time=max_time, resolution=resolution)

    object_list = []
    constraints = []

    background = SimulationVolume(
        partial_real_shape=total_size,
        material=background_material,
        name="Background",
    )
    object_list.append(background)

    bound_cfg = BoundaryConfig(
        thickness_grid_minx=pml_layers,
        thickness_grid_maxx=pml_layers,
        thickness_grid_miny=pml_layers,
        thickness_grid_maxy=pml_layers,
        thickness_grid_minz=pml_layers,
        thickness_grid_maxz=pml_layers,
    )
    boundary_dict, boundary_constraints = boundary_objects_from_config(bound_cfg, background)
    object_list.extend(boundary_dict.values())
    constraints.extend(boundary_constraints)

    def _center_at(obj, offset: tuple[float, float, float]):
        """Constrain obj centre to core-region position offset."""
        return obj.place_relative_to(
            background,
            axes=(0, 1, 2),
            own_positions=(0.0, 0.0, 0.0),
            other_positions=(-1.0, -1.0, -1.0),
            margins=(
                offset[0],
                offset[1],
                offset[2],
            ),
        )

    for poly, offset in polygons:
        object_list.append(poly)
        constraints.append(_center_at(poly, offset))

    center_wave_character = WaveCharacter(wavelength=wavelength)
    width_wave_character = WaveCharacter(wavelength=wavelength * 10)
    profile = GaussianPulseProfile(center_wave=center_wave_character, spectral_width=width_wave_character)

    for i, port in enumerate(input_ports):
        name = port.name if port.name else f"Source_{i}"
        source = ModePlaneSource(
            mode_index=port.mode_index,
            filter_pol=port.filter_pol,
            direction=port.direction,
            temporal_profile=profile,
            wave_character=center_wave_character,
            partial_real_shape=_make_port_shape(port.axis, resolution, port.width, port.height),
            name=name,
        )
        object_list.append(source)
        constraints.append(_center_at(source, port.center))

        input_detector = ModeOverlapDetector(
            mode_index=port.mode_index,
            filter_pol=port.filter_pol,
            direction=port.direction,
            wave_characters=(center_wave_character,),
            partial_real_shape=_make_port_shape(port.axis, resolution, port.width, port.height),
            name=f"{name}_input_normalization",
        )
        object_list.append(input_detector)
        constraints.append(_center_at(input_detector, port.center))

    for i, port in enumerate(output_ports):
        name = port.name if port.name else f"Detector_{i}"
        detector = ModeOverlapDetector(
            mode_index=port.mode_index,
            filter_pol=port.filter_pol,
            direction=port.direction,
            wave_characters=(center_wave_character,),
            partial_real_shape=_make_port_shape(port.axis, resolution, port.width, port.height),
            name=name,
        )
        object_list.append(detector)
        constraints.append(_center_at(detector, port.center))

    objects, arrays, _, config, _ = place_objects(
        object_list=object_list,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays = extend_material_to_pml(
        objects=objects,
        arrays=arrays,
    )

    return objects, arrays, config


def calculate_sparam(
    objects: ObjectContainer,
    arrays: ArrayContainer,
    config: SimulationConfig,
    input_port_name: str,
    show_progress: bool = True,
    input_normalization_detector_name: str | None = None,
    key: jax.Array | None = None,
) -> tuple[dict[tuple[str, str], jax.Array], dict[str, DetectorState]]:
    """Run the FDTD simulation and extract S-parameters from mode-overlap detectors.

    Intended to be called with the outputs of :func:`setup_sparams_simulation`.
    Each :class:`~fdtdx.objects.detectors.mode.ModeOverlapDetector` in *objects*
    contributes one entry to the returned dictionary.  Because a single simulation
    (with one active input port) measures the transmission to **all** output ports
    simultaneously, the dictionary keys are ``(detector_name, input_port_name)``
    tuples so that results from multiple calls can be merged into a full S-matrix.

    To simulate all input ports in one call (multiple simulations), use :func:`calculate_sparams`.

    Args:
        objects: ObjectContainer from :func:`setup_sparams_simulation`.
        arrays: ArrayContainer from :func:`setup_sparams_simulation`.
        config: SimulationConfig from :func:`setup_sparams_simulation`.
        input_port_name: Name of the active input port.  Should match the
            ``name`` field of the corresponding :class:`PortSpec`, or the
            auto-generated name ``"Source_<i>"`` when no name was supplied.
        show_progress: Whether to display the simulation progress bar.
        input_normalization_detector_name: Name (or substring) of the detector
            used to normalise the input power.  Defaults to a detector whose
            name contains *input_port_name*.
        key: JAX random key.  Defaults to ``PRNGKey(0)``.

    Returns:
        A 2-tuple ``(sparams, detector_states)`` where *sparams* maps
        ``(detector_name, input_port_name)`` to a complex scattering amplitude
        and *detector_states* is the final :class:`~fdtdx.DetectorState` dict
        for every detector in the simulation.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # turn off all sources except for input port source
    found_input = False
    for source in objects.sources:
        if source.name == input_port_name:
            found_input = True
            continue
        source_idx = objects.index(source.name)
        objects = objects.aset(f"object_list->[{source_idx}]->switch->is_always_off", True)
    if not found_input:
        raise ValueError(f"{input_port_name=} does not exist")

    input_norm_name_part = (
        input_normalization_detector_name if input_normalization_detector_name is not None else input_port_name
    )
    input_norm_name = determine_input_norm_detector_name(input_norm_name_part, objects)

    # apply_params (with no device params) calls obj.apply() on every object, which triggers mode-profile computation
    # inside ModeOverlapDetector and ModePlaneSource.
    key, subkey = jax.random.split(key)
    arrays, objects, _ = apply_params(arrays, objects, {}, subkey)

    # run the simulation for at least 10% of max time specified
    stopping_condition = EnergyThresholdCondition(
        min_steps=round(config.time_steps_total / 10),
    )

    jitted_fdtd = jax.jit(run_fdtd, static_argnames=["show_progress"])
    _, final_arrays = jitted_fdtd(
        arrays=arrays,
        objects=objects,
        config=config,
        key=key,
        show_progress=show_progress,
        stopping_condition=stopping_condition,
    )

    input_det_state = final_arrays.detector_states[input_norm_name]
    input_det = objects[input_norm_name]
    assert isinstance(input_det, ModeOverlapDetector)
    input_overlap = input_det.compute_overlap(input_det_state)

    result: dict[tuple[str, str], jax.Array] = {}
    for obj in objects.object_list:
        if isinstance(obj, ModeOverlapDetector):
            state = final_arrays.detector_states[obj.name]
            raw_overlap = obj.compute_overlap(state)
            result[(obj.name, input_port_name)] = raw_overlap / input_overlap
    return result, final_arrays.detector_states


def calculate_sparams(
    objects: ObjectContainer,
    arrays: ArrayContainer,
    config: SimulationConfig,
    input_port_names: Sequence[str],
    show_progress: bool = True,
    input_normalization_detector_name: str | None = None,
    key: jax.Array | None = None,
    return_detector_states: bool = False,
) -> tuple[dict[tuple[str, str], jax.Array], list[dict[str, DetectorState]]]:
    """Run FDTD simulations for multiple input ports and merge S-parameters.

    Calls :func:`calculate_sparam` once per entry in *input_port_names* and
    merges all results into a single S-parameter dictionary.

    Args:
        objects: ObjectContainer from :func:`setup_sparams_simulation`.
        arrays: ArrayContainer from :func:`setup_sparams_simulation`.
        config: SimulationConfig from :func:`setup_sparams_simulation`.
        input_port_names: Names of the input ports to simulate.
        show_progress: Whether to display the simulation progress bar.
        input_normalization_detector_name: Passed through to :func:`calculate_sparam`.
        key: JAX random key.  Defaults to ``PRNGKey(0)``.
        return_detector_states: When ``True``, return the detector states from
            each simulation run as a list (one entry per input port).  When
            ``False`` an empty list is returned.

    Returns:
        A 2-tuple ``(sparams, detector_states_list)`` where *sparams* is the
        merged ``dict[tuple[str, str], jax.Array]`` across all simulations and
        *detector_states_list* is either a list of per-simulation detector
        state dicts or an empty list.
    """
    merged: dict[tuple[str, str], jax.Array] = {}
    all_states: list[dict[str, DetectorState]] = []
    for name in input_port_names:
        sparam_dict, states = calculate_sparam(
            objects,
            arrays,
            config,
            name,
            show_progress,
            input_normalization_detector_name,
            key,
        )
        merged.update(sparam_dict)
        if return_detector_states:
            all_states.append(states)
    return merged, all_states


def determine_input_norm_detector_name(name_part: str, objects: ObjectContainer) -> str:
    results = []
    for obj in objects.object_list:
        if isinstance(obj, ModeOverlapDetector):
            if name_part in obj.name:
                results.append(obj.name)
    if len(results) == 1:
        return results[0]
    if not results:
        raise Exception(f"Cannot find input normalization detector: No detector has {name_part} in name.")
    raise Exception(
        f"Cannot uniquely determine input normalization detector. Found multiple detector with {name_part} as part"
        f" of their name. Found: {results}"
    )
