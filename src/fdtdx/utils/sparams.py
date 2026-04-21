from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax

from fdtdx import extend_material_to_pml
from fdtdx.config import SimulationConfig
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.initialization import apply_params, place_objects
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
        axis: Propagation axis – ``0`` for x, ``1`` for y, ``2`` for z.
        direction: Propagation direction along ``axis`` – ``'+'`` or ``'-'``.
        width: Cross-section extent (metres) along the first transverse axis.
        height: Cross-section extent (metres) along the second transverse axis.
        mode_index: Waveguide mode index (default 0 = fundamental mode).
        filter_pol: Polarisation filter – ``'te'``, ``'tm'``, or ``None``.
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

    wave_character = WaveCharacter(wavelength=wavelength)

    for i, port in enumerate(input_ports):
        name = port.name if port.name else f"Source_{i}"
        source = ModePlaneSource(
            mode_index=port.mode_index,
            filter_pol=port.filter_pol,
            direction=port.direction,
            wave_character=wave_character,
            partial_real_shape=_make_port_shape(port.axis, resolution, port.width, port.height),
            name=name,
        )
        object_list.append(source)
        constraints.append(_center_at(source, port.center))

    for i, port in enumerate(output_ports):
        name = port.name if port.name else f"Detector_{i}"
        detector = ModeOverlapDetector(
            mode_index=port.mode_index,
            filter_pol=port.filter_pol,
            direction=port.direction,
            wave_characters=(wave_character,),
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
    key: jax.Array | None = None,
    show_progress: bool = True,
) -> dict[tuple[str, str], jax.Array]:
    """Run the FDTD simulation and extract S-parameters from mode-overlap detectors.

    Intended to be called with the outputs of :func:`setup_sparams_simulation`.
    Each :class:`~fdtdx.objects.detectors.mode.ModeOverlapDetector` in *objects*
    contributes one entry to the returned dictionary.  Because a single simulation
    (with one active input port) measures the transmission to **all** output ports
    simultaneously, the dictionary keys are ``(detector_name, input_port_name)``
    tuples so that results from multiple calls can be merged into a full S-matrix.

    Args:
        objects: ObjectContainer from :func:`setup_sparams_simulation`.
        arrays: ArrayContainer from :func:`setup_sparams_simulation`.
        config: SimulationConfig from :func:`setup_sparams_simulation`.
        input_port_name: Name of the active input port.  Should match the
            ``name`` field of the corresponding :class:`PortSpec`, or the
            auto-generated name ``"Source_<i>"`` when no name was supplied.
        key: JAX random key.  Defaults to ``PRNGKey(0)``.
        show_progress: Whether to display the simulation progress bar.

    Returns:
        Dictionary mapping ``(detector_name, input_port_name)`` to a complex
        scattering amplitude.  Results from multiple calls (one per input port)
        can be merged to build the full S-matrix.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    # apply_params (with no device params) calls obj.apply() on every object, which triggers mode-profile computation
    # inside ModeOverlapDetector and ModePlaneSource.
    arrays, objects, _ = apply_params(arrays, objects, {}, subkey)

    key, subkey = jax.random.split(key)
    _, final_arrays = run_fdtd(
        arrays=arrays,
        objects=objects,
        config=config,
        key=subkey,
        show_progress=show_progress,
    )

    result: dict[tuple[str, str], jax.Array] = {}
    for obj in objects.object_list:
        if isinstance(obj, ModeOverlapDetector):
            state = final_arrays.detector_states[obj.name]
            result[(obj.name, input_port_name)] = obj.compute_overlap(state)
    return result
