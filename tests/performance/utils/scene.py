"""Scene builder utilities for GDS-imported fdtdx simulation setups.

Port dict format (all quantities in SI metres):
  {"x_m": float, "y_m": float, "width_m": float, "orientation": float}
  orientation in degrees: 0=+x, 90=+y, 180=-x, 270=-y  (gdsfactory convention)
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def port_to_sim_coords(
    port: dict,
    domain_shape: tuple[float, float, float],
    gds_center: tuple[float, float],
) -> tuple[float, float]:
    """Convert a GDS port position to absolute simulation coordinates (metres).

    The simulation domain runs from 0 to domain_size on each axis, so:

        sim_x = port_x + domain_x/2 - gds_center_x
        sim_y = port_y + domain_y/2 - gds_center_y

    Args:
        port: Port dict with keys ``x_m`` and ``y_m`` in metres.
        domain_shape: (x, y, z) total simulation domain in metres.
        gds_center: GDS (x, y) coordinate in metres that maps to the simulation centre.

    Returns:
        (sim_x, sim_y) in absolute simulation metres (0 to domain_size).
    """
    return (
        port["x_m"] + domain_shape[0] / 2.0 - gds_center[0],
        port["y_m"] + domain_shape[1] / 2.0 - gds_center[1],
    )


def build_domain(
    domain_shape: tuple[float, float, float],
    pml_cells: int,
    background_material,
    grid,
    sim_time: float,
    dtype=None,
) -> tuple:
    """Create SimulationVolume, PML boundaries, background fill, and SimulationConfig.

    Args:
        domain_shape: (x, y, z) total domain extents in metres including PML.
        pml_cells: Number of PML cells on each face.
        background_material: :class:`fdtdx.Material` filling the entire volume.
        grid: Grid object (:class:`fdtdx.RectilinearGrid` or :class:`fdtdx.UniformGrid`).
        sim_time: Total simulation time in seconds.
        dtype: JAX dtype for field arrays; defaults to ``jnp.float32``.

    Returns:
        ``(objects, constraints, config, volume)``
    """
    import jax.numpy as jnp_

    import fdtdx
    from fdtdx.config import SimulationConfig

    if dtype is None:
        dtype = jnp_.float32

    config = SimulationConfig(
        grid=grid,
        time=sim_time,
        dtype=dtype,
        gradient_config=None,
    )

    objects: list = []
    constraints: list = []

    volume = fdtdx.SimulationVolume(partial_real_shape=domain_shape)
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=pml_cells)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    cladding = fdtdx.UniformMaterialObject(
        name="cladding",
        partial_real_shape=(None, None, None),
        material=background_material,
    )
    constraints.extend(cladding.same_position_and_size(volume))
    objects.append(cladding)

    return objects, constraints, config, volume


def add_gds_geometry(
    objects: list,
    constraints: list,
    gds_path,
    cell_name: str,
    layer_specs: list,
    materials: dict,
    volume,
    gds_center: tuple[float, float],
) -> None:
    """Import GDS polygons and add extruded layer objects to the scene in-place.

    Args:
        objects: Scene object list to extend.
        constraints: Scene constraint list to extend.
        gds_path: Path to the ``.gds`` file.
        cell_name: GDS cell containing the device polygons.
        layer_specs: List of :class:`~fdtdx.objects.static_material.gds_layer_stack.GDSLayerSpec`.
        materials: Dict mapping material name strings to :class:`fdtdx.Material`.
        volume: :class:`fdtdx.SimulationVolume` used for size/position constraints.
        gds_center: GDS (x, y) coordinate in metres mapped to the simulation centre.
    """
    from fdtdx.objects.static_material.gds_layer_stack import gds_layer_stack

    gds_objs, gds_cons = gds_layer_stack(
        gds_source=gds_path,
        cell_name=cell_name,
        layers=layer_specs,
        materials=materials,
        simulation_volume=volume,
        gds_center=gds_center,
    )
    objects.extend(gds_objs)
    constraints.extend(gds_cons)


def add_mode_source(
    objects: list,
    constraints: list,
    port: dict,
    name: str,
    wave_character,
    temporal_profile,
    volume,
    domain_shape: tuple[float, float, float],
    gds_center: tuple[float, float],
    span_y: float,
    mode_index: int = 0,
    filter_pol: str | None = "te",
    direction: str = "+",
) -> None:
    """Add a :class:`fdtdx.ModePlaneSource` placed at a port position.

    Args:
        objects: Scene object list to extend.
        constraints: Scene constraint list to extend.
        port: Port dict with keys ``x_m``, ``y_m`` in metres.
        name: Unique object name for this source.
        wave_character: :class:`fdtdx.WaveCharacter` (wavelength / frequency).
        temporal_profile: Temporal envelope (e.g. :class:`fdtdx.GaussianPulseProfile`).
        volume: :class:`fdtdx.SimulationVolume` reference for constraints.
        domain_shape: (x, y, z) total domain in metres.
        gds_center: GDS (x, y) in metres mapped to the simulation centre.
        span_y: Transverse (y) extent of the source cross-section in metres.
        mode_index: Waveguide mode order (0 = fundamental).
        filter_pol: Polarisation filter ``"te"``, ``"tm"``, or ``None``.
        direction: Propagation direction ``"+"`` or ``"-"``.
    """
    import fdtdx

    sim_x, sim_y = port_to_sim_coords(port, domain_shape, gds_center)
    source = fdtdx.ModePlaneSource(
        name=name,
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, span_y, None),
        wave_character=wave_character,
        temporal_profile=temporal_profile,
        direction=direction,
        mode_index=mode_index,
        filter_pol=filter_pol,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(2,)),
            source.place_at_center(volume, axes=(2,)),
            fdtdx.RealCoordinateConstraint(
                object=name,
                axes=(0,),
                sides=("-",),
                coordinates=(sim_x,),
            ),
            fdtdx.RealCoordinateConstraint(
                object=name,
                axes=(1,),
                sides=("-",),
                coordinates=(sim_y - span_y / 2,),
            ),
        ]
    )
    objects.append(source)


def add_mode_detector(
    objects: list,
    constraints: list,
    port: dict,
    name: str,
    wave_characters: tuple,
    volume,
    domain_shape: tuple[float, float, float],
    gds_center: tuple[float, float],
    span_y: float,
    mode_index: int = 0,
    filter_pol: str | None = "te",
    direction: str = "+",
    scaling_mode: str = "pulse",
) -> None:
    """Add a :class:`fdtdx.ModeOverlapDetector` placed at a port position.

    Args:
        objects: Scene object list to extend.
        constraints: Scene constraint list to extend.
        port: Port dict with keys ``x_m``, ``y_m`` in metres.
        name: Unique object name for this detector.
        wave_characters: Tuple of :class:`fdtdx.WaveCharacter` objects.
        volume: :class:`fdtdx.SimulationVolume` reference for constraints.
        domain_shape: (x, y, z) total domain in metres.
        gds_center: GDS (x, y) in metres mapped to the simulation centre.
        span_y: Transverse (y) extent in metres.
        mode_index: Waveguide mode order.
        filter_pol: Polarisation filter or ``None``.
        direction: Propagation direction ``"+"`` or ``"-"``.
        scaling_mode: ``"pulse"`` (default) or ``"continuous"``.
    """
    import fdtdx

    sim_x, sim_y = port_to_sim_coords(port, domain_shape, gds_center)
    det = fdtdx.ModeOverlapDetector(
        name=name,
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, span_y, None),
        wave_characters=wave_characters,
        direction=direction,
        mode_index=mode_index,
        filter_pol=filter_pol,
        scaling_mode=scaling_mode,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(2,)),
            det.place_at_center(volume, axes=(2,)),
            fdtdx.RealCoordinateConstraint(
                object=name,
                axes=(0,),
                sides=("-",),
                coordinates=(sim_x,),
            ),
            fdtdx.RealCoordinateConstraint(
                object=name,
                axes=(1,),
                sides=("-",),
                coordinates=(sim_y - span_y / 2,),
            ),
        ]
    )
    objects.append(det)


def extend_gds_with_port_stubs(
    gds_path: Path,
    cell_name: str,
    ports: dict,
    layer_specs: list,
    gds_center: tuple[float, float],
    domain_shape: tuple[float, float, float],
) -> Path:
    """Return path to a temp GDS with waveguide stub rectangles added at each port.

    Extends each port waveguide from the port position to the domain boundary,
    ensuring the mode source injects into a continuous waveguide rather than a
    Si/cladding discontinuity.

    Args:
        gds_path: Path to the original ``.gds`` file.
        cell_name: GDS cell to modify.
        ports: Port dict ``{port_name: {"x_m", "y_m", "width_m", "orientation"}}``.
        layer_specs: Layer specs from which GDS layer/datatype are extracted.
        gds_center: GDS (x, y) in metres mapped to simulation centre.
        domain_shape: (x, y, z) total domain in metres.

    Returns:
        Path to a temporary ``.gds`` file with stubs added.
    """
    import gdstk

    lib = gdstk.read_gds(str(gds_path))
    real_cells = {c.name: c for c in lib.cells if not c.name.startswith("$$$")}
    target = real_cells[cell_name]

    gds_cx_um = gds_center[0] * 1e6
    domain_x_um = domain_shape[0] * 1e6
    left_um = gds_cx_um - domain_x_um / 2
    right_um = gds_cx_um + domain_x_um / 2

    for spec in layer_specs:
        layer = spec.gds_layer
        datatype = getattr(spec, "gds_datatype", 0)
        for port in ports.values():
            gds_x = port["x_m"] * 1e6
            gds_y = port["y_m"] * 1e6
            half_w = port["width_m"] * 1e6 / 2
            if abs(port["orientation"] - 180.0) < 1.0:
                rect = gdstk.rectangle(
                    (left_um, gds_y - half_w),
                    (gds_x, gds_y + half_w),
                    layer=layer,
                    datatype=datatype,
                )
            else:
                rect = gdstk.rectangle(
                    (gds_x, gds_y - half_w),
                    (right_um, gds_y + half_w),
                    layer=layer,
                    datatype=datatype,
                )
            target.add(rect)

    tmp = tempfile.NamedTemporaryFile(suffix=".gds", delete=False)
    lib.write_gds(tmp.name)
    return Path(tmp.name)


def add_phasor_monitors(
    objects: list,
    constraints: list,
    volume,
    wave_character,
    z_height: float,
) -> None:
    """Add a top-view XY frequency-domain field monitor to the scene in-place.

    Args:
        objects: Scene object list to extend.
        constraints: Scene constraint list to extend.
        volume: :class:`fdtdx.SimulationVolume` reference.
        wave_character: :class:`fdtdx.WaveCharacter` giving the monitoring wavelength.
        z_height: z coordinate in simulation space (metres) for the XY slice.
    """
    import fdtdx

    det_xy = fdtdx.PhasorDetector(
        name="phasor_xy",
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave_character,),
        components=("Ey",),
        reduce_volume=False,
        plot=False,
        scaling_mode="continuous",
    )
    objects.append(det_xy)
    constraints += [
        det_xy.same_size(volume, axes=(0, 1)),
        det_xy.place_at_center(volume, axes=(0, 1)),
        fdtdx.RealCoordinateConstraint(
            object="phasor_xy",
            axes=(2,),
            sides=("-",),
            coordinates=(z_height,),
        ),
    ]


def build_scene(
    *,
    gds_path,
    cell_name: str,
    layer_specs: list,
    materials: dict,
    ports: dict,
    source_port: str,
    detector_ports: list[tuple[str, str]],
    gds_center: tuple[float, float],
    domain_shape: tuple[float, float, float],
    pml_cells: int,
    background_material,
    wave_char,
    temporal_profile,
    grid,
    sim_time: float,
    source_span_y: float,
    norm_det_dx: float = 30e-9,
    norm_det_name: str = "det_source",
    with_port_stubs: bool = True,
    with_phasor_monitors: bool = False,
    phasor_z_height: float | None = None,
) -> tuple:
    """Build a complete fdtdx scene for a GDS-imported SOI device.

    Combines :func:`build_domain`, :func:`add_gds_geometry`,
    :func:`add_mode_source`, :func:`add_mode_detector`, and
    :func:`add_phasor_monitors` into a single call.

    Args:
        gds_path: Path to the ``.gds`` file.
        cell_name: GDS cell name.
        layer_specs: List of :class:`~fdtdx.objects.static_material.gds_layer_stack.GDSLayerSpec`.
        materials: Dict mapping material names to :class:`fdtdx.Material`.
        ports: Port dict ``{port_name: {"x_m", "y_m", "width_m", "orientation"}}``.
        source_port: Key in ``ports`` to use as the excitation port.
        detector_ports: ``[(detector_name, port_key), ...]``.
        gds_center: GDS (x, y) in metres mapped to the simulation centre.
        domain_shape: (x, y, z) total domain in metres including PML.
        pml_cells: PML thickness in grid cells on every face.
        background_material: :class:`fdtdx.Material` for the cladding fill.
        wave_char: :class:`fdtdx.WaveCharacter` for the source and all detectors.
        temporal_profile: Source temporal envelope.
        grid: :class:`fdtdx.RectilinearGrid` or :class:`fdtdx.UniformGrid`.
        sim_time: Total simulation time in seconds.
        source_span_y: Transverse (y) span in metres for the source and detectors.
        norm_det_dx: x-offset in metres from the source port for the normalisation detector.
        norm_det_name: Name for the normalisation detector.
        with_port_stubs: Extend port waveguides to the domain boundary.
        with_phasor_monitors: Add an XY :class:`fdtdx.PhasorDetector` monitor.
        phasor_z_height: z coordinate (metres) for the XY phasor slice.

    Returns:
        ``(objects, constraints, config, volume)``
    """
    objects, constraints, config, volume = build_domain(
        domain_shape=domain_shape,
        pml_cells=pml_cells,
        background_material=background_material,
        grid=grid,
        sim_time=sim_time,
    )

    import_gds = gds_path
    if with_port_stubs:
        import_gds = extend_gds_with_port_stubs(
            gds_path=gds_path,
            cell_name=cell_name,
            ports=ports,
            layer_specs=layer_specs,
            gds_center=gds_center,
            domain_shape=domain_shape,
        )

    add_gds_geometry(
        objects,
        constraints,
        gds_path=import_gds,
        cell_name=cell_name,
        layer_specs=layer_specs,
        materials=materials,
        volume=volume,
        gds_center=gds_center,
    )

    add_mode_source(
        objects,
        constraints,
        port=ports[source_port],
        name="source",
        wave_character=wave_char,
        temporal_profile=temporal_profile,
        volume=volume,
        domain_shape=domain_shape,
        gds_center=gds_center,
        span_y=source_span_y,
    )

    norm_port = {**ports[source_port], "x_m": ports[source_port]["x_m"] + norm_det_dx}
    add_mode_detector(
        objects,
        constraints,
        port=norm_port,
        name=norm_det_name,
        wave_characters=(wave_char,),
        volume=volume,
        domain_shape=domain_shape,
        gds_center=gds_center,
        span_y=source_span_y,
    )

    for det_name, port_key in detector_ports:
        add_mode_detector(
            objects,
            constraints,
            port=ports[port_key],
            name=det_name,
            wave_characters=(wave_char,),
            volume=volume,
            domain_shape=domain_shape,
            gds_center=gds_center,
            span_y=source_span_y,
        )

    if with_phasor_monitors:
        add_phasor_monitors(
            objects,
            constraints,
            volume=volume,
            wave_character=wave_char,
            z_height=phasor_z_height,
        )

    return objects, constraints, config, volume
