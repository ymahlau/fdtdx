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
    """GDS port -> absolute sim coords: the domain runs 0..domain_size, centred on gds_center."""
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
    """Create SimulationVolume, PML boundaries, background fill, and SimulationConfig."""
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
    """Import GDS polygons and add extruded layer objects to the scene in-place."""
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


def _place_at_port(obj, name: str, port: dict, volume, domain_shape, gds_center, span_y: float) -> list:
    """Constraints placing ``obj`` at ``port``'s position, spanning ``volume`` in z."""
    import fdtdx

    sim_x, sim_y = port_to_sim_coords(port, domain_shape, gds_center)
    return [
        obj.same_size(volume, axes=(2,)),
        obj.place_at_center(volume, axes=(2,)),
        fdtdx.RealCoordinateConstraint(object=name, axes=(0,), sides=("-",), coordinates=(sim_x,)),
        fdtdx.RealCoordinateConstraint(object=name, axes=(1,), sides=("-",), coordinates=(sim_y - span_y / 2,)),
    ]


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
    """Add a :class:`fdtdx.ModePlaneSource` placed at a port position."""
    import fdtdx

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
    constraints.extend(_place_at_port(source, name, port, volume, domain_shape, gds_center, span_y))
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
    """Add a :class:`fdtdx.ModeOverlapDetector` placed at a port position."""
    import fdtdx

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
    constraints.extend(_place_at_port(det, name, port, volume, domain_shape, gds_center, span_y))
    objects.append(det)


def extend_gds_with_port_stubs(
    gds_path: Path,
    cell_name: str,
    ports: dict,
    layer_specs: list,
    gds_center: tuple[float, float],
    domain_shape: tuple[float, float, float],
) -> Path:
    """Return path to a temp GDS with each port waveguide extended to the domain boundary.

    Without this, the mode source would inject into a Si/cladding discontinuity
    instead of a continuous waveguide.
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
    """Add a top-view XY frequency-domain field monitor to the scene in-place."""
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
