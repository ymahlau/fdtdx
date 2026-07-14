from typing import TYPE_CHECKING, Literal, Self

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.axis import get_oriented_transverse_axes
from fdtdx.core.grid import calculate_time_offset_yee
from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field, private_field
from fdtdx.core.linalg import get_wave_vector_raw, rotate_vector
from fdtdx.core.misc import normalize_polarization_for_source
from fdtdx.dispersion import effective_inv_permittivity
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.sources.tfsf import (
    TFSFPlaneSource,
    _build_dispersive_H_filter,
    _source_impedance,
    _tfsf_inject_E_face,
    _tfsf_inject_H_face,
)

if TYPE_CHECKING:
    from fdtdx.fdtd.container import ObjectContainer


@autoinit
class TFSFPlaneSourceRegion(TFSFPlaneSource):
    """Total-Field/Scattered-Field (TFSF) *box* source.

    A uniform plane wave is injected on the bounding faces of a rectangular
    box so that the **total field lives inside** the box and **only the
    scattered field exists outside**. A scatterer placed inside the box sees a
    clean incident plane wave; the scattered field radiates outward into the
    scattered-field region where it can be absorbed by PML or measured.

    Unlike :class:`~fdtdx.objects.sources.tfsf.TFSFPlaneSource` (a single-plane,
    one-way launcher), this is a volume object with an explicit
    :attr:`propagation_axis`. Every active face evaluates *one coherent* plane
    wave sharing a single spatial phase origin (the box lower corner): the
    entry faces launch the wave and the exit faces terminate it (sign-flipped),
    keeping the propagation direction of ``k = E x H`` intact.

    Faces are injected on the propagation axis (always, both caps) and on each
    transverse axis that is *confined*. A transverse axis listed in
    :attr:`periodic_axes` is treated as "wrap-around": no correction is applied
    on its faces and the periodic boundaries close the total-field region in
    that direction. A wrap axis must have periodic/Bloch boundaries on both
    sides and the box must span the full domain along it (validated at
    placement time).

    Media parity: the field injection handles isotropic, diagonally
    anisotropic, and fully anisotropic tensors as well as dispersive
    backgrounds — the same code paths as ``TFSFPlaneSource``. As with the
    single-plane source, the propagation time-of-flight requires the material
    at the source faces to be locally isotropic (a genuinely anisotropic face
    raises ``NotImplementedError`` from ``calculate_time_offset_yee``).
    """

    #: the propagation axis (0=x, 1=y, 2=z). Overrides the size-1-axis inference
    #: of :class:`DirectionalPlaneSourceBase`, which does not apply to a box.
    propagation_axis: int = frozen_field()

    #: transverse axes treated as periodic/wrap (no TFSF correction on their
    #: faces). Must be a subset of the two transverse axes; the propagation axis
    #: may never appear here.
    periodic_axes: tuple[int, ...] = frozen_field(default=())

    #: the amplitude of the uniform plane wave
    amplitude: float = frozen_field(default=1.0)

    #: the electric polarization vector
    fixed_E_polarization_vector: tuple[float, float, float] | None = frozen_field(default=None)

    #: the magnetic polarization vector
    fixed_H_polarization_vector: tuple[float, float, float] | None = frozen_field(default=None)

    # Per-face static metadata (excluded from pytree traversal).
    _face_normal_axes: tuple[int, ...] | None = frozen_private_field(default=None)
    _face_signs: tuple[int, ...] | None = frozen_private_field(default=None)
    # A TFSF box is a two-node (staggered) connecting condition: the tangential E
    # correction sits on the total-field E node adjacent to the surface, and the
    # tangential H correction sits half a cell further, on the scattered-side H
    # node. These are DIFFERENT Yee cells (see ``apply``), so each face stores an
    # E-plane slice (where update_E writes) and an H-plane slice (where update_H
    # writes). Co-locating them — as a naive single-plane injector would — leaves
    # a net magnetic correction at the box edges/corners that never radiates away
    # and accumulates as a static field (the "hot corner" artifact).
    _face_E_slice_tuples: tuple | None = frozen_private_field(default=None)
    _face_H_slice_tuples: tuple | None = frozen_private_field(default=None)

    # Per-face traced leaves (parallel to the metadata tuples above).
    _face_incident_E: tuple | None = private_field(default=None)
    _face_incident_H: tuple | None = private_field(default=None)
    _face_time_offset_E: tuple | None = private_field(default=None)
    _face_time_offset_H: tuple | None = private_field(default=None)
    _face_H_filter: tuple | None = private_field(default=None)

    def _region_resolution(self) -> float:
        config = self._config
        if config.has_nonuniform_grid:
            assert config.resolved_grid is not None
            return config.resolved_grid.min_spacing
        return config.uniform_spacing()

    def _box_edges(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Physical edge coordinates spanning the box, origin at the box corner.

        Returns three 1-D arrays (one per axis) shifted so the box lower corner
        is the local origin. This is the coherent coordinate system shared by
        every face so a single plane-wave phase sweeps the whole box.
        """
        config = self._config
        if config.has_nonuniform_grid:
            grid = config.resolved_grid
            assert grid is not None
            edges = []
            for axis in range(3):
                lower, upper = self.grid_slice_tuple[axis]
                e = grid.edges(axis)[lower : upper + 1]
                edges.append(e - e[0])
            return edges[0], edges[1], edges[2]
        spacing = config.uniform_spacing()
        edges = [jnp.arange(self.grid_shape[axis] + 1, dtype=config.dtype) * spacing for axis in range(3)]
        return edges[0], edges[1], edges[2]

    def _normal_cell_window(self, normal_axis: int, abs_cell_index: int) -> jax.Array:
        """Two-edge window (box-corner origin) for one cell along ``normal_axis``.

        Used to phase the incident field at a specific Yee plane. ``abs_cell_index``
        is an absolute grid index and may lie one cell outside the box (the H node
        just below the min face, or the E node just above the max face), so the
        window is built from the resolved grid / spacing rather than ``_box_edges``
        (which only spans the box interior).
        """
        config = self._config
        lower = self.grid_slice_tuple[normal_axis][0]
        if config.has_nonuniform_grid:
            grid = config.resolved_grid
            assert grid is not None
            edges = grid.edges(normal_axis)
            origin = edges[lower]
            return edges[abs_cell_index : abs_cell_index + 2] - origin
        spacing = config.uniform_spacing()
        rel = abs_cell_index - lower
        return jnp.asarray([rel * spacing, (rel + 1) * spacing], dtype=config.dtype)

    def _active_faces(self) -> list[tuple[int, Literal["-", "+"]]]:
        """Enumerate ``(normal_axis, side)`` for every injected face.

        Always both propagation caps; both faces of each confined transverse
        axis; nothing for a transverse axis listed in ``periodic_axes``.
        """
        faces: list[tuple[int, Literal["-", "+"]]] = [
            (self.propagation_axis, "-"),
            (self.propagation_axis, "+"),
        ]
        for t_axis in get_oriented_transverse_axes(self.propagation_axis):
            if t_axis in tuple(self.periodic_axes):
                continue
            faces.append((t_axis, "-"))
            faces.append((t_axis, "+"))
        return faces

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
        electric_conductivity: jax.Array | None = None,
        dispersive_c4: jax.Array | None = None,
    ) -> Self:
        del electric_conductivity
        config = self._config
        p_axis = self.propagation_axis
        axes_tpl = (self.horizontal_axis, self.vertical_axis, self.propagation_axis)

        # polarization and (possibly tilted) wave vector — shared by all faces
        e_pol_raw, h_pol_raw = normalize_polarization_for_source(
            direction=self.direction,
            propagation_axis=p_axis,
            fixed_E_polarization_vector=self.fixed_E_polarization_vector,
            fixed_H_polarization_vector=self.fixed_H_polarization_vector,
            dtype=config.dtype,
        )
        wave_vector_raw = get_wave_vector_raw(direction=self.direction, propagation_axis=p_axis, dtype=config.dtype)

        key, subkey = jax.random.split(key)
        azimuth, elevation = self._get_azimuth_elevation(subkey)
        wave_vector = rotate_vector(wave_vector_raw, azimuth, elevation, axes_tpl)
        e_pol = rotate_vector(e_pol_raw, azimuth, elevation, axes_tpl)
        h_pol = rotate_vector(h_pol_raw, azimuth, elevation, axes_tpl)

        box_edges = self._box_edges()
        # Single coherent phase origin (box lower corner) reused for every face.
        center_physical = jnp.zeros(3, dtype=config.dtype)
        resolution = self._region_resolution()
        omega_c = 2.0 * np.pi * self.wave_character.get_frequency()

        box_slice = self.grid_slice_tuple

        normal_axes: list[int] = []
        signs: list[int] = []
        e_slice_tuples: list = []
        h_slice_tuples: list = []
        inc_E_list: list = []
        inc_H_list: list = []
        toff_E_list: list = []
        toff_H_list: list = []
        filt_list: list = []

        for normal_axis, side in self._active_faces():
            lower, upper = box_slice[normal_axis]
            # Two-node connecting condition (see class docstring / _face_*_slice_tuples):
            #   * the tangential-E correction sits on the total-field E node adjacent
            #     to the TFSF surface,
            #   * the tangential-H correction sits half a cell further on the
            #     scattered-side H node.
            # Because H_a/H_b are staggered +1/2 along the normal, the H node is at
            # (E node - 1) for a min face and coincident-index-minus-nothing for a
            # max face; concretely:
            if side == "-":
                sign = 1  # min face
                e_abs = lower  # total-field E node on the surface
                h_abs = lower - 1  # scattered-side H node, half a cell outside
            else:
                sign = -1  # max face
                e_abs = upper  # scattered-side E node just past the last total node
                h_abs = upper - 1  # total-field H node on the surface

            e_slice_tuple = self._normal_plane_slice(normal_axis, e_abs)
            h_slice_tuple = self._normal_plane_slice(normal_axis, h_abs)
            e_face_slice = tuple(slice(a, b) for (a, b) in e_slice_tuple)
            face_shape = tuple(b - a for (a, b) in e_slice_tuple)  # H plane has the same shape

            # Incident amplitude is sampled from the medium at the E node (the
            # tangential-E correction plane). E_inc is material-independent; only the
            # H_inc amplitude carries the impedance, and the source region is assumed
            # locally homogeneous across the half-cell E/H offset.
            inv_eps_face = inv_permittivities[:, *e_face_slice]
            if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
                inv_mu_face = inv_permeabilities[:, *e_face_slice]
            else:
                inv_mu_face = inv_permeabilities
            inv_eps_inf_face = inv_eps_face  # raw ε∞ before any carrier correction

            # dispersive carrier-frequency correction (matches the single-plane source)
            c1_s = c2_s = c3_s = c4_s = None
            if dispersive_c1 is not None and dispersive_c2 is not None and dispersive_c3 is not None:
                c1_s = dispersive_c1[:, :, *e_face_slice]
                c2_s = dispersive_c2[:, :, *e_face_slice]
                c3_s = dispersive_c3[:, :, *e_face_slice]
                c4_s = None if dispersive_c4 is None else dispersive_c4[:, :, *e_face_slice]
                inv_eps_face = effective_inv_permittivity(
                    inv_eps=inv_eps_face,
                    c1=c1_s,
                    c2=c2_s,
                    c3=c3_s,
                    omega=omega_c,
                    dt=config.time_step_duration,
                    c4=c4_s,
                )

            impedance = _source_impedance(inv_eps_face, inv_mu_face, e_pol, h_pol)

            E_inc = jnp.broadcast_to(self.amplitude * e_pol[:, None, None, None], (3, *face_shape)).astype(config.dtype)
            H_face = self.amplitude * h_pol[:, None, None, None] / impedance
            H_inc = jnp.broadcast_to(H_face, (3, *face_shape)).astype(config.dtype)

            # Phase the incident field at the exact Yee plane each correction uses:
            # incident E (for the H correction) at the E node, incident H (for the E
            # correction) at the H node. Sampling at the naive single-plane position
            # is what left the uncancelled static corner field.
            time_offset_E, _ = self._plane_time_offsets(
                normal_axis,
                e_abs,
                box_edges,
                wave_vector,
                e_pol,
                h_pol,
                inv_permittivities,
                inv_permeabilities,
                resolution,
                center_physical,
            )
            _, time_offset_H = self._plane_time_offsets(
                normal_axis,
                h_abs,
                box_edges,
                wave_vector,
                e_pol,
                h_pol,
                inv_permittivities,
                inv_permeabilities,
                resolution,
                center_physical,
            )

            filt = None
            if c1_s is not None and c2_s is not None and c3_s is not None:
                filt = _build_dispersive_H_filter(
                    temporal_profile=self.temporal_profile,
                    wave_character=self.wave_character,
                    dt=config.time_step_duration,
                    num_time_steps=config.time_steps_total,
                    c1_slice=c1_s,
                    c2_slice=c2_s,
                    c3_slice=c3_s,
                    inv_eps_inf_slice=inv_eps_inf_face,
                    dtype=config.dtype,
                    c4_slice=c4_s,
                )

            normal_axes.append(normal_axis)
            signs.append(sign)
            e_slice_tuples.append(e_slice_tuple)
            h_slice_tuples.append(h_slice_tuple)
            inc_E_list.append(E_inc)
            inc_H_list.append(H_inc)
            toff_E_list.append(time_offset_E)
            toff_H_list.append(time_offset_H)
            filt_list.append(filt)

        self = self.aset("_face_normal_axes", tuple(normal_axes), create_new_ok=True)
        self = self.aset("_face_signs", tuple(signs), create_new_ok=True)
        self = self.aset("_face_E_slice_tuples", tuple(e_slice_tuples), create_new_ok=True)
        self = self.aset("_face_H_slice_tuples", tuple(h_slice_tuples), create_new_ok=True)
        self = self.aset("_face_incident_E", tuple(inc_E_list), create_new_ok=True)
        self = self.aset("_face_incident_H", tuple(inc_H_list), create_new_ok=True)
        self = self.aset("_face_time_offset_E", tuple(toff_E_list), create_new_ok=True)
        self = self.aset("_face_time_offset_H", tuple(toff_H_list), create_new_ok=True)
        self = self.aset("_face_H_filter", tuple(filt_list), create_new_ok=True)
        return self

    def _normal_plane_slice(self, normal_axis: int, abs_index: int) -> tuple:
        """Return the box slice with ``normal_axis`` restricted to the single cell ``abs_index``."""
        slice_tuple = list(self.grid_slice_tuple)
        slice_tuple[normal_axis] = (abs_index, abs_index + 1)
        return tuple(slice_tuple)

    def _plane_time_offsets(
        self,
        normal_axis: int,
        abs_index: int,
        box_edges: tuple[jax.Array, jax.Array, jax.Array],
        wave_vector: jax.Array,
        e_pol: jax.Array,
        h_pol: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        resolution: float,
        center_physical: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Yee time offsets for the incident field phased at the cell ``abs_index``.

        The transverse coordinates come from the shared ``box_edges`` (box-corner
        origin) so every face and plane stays phase-coherent; only the normal-axis
        window moves to ``abs_index`` (which may be one cell outside the box).
        """
        config = self._config
        coord_edges = list(box_edges)
        coord_edges[normal_axis] = self._normal_cell_window(normal_axis, abs_index)
        coord_edges = (coord_edges[0], coord_edges[1], coord_edges[2])
        plane_slice = tuple(slice(a, b) for (a, b) in self._normal_plane_slice(normal_axis, abs_index))
        inv_eps = inv_permittivities[:, *plane_slice]
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_mu = inv_permeabilities[:, *plane_slice]
        else:
            inv_mu = inv_permeabilities
        return calculate_time_offset_yee(
            center=jnp.zeros(2, dtype=config.dtype),
            wave_vector=wave_vector,
            inv_permittivities=inv_eps,
            inv_permeabilities=inv_mu,
            resolution=resolution,
            time_step_duration=config.time_step_duration,
            e_polarization=e_pol,
            h_polarization=h_pol,
            coordinate_edges=coord_edges,
            center_physical=center_physical,
        )

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permeabilities
        normal_axes = self._face_normal_axes
        signs = self._face_signs
        slice_tuples = self._face_E_slice_tuples  # E correction lives on the total-field E node
        incident_H = self._face_incident_H
        time_offset_H = self._face_time_offset_H
        h_filter = self._face_H_filter
        if (
            normal_axes is None
            or signs is None
            or slice_tuples is None
            or incident_H is None
            or time_offset_H is None
            or h_filter is None
        ):
            raise Exception("Need to apply random key before calling update")

        delta_t = self._config.time_step_duration
        for i, normal_axis in enumerate(normal_axes):
            sign = -signs[i] if inverse else signs[i]
            face_slice_tuple = slice_tuples[i]
            grid_slice = tuple(slice(a, b) for (a, b) in face_slice_tuple)
            start_index = face_slice_tuple[normal_axis][0]
            c = self._config.courant_number * self._metric_scale_at_plane("backward", normal_axis, start_index)
            E = _tfsf_inject_E_face(
                E,
                grid_slice=grid_slice,
                normal_axis=normal_axis,
                sign=sign,
                incident_H=incident_H[i],
                time_offset_H=time_offset_H[i],
                temporal_H_filter=h_filter[i],
                inv_permittivities=inv_permittivities,
                c=c,
                time_step=time_step,
                delta_t=delta_t,
                temporal_profile=self.temporal_profile,
                wave_character=self.wave_character,
                static_amplitude_factor=self.static_amplitude_factor,
            )
        return E

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities
        normal_axes = self._face_normal_axes
        signs = self._face_signs
        slice_tuples = self._face_H_slice_tuples  # H correction lives on the scattered-side H node
        incident_E = self._face_incident_E
        time_offset_E = self._face_time_offset_E
        if normal_axes is None or signs is None or slice_tuples is None or incident_E is None or time_offset_E is None:
            raise Exception("Need to apply random key before calling update")

        delta_t = self._config.time_step_duration
        for i, normal_axis in enumerate(normal_axes):
            sign = -signs[i] if inverse else signs[i]
            face_slice_tuple = slice_tuples[i]
            grid_slice = tuple(slice(a, b) for (a, b) in face_slice_tuple)
            start_index = face_slice_tuple[normal_axis][0]
            c = self._config.courant_number * self._metric_scale_at_plane("forward", normal_axis, start_index)
            H = _tfsf_inject_H_face(
                H,
                grid_slice=grid_slice,
                normal_axis=normal_axis,
                sign=sign,
                incident_E=incident_E[i],
                time_offset_E=time_offset_E[i],
                inv_permeabilities=inv_permeabilities,
                c=c,
                time_step=time_step,
                delta_t=delta_t,
                temporal_profile=self.temporal_profile,
                wave_character=self.wave_character,
                static_amplitude_factor=self.static_amplitude_factor,
            )
        return H

    def validate_placement(self, objects: "ObjectContainer") -> list[str]:
        """Validate the box against the surrounding boundaries after placement.

        For every axis marked periodic/wrap: it must be a transverse axis (not
        the propagation axis), have periodic/Bloch boundaries on both sides, and
        the box must span the full simulation domain on that axis. Tilted
        incidence with a nonzero wavevector component along a periodic axis is
        rejected in v1 (it would require a matching Bloch phase).
        """
        errors: list[str] = []
        p_axis = self.propagation_axis
        transverse = get_oriented_transverse_axes(p_axis)
        periodic = tuple(self.periodic_axes)

        if p_axis in periodic:
            errors.append(f"propagation axis {p_axis} cannot be a periodic/wrap axis; it always needs TFSF correction")
        for a in periodic:
            if a not in (0, 1, 2):
                errors.append(f"periodic/wrap axis {a} is not a valid axis (must be 0, 1 or 2)")
            elif a != p_axis and a not in transverse:
                errors.append(f"periodic/wrap axis {a} must be one of the transverse axes {transverse}")

        box_slice = self.grid_slice_tuple
        volume_slice = objects.volume.grid_slice_tuple

        # Every confined (non-periodic) face needs one cell of scattered-field margin
        # on each side: the staggered connecting condition writes the tangential-H
        # correction on the node one cell below the min face (index lower-1) and the
        # tangential-E correction on the node one cell above the max face (index
        # upper). Without the margin those planes fall outside the domain.
        for a in range(3):
            if a in periodic:
                continue
            box_lo, box_hi = box_slice[a]
            vol_lo, vol_hi = volume_slice[a]
            if box_lo - 1 < vol_lo or box_hi + 1 > vol_hi:
                errors.append(
                    f"TFSF box has no scattered-field margin on axis {a} (box {box_slice[a]} vs domain "
                    f"{volume_slice[a]}); a confined TFSF face needs at least one cell between the box and the "
                    f"domain edge on both sides. Shrink the box or enlarge the domain/boundary on axis {a}."
                )

        for a in periodic:
            if a == p_axis or a not in (0, 1, 2):
                continue
            wrap_boundaries = [b for b in objects.boundary_objects if b.axis == a and isinstance(b, BlochBoundary)]
            periodic_sides = {b.direction for b in wrap_boundaries}
            if periodic_sides != {"-", "+"}:
                errors.append(
                    f"axis {a} is marked periodic/wrap but does not have periodic (Bloch) boundaries on both "
                    f"sides; add periodic boundaries on axis {a} or make it a confined TFSF axis"
                )
            elif any(b.needs_complex_fields for b in wrap_boundaries):
                errors.append(
                    f"axis {a} uses a phase-shifted Bloch boundary; the TFSF box supports only plain periodic "
                    f"(zero Bloch phase) wrap axes in v1 (matched oblique incidence is not yet supported)"
                )
            if box_slice[a] != volume_slice[a]:
                errors.append(
                    f"axis {a} is marked periodic/wrap but the TFSF box does not span the full simulation domain "
                    f"on that axis (box {box_slice[a]} vs volume {volume_slice[a]}); a wrap axis must span the domain"
                )

        if len(periodic) > 0:
            if self.max_angle_random_offset != 0.0:
                errors.append(
                    "random angle offset (max_angle_random_offset) is not supported together with periodic/wrap "
                    "axes in v1; set max_angle_random_offset=0 (normal or in-plane incidence only)"
                )
            axes_tpl = (self.horizontal_axis, self.vertical_axis, self.propagation_axis)
            wave_vector = rotate_vector(
                get_wave_vector_raw(self.direction, p_axis, dtype=self._config.dtype),
                self.azimuth_radians,
                self.elevation_radians,
                axes_tpl,
            )
            for a in periodic:
                if a in (0, 1, 2) and a != p_axis and abs(float(wave_vector[a])) > 1e-6:
                    errors.append(
                        f"tilted incidence produces a nonzero wavevector component along periodic axis {a}; "
                        f"this requires a matching Bloch vector and is not supported in v1 (use incidence with "
                        f"zero component along that axis, or make it a confined TFSF axis)"
                    )
        return errors
