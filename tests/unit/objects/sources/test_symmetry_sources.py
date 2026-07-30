"""Plane sources under ``config.symmetry``.

A symmetry-reduced simulation must inject exactly the part of the full-domain source that falls
inside the kept half/quadrant — same profile, same absolute amplitude. These tests compare the
injected fields of a reduced run against the same model run without symmetry, which needs no FDTD
time stepping: ``place_objects`` + ``apply_params`` already populate ``source._E`` / ``source._H``.

Covers both wall types (PEC and PMC), one and two symmetry axes, all three propagation axes, and
both propagation directions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from loguru import logger

import fdtdx
from fdtdx.core.jax.pytrees import autoinit
from fdtdx.objects.sources.linear_polarization import LinearlyPolarizedPlaneSource

_SPACING = 100e-9
_WAVELENGTH = 1e-6
_TRANSVERSE_CELLS = 20  # even, so every symmetric axis splits exactly
_PROPAGATION_CELLS = 12


def _axes(propagation_axis: int) -> tuple[int, int]:
    """(horizontal, vertical) transverse axes, in fdtdx's right-hand cyclic order."""
    return (propagation_axis + 1) % 3, (propagation_axis + 2) % 3


def _compatible_symmetry(propagation_axis: int, axes: str) -> tuple[int, int, int]:
    """Symmetry tuple whose walls match an E-along-horizontal plane wave.

    E is normal to the horizontal plane (even under PEC) and tangential to the vertical plane
    (even under PMC), so a mirror-symmetric beam needs PEC on the horizontal axis and PMC on the
    vertical one. ``axes`` selects "h", "v" or "hv".
    """
    horizontal, vertical = _axes(propagation_axis)
    symmetry = [0, 0, 0]
    if "h" in axes:
        symmetry[horizontal] = -1  # PEC
    if "v" in axes:
        symmetry[vertical] = 1  # PMC
    return symmetry[0], symmetry[1], symmetry[2]


@autoinit
class _RampSource(LinearlyPolarizedPlaneSource):
    """Source whose transverse profile is unique per cell, so any mirror/transpose shows up."""

    def _get_amplitude_raw(self, center):
        del center
        w = jnp.arange(self.grid_shape[self.horizontal_axis], dtype=self._config.dtype)
        h = jnp.arange(self.grid_shape[self.vertical_axis], dtype=self._config.dtype)
        return self._hv_to_grid(w[:, None] + 100.0 * h[None, :])


def _build(
    symmetry: tuple[int, int, int],
    propagation_axis: int = 2,
    direction: str = "+",
    source_kind: str = "gaussian",
    normalize_by_energy: bool = True,
    transverse_cells: tuple[int, int] | None = None,
    source_cells: tuple[int, int] | None = None,
    source_start: tuple[int, int] | None = None,
    **source_kwargs,
):
    """Build a model with one plane source spanning (or centered in) the transverse plane."""
    horizontal, vertical = _axes(propagation_axis)
    h_cells, v_cells = transverse_cells or (_TRANSVERSE_CELLS, _TRANSVERSE_CELLS)
    shape = [0, 0, 0]
    shape[propagation_axis] = _PROPAGATION_CELLS
    shape[horizontal] = h_cells
    shape[vertical] = v_cells

    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_SPACING),
        time=5e-15,
        dtype=jnp.float32,
        symmetry=symmetry,
    )
    volume = fdtdx.SimulationVolume(partial_grid_shape=(shape[0], shape[1], shape[2]))
    objects, constraints = [volume], []
    bound_dict, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=3), volume
    )
    constraints.extend(boundary_constraints)
    objects.extend(bound_dict.values())

    polarization = [0.0, 0.0, 0.0]
    polarization[horizontal] = 1.0
    partial_shape: list[int | None] = [None, None, None]
    partial_shape[propagation_axis] = 1
    if source_cells is not None:
        partial_shape[horizontal] = source_cells[0]
        partial_shape[vertical] = source_cells[1]
    shared = dict(
        name="src",
        partial_grid_shape=(partial_shape[0], partial_shape[1], partial_shape[2]),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        direction=direction,
        fixed_E_polarization_vector=(polarization[0], polarization[1], polarization[2]),
        normalize_by_energy=normalize_by_energy,
        **source_kwargs,
    )
    if source_kind == "gaussian":
        source = fdtdx.GaussianPlaneSource(radius=h_cells * _SPACING / 2, std=1 / 5, **shared)
    elif source_kind == "uniform":
        source = fdtdx.UniformPlaneSource(**shared)
    else:
        source = _RampSource(**shared)

    if source_start is None:
        constraints.append(source.place_at_center(volume, axes=(horizontal, vertical)))
    else:
        # Absolute placement, used to park a small source strictly inside the kept upper half.
        constraints.append(
            source.set_grid_coordinates(axes=(horizontal, vertical), sides=("-", "-"), coordinates=source_start)
        )
    if source_cells is None:
        constraints.append(source.same_size(volume, axes=(horizontal, vertical)))
    constraints.append(
        source.set_grid_coordinates(axes=(propagation_axis,), sides=("-",), coordinates=(_PROPAGATION_CELLS // 2,))
    )
    objects.append(source)
    return objects, constraints, config


def _place(symmetry, **kwargs):
    """Place + apply a model and return the resulting source object."""
    objects, constraints, config = _build(symmetry, **kwargs)
    key = jax.random.PRNGKey(0)
    container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, container, _ = fdtdx.apply_params(arrays, container, params, key=key)
    return next(obj for obj in container.objects if obj.name == "src")


def _kept_half(array: np.ndarray, symmetry: tuple[int, int, int], leading: int = 1) -> np.ndarray:
    """Slice the upper half along every symmetric axis (``leading`` non-spatial axes in front)."""
    index: list[slice] = [slice(None)] * (leading + 3)
    for axis in range(3):
        if symmetry[axis] != 0:
            index[leading + axis] = slice(array.shape[leading + axis] // 2, None)
    return array[tuple(index)]


class TestGaussianSourceUnderSymmetry:
    """A clipped Gaussian must stay centered on the symmetry plane, not on the kept quadrant."""

    @pytest.mark.parametrize("propagation_axis", [0, 1, 2])
    @pytest.mark.parametrize("axes", ["h", "v", "hv"])
    @pytest.mark.parametrize("direction", ["+", "-"])
    def test_matches_full_domain_kept_half(self, propagation_axis, axes, direction):
        symmetry = _compatible_symmetry(propagation_axis, axes)
        kwargs = dict(propagation_axis=propagation_axis, direction=direction)
        full = _place((0, 0, 0), **kwargs)
        reduced = _place(symmetry, **kwargs)

        for name in ("_E", "_H"):
            full_field = np.asarray(getattr(full, name))
            reduced_field = np.asarray(getattr(reduced, name))
            expected = _kept_half(full_field, symmetry)
            assert reduced_field.shape == expected.shape
            scale = np.abs(expected).max()
            assert scale > 0
            error = np.abs(reduced_field - expected).max() / scale
            assert error < 1e-6, f"{name}: reduced injection differs from the full-domain half by {error:.2e}"

    @pytest.mark.parametrize("axes", ["h", "v", "hv"])
    def test_peak_sits_on_the_symmetry_corner(self, axes):
        # The full-domain beam peaks at the domain center, i.e. on the symmetry plane(s), so the
        # reduced profile must peak in the very first cell of every straddled axis.
        symmetry = _compatible_symmetry(2, axes)
        reduced = _place(symmetry)
        profile = np.abs(np.asarray(reduced._E)[0, :, :, 0])
        peak = np.unravel_index(np.argmax(profile), profile.shape)
        horizontal, vertical = _axes(2)
        assert (peak[0] == 0) == (symmetry[horizontal] != 0)
        assert (peak[1] == 0) == (symmetry[vertical] != 0)

    @pytest.mark.parametrize("normalize_by_energy", [True, False])
    def test_absolute_amplitude_matches_full_domain(self, normalize_by_energy):
        # Guards the 2**k profile/energy normalization: both are sums over the source plane, which
        # the reduction shrinks by the plane multiplicity.
        symmetry = _compatible_symmetry(2, "hv")
        full = _place((0, 0, 0), normalize_by_energy=normalize_by_energy)
        reduced = _place(symmetry, normalize_by_energy=normalize_by_energy)
        expected = _kept_half(np.asarray(full._E), symmetry)
        ratio = np.abs(np.asarray(reduced._E)).max() / np.abs(expected).max()
        assert abs(ratio - 1.0) < 1e-5, f"amplitude ratio reduced/full = {ratio:.6f}"

    def test_source_inside_kept_half_is_untouched(self):
        # A source parked strictly inside the kept half is not clipped, so it keeps the full-domain
        # profile centered on itself — the symmetry-aware center must not move it.
        symmetry = _compatible_symmetry(2, "hv")
        kwargs = dict(source_cells=(6, 6), source_start=(12, 12))
        full = _place((0, 0, 0), **kwargs)
        reduced = _place(symmetry, **kwargs)
        assert reduced.grid_shape == full.grid_shape
        assert not any(reduced.straddles_symmetry_plane(a) for a in range(3))
        assert np.allclose(np.asarray(reduced._E), np.asarray(full._E), atol=1e-6)

    def test_source_touching_the_plane_from_above_is_not_mirror_extended(self):
        # Starts exactly at the symmetry plane but lies entirely in the kept half: it was never
        # clipped, so neither its center nor its normalization may assume a mirrored other half.
        symmetry = _compatible_symmetry(2, "hv")
        kwargs = dict(source_cells=(6, 6), source_start=(10, 10))
        full = _place((0, 0, 0), **kwargs)
        reduced = _place(symmetry, **kwargs)
        assert reduced.grid_slice_tuple[0][0] == 0 and reduced.grid_slice_tuple[1][0] == 0
        assert not any(reduced.straddles_symmetry_plane(a) for a in range(3))
        assert np.allclose(np.asarray(reduced._E), np.asarray(full._E), atol=1e-6)


class TestUniformSourceUnderSymmetry:
    """UniformPlaneSource has no profile center, but its energy normalization is still a plane sum."""

    @pytest.mark.parametrize("axes", ["h", "hv"])
    @pytest.mark.parametrize("normalize_by_energy", [True, False])
    def test_matches_full_domain_kept_half(self, axes, normalize_by_energy):
        symmetry = _compatible_symmetry(2, axes)
        kwargs = dict(source_kind="uniform", normalize_by_energy=normalize_by_energy)
        full = _place((0, 0, 0), **kwargs)
        reduced = _place(symmetry, **kwargs)
        expected = _kept_half(np.asarray(full._E), symmetry)
        error = np.abs(np.asarray(reduced._E) - expected).max() / np.abs(expected).max()
        assert error < 1e-6, f"reduced uniform injection differs by {error:.2e}"


class TestTransverseProfileOrientation:
    """The in-plane projection must not mirror or transpose the profile."""

    @pytest.mark.parametrize("propagation_axis", [0, 1, 2])
    def test_backward_direction_does_not_mirror_profile(self, propagation_axis):
        # A backward-propagating source carries the same transverse profile as a forward one; only
        # the field orientation and the time-of-flight offsets differ.
        forward = _place((0, 0, 0), propagation_axis=propagation_axis, direction="+", source_kind="ramp")
        backward = _place((0, 0, 0), propagation_axis=propagation_axis, direction="-", source_kind="ramp")

        def profile(source):
            field = np.asarray(source._E)
            component = int(np.argmax(np.abs(field).sum(axis=(1, 2, 3))))
            values = source._grid_to_hv(jnp.asarray(field[component]))
            return np.abs(np.asarray(values))

        assert np.allclose(profile(forward), profile(backward), rtol=1e-6)

    @pytest.mark.parametrize("propagation_axis", [0, 1, 2])
    def test_non_square_plane_keeps_axis_order(self, propagation_axis):
        # Propagation along y has descending oriented transverse axes; a profile built in
        # (horizontal, vertical) order has to be transposed before it is placed on the grid.
        source = _place(
            (0, 0, 0),
            propagation_axis=propagation_axis,
            transverse_cells=(16, 8),
            source_kind="ramp",
        )
        field = np.asarray(source._E)
        assert field.shape == (3, *source.grid_shape)
        component = int(np.argmax(np.abs(field).sum(axis=(1, 2, 3))))
        profile = np.abs(np.asarray(source._grid_to_hv(jnp.asarray(field[component]))))
        assert profile.shape == (16, 8)
        # The ramp increases by 1 per horizontal cell and by 100 per vertical cell.
        scale = profile[1, 0] - profile[0, 0]
        assert scale > 0
        assert np.isclose((profile[0, 1] - profile[0, 0]) / scale, 100.0, rtol=1e-3)


class TestUnsupportedSymmetryPlacements:
    """Configurations the reduction cannot represent must fail loudly, not silently."""

    @pytest.mark.parametrize("kwargs", [{"azimuth_angle": 10.0}, {"elevation_angle": 10.0}])
    def test_tilted_source_straddling_plane_raises(self, kwargs):
        objects, constraints, config = _build(_compatible_symmetry(2, "hv"), **kwargs)
        with pytest.raises(ValueError, match="symmetry"):
            fdtdx.place_objects(object_list=objects, config=config, constraints=constraints, key=jax.random.PRNGKey(0))

    def test_random_offset_source_straddling_plane_raises(self):
        objects, constraints, config = _build(_compatible_symmetry(2, "hv"), max_horizontal_offset=2e-7)
        with pytest.raises(ValueError, match="symmetry"):
            fdtdx.place_objects(object_list=objects, config=config, constraints=constraints, key=jax.random.PRNGKey(0))


class TestPolarizationWallCompatibility:
    """The wall type has to match the polarization; a mismatch is silent otherwise."""

    @staticmethod
    def _warnings(symmetry):
        """Collect fdtdx's loguru warnings emitted while placing a model."""
        messages: list[str] = []
        handler_id = logger.add(lambda message: messages.append(message), level="WARNING")
        try:
            _place(symmetry)
        finally:
            logger.remove(handler_id)
        return "".join(messages)

    def test_mismatched_polarization_warns(self):
        # E along the horizontal axis is *normal* to the horizontal plane, so it is even only under
        # PEC. Requesting PMC there makes it odd and the wall suppresses the injected field - the
        # single most common way to get a plausible-looking but wrong symmetric simulation.
        horizontal, _vertical = _axes(2)
        symmetry = [0, 0, 0]
        symmetry[horizontal] = 1  # PMC where PEC is required
        output = self._warnings((symmetry[0], symmetry[1], symmetry[2]))
        assert "makes Ex" in output and "odd" in output
        assert "use PEC (-1)" in output  # names the fix

    def test_matching_polarization_does_not_warn(self):
        assert "mirror-even profile" not in self._warnings(_compatible_symmetry(2, "hv"))
