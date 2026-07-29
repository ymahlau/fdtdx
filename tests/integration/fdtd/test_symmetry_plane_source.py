"""Regression tests for spatially varying plane sources under mirror symmetry (issue #425).

Three defects were fixed:

1. ``TFSFPlaneSource._get_center`` computed the transverse profile center from the
   *clipped* extent, so a Gaussian spot re-centered on the reduced quadrant instead of
   staying on the symmetry plane.
2. ``normalize_by_energy`` (and the Gaussian profile's own sum normalization) summed over
   the clipped plane, inflating the injected amplitude by ``2**(n_axes/2)``.
3. ``v_basis = cross(wave_vector, u_basis)`` mirrored the sampled profile about ``center``
   along the vertical axis for some axis/direction combinations. Predates symmetry; only
   observable for a profile that is not symmetric about its own center.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx

_L = 2e-6
_H = 3e-6
_SPACING = 200e-9
_WAVELENGTH = 1.5e-6


def _build(symmetry, *, direction="-", normalize=False, source_cls=None, **source_kwargs):
    config = fdtdx.SimulationConfig(
        time=20e-15,
        dtype=jnp.float32,
        courant_factor=0.99,
        grid=fdtdx.UniformGrid(spacing=_SPACING),
        symmetry=symmetry,
    )
    object_list, constraints = [], []
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(2 * _L, 2 * _L, _H),
        material=fdtdx.Material(permittivity=1.0, permeability=1.0),
    )
    object_list.append(volume)

    cls = source_cls or fdtdx.GaussianPlaneSource
    src = cls(
        name="Source",
        partial_grid_shape=(None, None, 1),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        direction=direction,
        normalize_by_energy=normalize,
        **source_kwargs,
    )
    object_list.append(src)
    own = 1 if direction == "-" else -1
    margin = -0.6e-6 if direction == "-" else 0.6e-6
    constraints.extend(
        [
            src.place_relative_to(
                volume, axes=(2,), own_positions=(own,), other_positions=(own,), margins=(margin,)
            ),
            src.place_at_center(volume, axes=(0, 1)),
            src.size_relative_to(volume, axes=(0, 1), proportions=(1.0, 1.0)),
        ]
    )

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="pml", thickness=5)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    object_list.extend(bound_dict.values())

    key = jax.random.PRNGKey(0)
    objs, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list, config=config, constraints=constraints, key=key
    )
    arrays, objs, _ = fdtdx.apply_params(arrays, objs, params, key=key)
    return objs, arrays, config


def _injected_profile(objs):
    """The transverse |E| profile the source injects, squeezed to 2D."""
    src = next(s for s in objs.sources if s.name == "Source")
    return np.abs(np.asarray(src._E)[0, :, :, 0]), src


class _OffCenterSource(fdtdx.UniformPlaneSource):
    """Single bright cell at a deliberately off-center transverse index."""

    def _get_amplitude_raw(self, center):
        del center
        idx = [0, 0, 0]
        idx[self.horizontal_axis] = 2
        idx[self.vertical_axis] = 3
        return jnp.zeros(self.grid_shape).at[tuple(idx)].set(1.0)


class TestProfileOrientation:
    """Fix 3: the sampled profile must not be mirrored about ``center``."""

    @pytest.mark.parametrize("direction", ["+", "-"])
    def test_off_center_profile_is_not_mirrored(self, direction):
        objs, _, _ = _build((0, 0, 0), direction=direction, source_cls=_OffCenterSource, amplitude=1.0)
        prof, _ = _injected_profile(objs)
        peak = np.unravel_index(np.argmax(prof), prof.shape)
        assert peak == (2, 3), f"profile mirrored for direction={direction!r}: peak at {peak}"
        assert np.count_nonzero(prof > 1e-12) == 1


class TestSymmetryProfileCentering:
    """Fixes 1 and 2: the reduced profile must equal the kept half of the full-domain profile."""

    @pytest.mark.parametrize("symmetry", [(-1, 0, 0), (0, 1, 0), (-1, 1, 0)])
    @pytest.mark.parametrize("normalize", [False, True])
    def test_reduced_profile_matches_full_half(self, symmetry, normalize):
        full_objs, _, _ = _build((0, 0, 0), normalize=normalize, radius=_L, std=1 / 3)
        red_objs, _, _ = _build(symmetry, normalize=normalize, radius=_L, std=1 / 3)
        full, _ = _injected_profile(full_objs)
        red, _ = _injected_profile(red_objs)

        half = full
        if symmetry[0] != 0:
            half = half[half.shape[0] // 2 :, :]
        if symmetry[1] != 0:
            half = half[:, half.shape[1] // 2 :]

        assert red.shape == half.shape
        np.testing.assert_allclose(red, half, rtol=0, atol=1e-5 * full.max())

    @pytest.mark.parametrize("symmetry", [(-1, 0, 0), (0, 1, 0), (-1, 1, 0)])
    def test_profile_peaks_on_the_symmetry_plane(self, symmetry):
        objs, _, _ = _build(symmetry, radius=_L, std=1 / 3)
        prof, _ = _injected_profile(objs)
        peak = np.unravel_index(np.argmax(prof), prof.shape)
        for a, arr_axis in ((0, 0), (1, 1)):
            if symmetry[a] != 0:
                assert peak[arr_axis] == 0, f"peak {peak} is off the {a}-symmetry plane"

    def test_clip_low_recorded(self):
        objs, _, _ = _build((-1, 1, 0), radius=_L, std=1 / 3)
        _, src = _injected_profile(objs)
        assert src._symmetry_clip_low == (10, 10, 0)
        assert src.unreduced_grid_shape == (20, 20, 1)

    def test_clip_low_zero_without_symmetry(self):
        objs, _, _ = _build((0, 0, 0), radius=_L, std=1 / 3)
        _, src = _injected_profile(objs)
        assert src._symmetry_clip_low == (0, 0, 0)
        assert src.unreduced_grid_shape == src.grid_shape
