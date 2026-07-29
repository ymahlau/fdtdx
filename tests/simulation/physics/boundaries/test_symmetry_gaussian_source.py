"""A symmetry-reduced Gaussian beam, unfolded, must reproduce the full-domain run (issue #425).

The existing symmetry simulation tests all use ``UniformPlaneSource`` with
``normalize_by_energy=False`` and ``direction="+"`` — a combination that cannot observe a
misplaced profile center, an inflated normalization, or a mirrored profile. This exercises
a spatially varying source instead.

A magnetic (PMC) symmetry plane deliberately gets no wall object: see
``test_reduced_simulation_matches_kept_half``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
import fdtdx.fdtd.symmetry

_L = 2e-6
_H = 3e-6
_SPACING = 200e-9
_WAVELENGTH = 1.5e-6
_TIME = 30e-15


def _run(symmetry, *, polarization, direction="-", normalize=True):
    config = fdtdx.SimulationConfig(
        time=_TIME,
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

    src = fdtdx.GaussianPlaneSource(
        name="Source",
        partial_grid_shape=(None, None, 1),
        fixed_E_polarization_vector=polarization,
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        direction=direction,
        radius=_L,
        std=1 / 3,
        normalize_by_energy=normalize,
    )
    object_list.append(src)
    own = 1 if direction == "-" else -1
    margin = -0.6e-6 if direction == "-" else 0.6e-6
    constraints.extend(
        [
            src.place_relative_to(volume, axes=(2,), own_positions=(own,), other_positions=(own,), margins=(margin,)),
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
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objs, config=config, key=key)
    return arrays


def _interior_error(reference, candidate):
    """Max |diff| over the PML-free interior, relative to the reference peak."""
    nx, ny, nz = reference.shape[1:]
    sl = (slice(8, nx - 8), slice(8, ny - 8), slice(6, nz - 6))
    a, b = reference[0][sl], candidate[0][sl]
    return float(np.abs(a - b).max() / np.abs(a).max())


@pytest.mark.parametrize("direction", ["-", "+"])
@pytest.mark.parametrize("normalize", [True, False])
def test_pec_symmetry_gaussian_matches_full_domain(direction, normalize):
    """An x-polarized Gaussian beam with a PEC x-symmetry plane, unfolded, matches full domain."""
    pol = (1, 0, 0)
    symmetry = (-1, 0, 0)
    full = np.asarray(_run((0, 0, 0), polarization=pol, direction=direction, normalize=normalize).fields.E)
    reduced = _run(symmetry, polarization=pol, direction=direction, normalize=normalize).fields.E

    # Half the domain in x, so the amplitude must not pick up a normalization factor.
    assert reduced.shape[1] == full.shape[1] // 2
    unfolded = np.asarray(fdtdx.unfold_fields(reduced, symmetry, "E"))
    assert unfolded.shape == full.shape
    assert _interior_error(full, unfolded) < 0.02


def test_pec_symmetry_preserves_amplitude_scale():
    """normalize_by_energy must not inflate the reduced-domain amplitude (fix 2)."""
    pol = (1, 0, 0)
    symmetry = (-1, 0, 0)
    full = np.asarray(_run((0, 0, 0), polarization=pol).fields.E)
    unfolded = np.asarray(fdtdx.unfold_fields(_run(symmetry, polarization=pol).fields.E, symmetry, "E"))
    nx, ny, nz = full.shape[1:]
    sl = (slice(8, nx - 8), slice(8, ny - 8), slice(6, nz - 6))
    a, b = full[0][sl], unfolded[0][sl]
    scale = float(np.sum(a * b) / np.sum(b * b))
    assert abs(scale - 1.0) < 0.05, f"amplitude off by {scale:.4f}x"


def _kept_half(field, symmetry):
    """The half of a full-domain array that the reduction keeps."""
    out = field
    for a in range(3):
        if symmetry[a] != 0:
            half = out.shape[1 + a] // 2
            out = np.take(out, np.arange(half, out.shape[1 + a]), axis=1 + a)
    return out


@pytest.mark.parametrize(
    "symmetry,polarization",
    [
        ((0, 1, 0), (1, 0, 0)),  # PMC on y  (x-polarized beam)
        ((1, 0, 0), (0, 1, 0)),  # PMC on x  (y-polarized beam)
        ((-1, 1, 0), (1, 0, 0)),  # PEC x + PMC y
        ((1, -1, 0), (0, 1, 0)),  # PMC x + PEC y
    ],
)
def test_reduced_simulation_matches_kept_half(symmetry, polarization):
    """The reduced run must reproduce the half of the full domain it represents.

    A PMC symmetry plane gets no explicit wall: the tangential-H node a magnetic mirror
    zeroes sits outside the reduced array, where zero-padding already supplies it. Inserting
    a PerfectMagneticConductor instead zeroed tangential H a full cell inside the domain,
    which showed up as ~30% error concentrated at the plane.
    """
    full = np.asarray(_run((0, 0, 0), polarization=polarization).fields.E)
    reduced = np.asarray(_run(symmetry, polarization=polarization).fields.E)
    kept = _kept_half(full, symmetry)
    assert reduced.shape == kept.shape
    err = np.abs(kept - reduced).max() / np.abs(full).max()
    assert err < 0.02, f"reduced run deviates from the kept half by {err:.4f}"


def test_no_pmc_wall_is_created_for_a_magnetic_symmetry_plane():
    from fdtdx.objects.boundaries.pmc import PerfectMagneticConductor

    config = fdtdx.SimulationConfig(
        time=_TIME,
        dtype=jnp.float32,
        courant_factor=0.99,
        grid=fdtdx.UniformGrid(spacing=_SPACING),
        symmetry=(-1, 1, 0),
    )
    walls = fdtdx.fdtd.symmetry.make_symmetry_walls(
        config=config,
        reduced_volume_shape=(8, 8, 8),
        key=jax.random.PRNGKey(0),
        existing_names=set(),
    )
    assert not any(isinstance(w, PerfectMagneticConductor) for w in walls)
    assert len(walls) == 1  # only the PEC x plane
