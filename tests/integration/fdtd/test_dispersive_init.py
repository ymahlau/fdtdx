"""Integration tests for dispersive-material array allocation and filling.

Covers the dispersive branches of ``_init_arrays`` (UniformMaterialObject,
StaticMultiMaterialObject) and ``apply_params`` (Device CONTINUOUS/DISCRETE).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.dispersion import (
    CCPRPole,
    DispersionModel,
    DrudePole,
    LorentzPole,
    compute_pole_delta_coefficients_per_axis,
    compute_pole_delta_coefficients_tensor,
)
from fdtdx.fdtd.initialization import apply_params, place_objects
from fdtdx.materials import Material
from fdtdx.objects.device.device import Device
from fdtdx.objects.device.parameters.discretization import ClosestIndex
from fdtdx.objects.object import GridCoordinateConstraint
from fdtdx.objects.static_material.sphere import Sphere
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject


def _delta_coeffs(poles, dt):
    """Delta-basis coefficients ``(a1, a0, b1, c4, b0)``, one scalar per pole.

    The stored arrays hold the delta basis (see :mod:`fdtdx.dispersion`), so
    reference values must come from there rather than from the P-form c1/c2.
    Valid for isotropic poles, where all three axis columns coincide.
    """
    return tuple(v[:, 0] for v in compute_pole_delta_coefficients_per_axis(poles, dt))


def _placed(container, name):
    """Return the placed copy of an object from an ObjectContainer by name."""
    for o in container.objects:
        if o.name == name:
            return o
    raise KeyError(name)


@pytest.fixture
def simple_config():
    from fdtdx.core.grid import UniformGrid

    return SimulationConfig(grid=UniformGrid(spacing=1e-7), time=1e-14, backend="cpu")


@pytest.fixture
def simple_volume():
    return SimulationVolume(name="volume", partial_grid_shape=(30, 30, 30))


def _lorentz_material(eps_inf=2.0):
    return Material(
        permittivity=eps_inf,
        dispersion=DispersionModel(poles=(LorentzPole(resonance_frequency=2e15, damping=1e13, delta_epsilon=1.5),)),
    )


def _drude_material(eps_inf=1.0):
    return Material(
        permittivity=eps_inf,
        dispersion=DispersionModel(poles=(DrudePole(plasma_frequency=1.37e16, damping=1e14),)),
    )


def _three_pole_material(eps_inf=2.0):
    return Material(
        permittivity=eps_inf,
        dispersion=DispersionModel(
            poles=(
                LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=1.0),
                LorentzPole(resonance_frequency=2e15, damping=2e13, delta_epsilon=0.5),
                DrudePole(plasma_frequency=5e15, damping=5e13),
            )
        ),
    )


# ---------------------------------------------------------------------------
# UniformMaterialObject tests
# ---------------------------------------------------------------------------


def test_dispersive_arrays_allocated(simple_config, simple_volume):
    """A dispersive UniformMaterialObject triggers allocation of polarization and
    coefficient arrays with the expected shapes, and coefficient values match the
    closed-form recurrence inside the object slice."""
    material = _lorentz_material(eps_inf=2.0)
    obj = UniformMaterialObject(name="slab", partial_grid_shape=(10, 10, 10), material=material)
    constraint = GridCoordinateConstraint(
        object="slab", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    objects, arrays, _, config, _ = place_objects([simple_volume, obj], simple_config, [constraint], key)
    placed = _placed(objects, "slab")

    Nx, Ny, Nz = simple_volume.partial_grid_shape  # type: ignore[misc]
    assert arrays.fields.dispersive_x1 is not None
    assert arrays.fields.dispersive_y2 is not None
    assert arrays.dispersive_a1 is not None
    assert arrays.dispersive_a0 is not None
    assert arrays.dispersive_b1 is not None
    assert arrays.fields.dispersive_x1.shape == (1, 3, Nx, Ny, Nz)
    assert arrays.fields.dispersive_y2.shape == (1, 3, Nx, Ny, Nz)
    assert arrays.dispersive_a1.shape == (1, 1, Nx, Ny, Nz)
    assert arrays.dispersive_a0.shape == (1, 1, Nx, Ny, Nz)
    assert arrays.dispersive_b1.shape == (1, 1, Nx, Ny, Nz)
    # polarization always starts at zero
    assert jnp.all(arrays.fields.dispersive_x1 == 0)
    assert jnp.all(arrays.fields.dispersive_y2 == 0)

    # Coefficient values inside the slab should match compute_pole_coefficients.
    a1_ref, a0_ref, b1_ref, _c4_ref, _ = _delta_coeffs(
        material.dispersion.poles,
        config.time_step_duration,  # type: ignore[union-attr]
    )
    xs, ys, zs = placed.grid_slice
    assert jnp.allclose(arrays.dispersive_a1[0, 0, xs, ys, zs], a1_ref[0])
    assert jnp.allclose(arrays.dispersive_a0[0, 0, xs, ys, zs], a0_ref[0])
    assert jnp.allclose(arrays.dispersive_b1[0, 0, xs, ys, zs], b1_ref[0])

    # c3 should be zero outside the slab (vacuum cells have no polarization)
    inside_mask = jnp.zeros((Nx, Ny, Nz), dtype=bool).at[xs, ys, zs].set(True)
    assert jnp.all(arrays.dispersive_b1[0, 0][~inside_mask] == 0.0)


def test_no_dispersive_arrays_when_unused(simple_config, simple_volume):
    """A non-dispersive simulation leaves all dispersive arrays as None."""
    material = Material(permittivity=2.25)
    obj = UniformMaterialObject(name="slab", partial_grid_shape=(10, 10, 10), material=material)
    constraint = GridCoordinateConstraint(
        object="slab", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    _, arrays, _, _, _ = place_objects([simple_volume, obj], simple_config, [constraint], key)

    assert arrays.fields.dispersive_x1 is None
    assert arrays.fields.dispersive_y2 is None
    assert arrays.dispersive_a1 is None
    assert arrays.dispersive_a0 is None
    assert arrays.dispersive_b1 is None


def test_pole_padding_mixed_pole_counts(simple_config, simple_volume):
    """A simulation mixing a 1-pole and a 3-pole material allocates num_poles=3
    and zero-pads the 1-pole material in the unused slots."""
    one_pole = _lorentz_material(eps_inf=2.0)
    three_pole = _three_pole_material(eps_inf=2.0)
    obj1 = UniformMaterialObject(name="one_pole_slab", partial_grid_shape=(6, 6, 6), material=one_pole)
    obj2 = UniformMaterialObject(name="three_pole_slab", partial_grid_shape=(6, 6, 6), material=three_pole)
    constraints = [
        GridCoordinateConstraint(object="one_pole_slab", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[2, 2, 2]),
        GridCoordinateConstraint(
            object="three_pole_slab", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[15, 15, 15]
        ),
    ]
    key = jax.random.PRNGKey(0)
    objects, arrays, _, _, _ = place_objects([simple_volume, obj1, obj2], simple_config, constraints, key)
    placed1 = _placed(objects, "one_pole_slab")
    placed2 = _placed(objects, "three_pole_slab")

    Nx, Ny, Nz = simple_volume.partial_grid_shape  # type: ignore[misc]
    assert arrays.dispersive_a1.shape == (3, 1, Nx, Ny, Nz)
    assert arrays.fields.dispersive_x1.shape == (3, 3, Nx, Ny, Nz)

    # Inside the 1-pole slab: pole slot 0 is populated, slots 1 and 2 are zero.
    xs1, ys1, zs1 = placed1.grid_slice
    c1_1p_slot0 = arrays.dispersive_a1[0, 0, xs1, ys1, zs1]
    c1_1p_slot1 = arrays.dispersive_a1[1, 0, xs1, ys1, zs1]
    c1_1p_slot2 = arrays.dispersive_a1[2, 0, xs1, ys1, zs1]
    c3_1p_slot0 = arrays.dispersive_b1[0, 0, xs1, ys1, zs1]
    c3_1p_slot1 = arrays.dispersive_b1[1, 0, xs1, ys1, zs1]
    c3_1p_slot2 = arrays.dispersive_b1[2, 0, xs1, ys1, zs1]
    assert jnp.all(c1_1p_slot0 != 0.0)
    assert jnp.all(c1_1p_slot1 == 0.0)
    assert jnp.all(c1_1p_slot2 == 0.0)
    assert jnp.all(c3_1p_slot0 > 0.0)
    assert jnp.all(c3_1p_slot1 == 0.0)
    assert jnp.all(c3_1p_slot2 == 0.0)

    # Inside the 3-pole slab: all three slots populated.
    xs2, ys2, zs2 = placed2.grid_slice
    for p in range(3):
        c3_slot = arrays.dispersive_b1[p, 0, xs2, ys2, zs2]
        assert jnp.all(c3_slot > 0.0)


def test_non_dispersive_overlap_clears_dispersive_coefficients(simple_config, simple_volume):
    """A non-dispersive ``UniformMaterialObject`` placed *on top of* a
    dispersive one must zero out the pole coefficients in the overlap.

    Motivation: object placement uses last-write-wins for material properties,
    so a plain slab stacked over a dispersive slab overwrites ε/μ in the
    overlap. The dispersive coefficients must follow the same rule — otherwise
    the overlap cells would have plain ε but still drive the ADE recurrence
    with stale pole coefficients, yielding unphysical dynamics.
    """
    disp_mat = _lorentz_material(eps_inf=2.0)
    plain_mat = Material(permittivity=4.0)
    # disp occupies [5..15] on each axis; plain occupies [10..20]; overlap: [10..15]
    disp = UniformMaterialObject(name="disp", partial_grid_shape=(10, 10, 10), material=disp_mat)
    plain = UniformMaterialObject(name="plain", partial_grid_shape=(10, 10, 10), material=plain_mat)
    constraints = [
        GridCoordinateConstraint(object="disp", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[5, 5, 5]),
        GridCoordinateConstraint(object="plain", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]),
    ]
    key = jax.random.PRNGKey(0)
    # Object order matters: plain is placed second, so it overwrites in the overlap.
    objects, arrays, _, _, _ = place_objects([simple_volume, disp, plain], simple_config, constraints, key)
    placed_disp = _placed(objects, "disp")
    placed_plain = _placed(objects, "plain")

    assert arrays.dispersive_a1 is not None
    assert arrays.dispersive_b1 is not None

    Nx, Ny, Nz = simple_volume.partial_grid_shape  # type: ignore[misc]
    disp_mask = jnp.zeros((Nx, Ny, Nz), dtype=bool).at[placed_disp.grid_slice].set(True)
    plain_mask = jnp.zeros((Nx, Ny, Nz), dtype=bool).at[placed_plain.grid_slice].set(True)
    overlap_mask = disp_mask & plain_mask
    disp_only_mask = disp_mask & ~plain_mask

    # Sanity: the scene actually has an overlap region and a disp-only region.
    assert jnp.any(overlap_mask), "test setup: expected a disp/plain overlap region"
    assert jnp.any(disp_only_mask), "test setup: expected a disp-only region"

    # Overlap: coefficients must be zero (plain slab cleared them).
    assert jnp.all(arrays.dispersive_a1[0, 0][overlap_mask] == 0.0), (
        "Non-dispersive overlap failed to clear c1 — stale pole coefficients remain"
    )
    assert jnp.all(arrays.dispersive_b1[0, 0][overlap_mask] == 0.0), (
        "Non-dispersive overlap failed to clear c3 — ADE recurrence would fire on plain cells"
    )

    # Disp-only region: coefficients must still be populated.
    assert jnp.all(arrays.dispersive_b1[0, 0][disp_only_mask] > 0.0)


def test_dispersive_with_non_dispersive_object(simple_config, simple_volume):
    """A dispersive slab plus a non-dispersive slab: dispersive cells have
    populated coefficients while non-dispersive cells see all zeros."""
    disp_mat = _lorentz_material(eps_inf=2.0)
    plain_mat = Material(permittivity=4.0)
    disp = UniformMaterialObject(name="disp", partial_grid_shape=(6, 6, 6), material=disp_mat)
    plain = UniformMaterialObject(name="plain", partial_grid_shape=(6, 6, 6), material=plain_mat)
    constraints = [
        GridCoordinateConstraint(object="disp", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[2, 2, 2]),
        GridCoordinateConstraint(object="plain", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[15, 15, 15]),
    ]
    key = jax.random.PRNGKey(0)
    objects, arrays, _, _, _ = place_objects([simple_volume, disp, plain], simple_config, constraints, key)
    placed_disp = _placed(objects, "disp")
    placed_plain = _placed(objects, "plain")

    assert arrays.dispersive_a1 is not None
    xs_d, ys_d, zs_d = placed_disp.grid_slice
    xs_p, ys_p, zs_p = placed_plain.grid_slice
    # Dispersive slab: c3 non-zero
    assert jnp.all(arrays.dispersive_b1[0, 0, xs_d, ys_d, zs_d] > 0.0)
    # Non-dispersive slab: c3 zero
    assert jnp.all(arrays.dispersive_b1[0, 0, xs_p, ys_p, zs_p] == 0.0)


# ---------------------------------------------------------------------------
# StaticMultiMaterialObject tests (Sphere)
# ---------------------------------------------------------------------------


def test_static_multi_material_dispersive(simple_config, simple_volume):
    """A Sphere containing a dispersive material has non-zero coefficients
    strictly inside the voxel mask and zeros everywhere else."""
    materials = {
        "background": Material(permittivity=1.0),
        "drude": _drude_material(eps_inf=1.0),
    }
    sphere = Sphere(
        name="sphere",
        materials=materials,
        material_name="drude",
        radius=5.0 * simple_config.uniform_spacing(),
    )
    constraint = GridCoordinateConstraint(object="sphere", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[8, 8, 8])
    key = jax.random.PRNGKey(0)
    objects, arrays, _, _, _ = place_objects([simple_volume, sphere], simple_config, [constraint], key)
    placed = _placed(objects, "sphere")

    assert arrays.dispersive_b1 is not None
    assert arrays.dispersive_b1.shape[0] == 1  # one pole total (Drude)

    # Outside the bounding box → strictly zero
    Nx, Ny, Nz = simple_volume.partial_grid_shape  # type: ignore[misc]
    xs, ys, zs = placed.grid_slice
    inside_mask = jnp.zeros((Nx, Ny, Nz), dtype=bool).at[xs, ys, zs].set(True)
    assert jnp.all(arrays.dispersive_b1[0, 0][~inside_mask] == 0.0)

    # Inside the voxel mask → non-zero
    voxel_mask = placed.get_voxel_mask_for_shape().astype(bool)
    inside_slab = arrays.dispersive_b1[0, 0, xs, ys, zs]
    assert jnp.any(inside_slab[voxel_mask] > 0.0)
    # Cells inside the bounding box but outside the sphere remain zero
    assert jnp.all(inside_slab[~voxel_mask] == 0.0)


# ---------------------------------------------------------------------------
# Device tests (apply_params)
# ---------------------------------------------------------------------------


def test_device_dispersive_continuous(simple_config, simple_volume):
    """Device with CONTINUOUS output writes interpolated coefficients into the
    dispersive arrays inside the device slice."""
    materials = {
        "air": Material(permittivity=1.0),
        "drude": _drude_material(eps_inf=1.0),
    }
    device = Device(
        name="device",
        partial_grid_shape=(10, 10, 10),
        partial_voxel_grid_shape=(5, 5, 5),
        materials=materials,
        param_transforms=[],  # empty → CONTINUOUS, needs exactly 2 materials
    )
    constraint = GridCoordinateConstraint(
        object="device", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = place_objects([simple_volume, device], simple_config, [constraint], key)
    placed_device = _placed(objects, "device")
    xs, ys, zs = placed_device.grid_slice

    # Force all-drude: params=1 -> cur_material_indices=1 everywhere in the device
    drude_params = {name: jnp.ones_like(p) for name, p in params.items()}
    arrays, objects, _ = apply_params(arrays, objects, drude_params, key)

    assert arrays.dispersive_a1 is not None
    assert arrays.dispersive_a1.shape == (1, 1, 30, 30, 30)

    # Inside the device: coefficients should equal the drude coefficients.
    a1_ref, a0_ref, b1_ref, _c4_ref, _ = _delta_coeffs(
        materials["drude"].dispersion.poles,  # type: ignore[union-attr]
        config.time_step_duration,
    )
    assert jnp.allclose(arrays.dispersive_a1[0, 0, xs, ys, zs], a1_ref[0])
    assert jnp.allclose(arrays.dispersive_a0[0, 0, xs, ys, zs], a0_ref[0])
    assert jnp.allclose(arrays.dispersive_b1[0, 0, xs, ys, zs], b1_ref[0])

    # All-air: params=0 -> coefficients should be zero (air has no dispersion)
    air_params = {name: jnp.zeros_like(p) for name, p in params.items()}
    arrays2, _, _ = apply_params(arrays, objects, air_params, key)
    assert jnp.all(arrays2.dispersive_a1[0, 0, xs, ys, zs] == 0.0)
    assert jnp.all(arrays2.dispersive_b1[0, 0, xs, ys, zs] == 0.0)

    # Half interpolation: params=0.5 -> coefficients should be half the drude values
    half_params = {name: 0.5 * jnp.ones_like(p) for name, p in params.items()}
    arrays3, _, _ = apply_params(arrays, objects, half_params, key)
    assert jnp.allclose(arrays3.dispersive_a1[0, 0, xs, ys, zs], 0.5 * a1_ref[0])
    assert jnp.allclose(arrays3.dispersive_b1[0, 0, xs, ys, zs], 0.5 * b1_ref[0])


def test_device_dispersive_discrete(simple_config, simple_volume):
    """Device with DISCRETE output (ClosestIndex) picks per-voxel coefficients
    from the material table."""
    materials = {
        "air": Material(permittivity=1.0),
        "drude": _drude_material(eps_inf=1.0),
    }
    device = Device(
        name="device",
        partial_grid_shape=(10, 10, 10),
        partial_voxel_grid_shape=(5, 5, 5),
        materials=materials,
        param_transforms=[ClosestIndex()],
    )
    constraint = GridCoordinateConstraint(
        object="device", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = place_objects([simple_volume, device], simple_config, [constraint], key)
    placed_device = _placed(objects, "device")
    xs, ys, zs = placed_device.grid_slice

    # All drude
    drude_params = {name: jnp.ones_like(p) for name, p in params.items()}
    arrays, objects, _ = apply_params(arrays, objects, drude_params, key)

    a1_ref, _, b1_ref, _, _ = _delta_coeffs(
        materials["drude"].dispersion.poles,  # type: ignore[union-attr]
        config.time_step_duration,
    )
    assert jnp.allclose(arrays.dispersive_a1[0, 0, xs, ys, zs], a1_ref[0])
    assert jnp.allclose(arrays.dispersive_b1[0, 0, xs, ys, zs], b1_ref[0])

    # All air: coefficients should be zero
    air_params = {name: jnp.zeros_like(p) for name, p in params.items()}
    arrays2, _, _ = apply_params(arrays, objects, air_params, key)
    assert jnp.all(arrays2.dispersive_b1[0, 0, xs, ys, zs] == 0.0)


def _aniso_plus_dispersive_scene(simple_volume):
    """Off-diagonal permittivity tensor slab next to a dispersive slab."""
    aniso = Material(
        permittivity=(2.0, 0.1, 0.0, 0.1, 2.5, 0.0, 0.0, 0.0, 3.0),  # off-diagonal
    )
    disp = _lorentz_material(eps_inf=2.0)
    obj1 = UniformMaterialObject(name="aniso", partial_grid_shape=(6, 6, 6), material=aniso)
    obj2 = UniformMaterialObject(name="disp", partial_grid_shape=(6, 6, 6), material=disp)
    constraints = [
        GridCoordinateConstraint(object="aniso", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[2, 2, 2]),
        GridCoordinateConstraint(object="disp", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[15, 15, 15]),
    ]
    return [simple_volume, obj1, obj2], constraints


def test_fully_anisotropic_plus_dispersive_allocates(simple_config, simple_volume):
    """Dispersion combined with an off-diagonal permittivity tensor runs through
    the fully anisotropic update path: allocation succeeds with 9-component
    permittivity while the (axis-aligned) dispersion keeps its natural tiers."""
    objects_list, constraints = _aniso_plus_dispersive_scene(simple_volume)
    key = jax.random.PRNGKey(0)
    _, arrays, _, _, _ = place_objects(objects_list, simple_config, constraints, key)
    assert arrays.inv_permittivities.shape[0] == 9
    assert arrays.dispersive_a1 is not None
    assert arrays.dispersive_a1.shape[1] == 1
    assert arrays.dispersive_b1.shape[1] == 1


def test_isotropic_dispersive_reversible_raises(simple_config, simple_volume):
    """Any dispersive material rejects the 'reversible' gradient method at
    initialization — reversing the ADE recurrence is not supported."""
    from fdtdx.config import GradientConfig
    from fdtdx.interfaces.recorder import Recorder

    obj = UniformMaterialObject(name="slab", partial_grid_shape=(10, 10, 10), material=_lorentz_material())
    constraint = GridCoordinateConstraint(
        object="slab", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    config = simple_config.aset("gradient_config", GradientConfig(method="reversible", recorder=Recorder(modules=[])))
    key = jax.random.PRNGKey(0)
    with pytest.raises(NotImplementedError, match="under active development"):
        place_objects([simple_volume, obj], config, [constraint], key)


def test_fully_anisotropic_plus_dispersive_reversible_raises(simple_config, simple_volume):
    """Same rejection on the fully anisotropic ADE path."""
    from fdtdx.config import GradientConfig
    from fdtdx.interfaces.recorder import Recorder

    objects_list, constraints = _aniso_plus_dispersive_scene(simple_volume)
    config = simple_config.aset("gradient_config", GradientConfig(method="reversible", recorder=Recorder(modules=[])))
    key = jax.random.PRNGKey(0)
    with pytest.raises(NotImplementedError, match="under active development"):
        place_objects(objects_list, config, constraints, key)


def _unstable_ccpr_material(eps_inf=2.0, integrator="central"):
    """CCPR material whose implicit-update divisor goes non-positive at the
    ``simple_config`` time step (~1.9e-16 s), i.e. large negative Re(residue).

    Pinned to ``integrator="central"``: its forward-difference dE/dt term puts the
    full ``b*dt`` on ``E^{n+1}``, which is what drives the divisor negative. The
    default ``"centered_edot"`` halves that and keeps the same pole runnable, so
    the rejection path can only be exercised with the legacy scheme.
    """
    pole = CCPRPole(pole=complex(-1e13, -2e15), residue=complex(-6e15, 1e15), integrator=integrator)
    return Material(permittivity=eps_inf, dispersion=DispersionModel(poles=(pole,)))


def test_centered_edot_places_the_material_central_rejects(simple_config, simple_volume):
    """The same pole that ``"central"`` cannot place is accepted under the default
    scheme — the resolution ceiling the forward difference imposes is lifted."""
    mat = _unstable_ccpr_material(integrator="centered_edot")
    obj = UniformMaterialObject(name="gold", partial_grid_shape=(10, 10, 10), material=mat)
    constraint = GridCoordinateConstraint(
        object="gold", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    _, arrays, _, _, _ = place_objects([simple_volume, obj], simple_config, [constraint], key)
    assert arrays.dispersive_c4 is not None
    assert arrays.dispersive_b0 is not None


def test_unstable_ccpr_material_raises_at_placement(simple_config, simple_volume):
    """A CCPR material with a non-positive implicit divisor must be rejected by
    place_objects (via _init_arrays -> validate_dispersive_divisor_stability)."""
    obj = UniformMaterialObject(name="gold", partial_grid_shape=(10, 10, 10), material=_unstable_ccpr_material())
    constraint = GridCoordinateConstraint(
        object="gold", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="gold"):
        place_objects([simple_volume, obj], simple_config, [constraint], key)


def test_lowering_courant_factor_stabilizes_ccpr(simple_config, simple_volume):
    """The remediation actually works: the same material places cleanly once the
    courant_factor is lowered below the value reported in the error message."""
    obj = UniformMaterialObject(name="gold", partial_grid_shape=(10, 10, 10), material=_unstable_ccpr_material())
    constraint = GridCoordinateConstraint(
        object="gold", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError) as exc:
        place_objects([simple_volume, obj], simple_config, [constraint], key)
    cf_max = float(str(exc.value).split("lower courant_factor to <= ")[1].split(" ")[0])
    safe_config = simple_config.aset("courant_factor", 0.9 * cf_max)
    # Must not raise now.
    place_objects([simple_volume, obj], safe_config, [constraint], key)


def test_stable_ccpr_material_places_cleanly(simple_config, simple_volume):
    """A CCPR material with a comfortably positive divisor places without error."""
    pole = CCPRPole(pole=complex(-1e13, -2e15), residue=complex(-2e15, 1e15))
    mat = Material(permittivity=2.0, dispersion=DispersionModel(poles=(pole,)))
    obj = UniformMaterialObject(name="metal", partial_grid_shape=(10, 10, 10), material=mat)
    constraint = GridCoordinateConstraint(
        object="metal", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    _, arrays, _, _, _ = place_objects([simple_volume, obj], simple_config, [constraint], key)
    assert arrays.dispersive_c4 is not None


def test_lorentz_material_unaffected_by_ccpr_validation(simple_config, simple_volume):
    """A Lorentz-only sim (c4 = 0) is not subject to the divisor validation and
    places cleanly even at the default courant_factor."""
    obj = UniformMaterialObject(name="slab", partial_grid_shape=(10, 10, 10), material=_lorentz_material(eps_inf=2.0))
    constraint = GridCoordinateConstraint(
        object="slab", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    place_objects([simple_volume, obj], simple_config, [constraint], key)


def test_non_dispersive_unused_import_guard():
    """Guard against accidental regression: importing the dispersion module
    should not break anything for non-dispersive materials."""
    mat = Material(permittivity=2.0)
    assert mat.is_dispersive is False
    # numpy is imported above; touch it so the linter doesn't complain about unused imports
    assert np.asarray(0.0).item() == 0.0


# ---------------------------------------------------------------------------
# Source-path dispersion plumbing
# ---------------------------------------------------------------------------


def test_device_continuous_half_interpolation(simple_config, simple_volume):
    """Device CONTINUOUS output with ``cur_material_indices == 0.5`` should
    linearly blend the two material coefficient arrays — the current STE-
    consistent convention. This regression-tests the interpolation branch
    explicitly at a non-trivial midpoint value."""
    materials = {
        "air": Material(permittivity=1.0),
        "drude": _drude_material(eps_inf=1.0),
    }
    device = Device(
        name="device",
        partial_grid_shape=(10, 10, 10),
        partial_voxel_grid_shape=(5, 5, 5),
        materials=materials,
        param_transforms=[],
    )
    constraint = GridCoordinateConstraint(
        object="device", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = place_objects([simple_volume, device], simple_config, [constraint], key)
    placed_device = _placed(objects, "device")
    xs, ys, zs = placed_device.grid_slice

    half_params = {name: 0.5 * jnp.ones_like(p) for name, p in params.items()}
    arrays_half, _, _ = apply_params(arrays, objects, half_params, key)

    a1_ref, _, b1_ref, _, _ = _delta_coeffs(
        materials["drude"].dispersion.poles,  # type: ignore[union-attr]
        config.time_step_duration,
    )
    # 0.5 * (air_coeffs = 0) + 0.5 * (drude_coeffs)
    assert jnp.allclose(arrays_half.dispersive_a1[0, 0, xs, ys, zs], 0.5 * a1_ref[0])
    assert jnp.allclose(arrays_half.dispersive_b1[0, 0, xs, ys, zs], 0.5 * b1_ref[0])
    # All coefficients outside the device slice remain zero
    mask = jnp.zeros(simple_volume.partial_grid_shape, dtype=bool).at[xs, ys, zs].set(True)  # type: ignore[misc]
    assert jnp.all(arrays_half.dispersive_a1[0, 0][~mask] == 0.0)


def test_dipole_source_samples_frequency_corrected_permittivity(simple_config, simple_volume):
    """A ``PointDipoleSource`` embedded in a dispersive medium should sample
    the real part of ``eps(omega)`` at its cell during ``apply_params``,
    not the high-frequency ``eps_inf``. This verifies the core of the
    dipole fix without depending on simulation-level physics."""
    from fdtdx.core.wavelength import WaveCharacter
    from fdtdx.objects.sources.dipole import PointDipoleSource

    # Lorentz pole such that at omega = resonance/2, eps_eff = eps_inf + 3
    eps_inf = 1.0
    resonance = 4.0e15  # rad/s
    target_omega = resonance / 2.0
    model = DispersionModel(poles=(LorentzPole(resonance_frequency=resonance, damping=1e11, delta_epsilon=2.25),))
    eps_full = eps_inf + complex(model.susceptibility(target_omega))
    # sanity — the pole must produce a meaningful correction
    assert eps_full.real > 3.5, f"premise broken: eps_full.real = {eps_full.real}"

    material = Material(permittivity=eps_inf, dispersion=model)
    slab = UniformMaterialObject(name="bg", partial_grid_shape=(30, 30, 30), material=material)
    dipole = PointDipoleSource(
        name="dip",
        partial_grid_shape=(1, 1, 1),
        wave_character=WaveCharacter(frequency=target_omega / (2.0 * np.pi)),
        polarization=0,
    )
    constraints = [
        GridCoordinateConstraint(object="bg", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[0, 0, 0]),
        GridCoordinateConstraint(object="dip", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[15, 15, 15]),
    ]
    key = jax.random.PRNGKey(0)
    objects, arrays, params, _, _ = place_objects([simple_volume, slab, dipole], simple_config, constraints, key)
    arrays, objects, _ = apply_params(arrays, objects, params, key)
    placed_dip = _placed(objects, "dip")

    # apply() should have stashed the frequency-corrected inv_eps at the source cell
    inv_eps_local = placed_dip._inv_eps_local
    # shape is (num_components, 1, 1, 1) for an isotropic, 1-cell source
    expected_inv_eps = 1.0 / eps_full.real
    assert jnp.allclose(inv_eps_local, expected_inv_eps, atol=5e-3), (
        f"dipole _inv_eps_local={np.array(inv_eps_local).ravel()}, expected ≈ {expected_inv_eps:.4f} (1/Re(eps))"
    )


def test_plane_source_apply_changes_with_dispersion(simple_config, simple_volume):
    """A ``UniformPlaneSource`` placed inside a dispersive slab should see a
    different ``_H`` buffer in ``apply()`` than the same source in a vacuum
    scene. This proves ``apply_params`` actually propagates the dispersive
    coefficient arrays into the source's ``apply()`` call."""
    from fdtdx.core.wavelength import WaveCharacter
    from fdtdx.objects.sources.linear_polarization import UniformPlaneSource

    eps_inf = 1.0
    wavelength = 0.8e-6
    omega = 2.0 * np.pi * c0 / wavelength
    # Pick a resonance above omega for a low-loss, high-eps medium
    model = DispersionModel(poles=(LorentzPole(resonance_frequency=4.0e15, damping=1e11, delta_epsilon=3.0),))

    def _build(material):
        plain_slab = UniformMaterialObject(name="bg", partial_grid_shape=(30, 30, 30), material=material)
        source = UniformPlaneSource(
            name="src",
            partial_grid_shape=(None, None, 1),
            wave_character=WaveCharacter(frequency=omega / (2.0 * np.pi)),
            direction="+",
            fixed_E_polarization_vector=(1, 0, 0),
        )
        constraints = [
            GridCoordinateConstraint(object="bg", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[0, 0, 0]),
            source.same_size(simple_volume, axes=(0, 1)),
            source.place_at_center(simple_volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(15,)),
        ]
        key = jax.random.PRNGKey(0)
        objects, arrays, params, _, _ = place_objects(
            [simple_volume, plain_slab, source], simple_config, constraints, key
        )
        arrays, objects, _ = apply_params(arrays, objects, params, key)
        return _placed(objects, "src")

    src_vacuum = _build(Material(permittivity=eps_inf))
    src_disp = _build(Material(permittivity=eps_inf, dispersion=model))

    H_vac = np.asarray(src_vacuum._H)
    H_disp = np.asarray(src_disp._H)
    # The H buffer is the polarization vector scaled by 1/impedance, and
    # impedance in the dispersive case is sqrt(1/Re(eps_eff)) != vacuum.
    # A non-zero fractional change proves the dispersive coefficients
    # actually reached apply().
    diff = np.max(np.abs(H_disp - H_vac)) / (np.max(np.abs(H_vac)) + 1e-30)
    assert diff > 0.1, (
        f"|H_disp - H_vac| / |H_vac| = {diff:.3f} — source apply() did not see the dispersive coefficients"
    )


# ---------------------------------------------------------------------------
# Per-axis (diagonally anisotropic) dispersion
# ---------------------------------------------------------------------------


def _per_axis_lorentz_material(eps_inf=(2.0, 2.0, 3.0)):
    from fdtdx.dispersion import compute_pole_coefficients_per_axis  # noqa: F401  (re-exported for tests below)

    return Material(
        permittivity=eps_inf,
        dispersion=DispersionModel(
            poles=(
                LorentzPole(
                    resonance_frequency=(2e15, 2e15, 1e15),
                    damping=(1e13, 1e13, 2e13),
                    delta_epsilon=(1.5, 1.5, 0.5),
                ),
            )
        ),
    )


def test_per_axis_dispersion_allocates_3_component_coefficients(simple_config, simple_volume):
    """A per-axis dispersive material widens the coefficient arrays to a
    3-entry component axis and bakes per-axis values inside the object slice."""

    material = _per_axis_lorentz_material()
    obj = UniformMaterialObject(name="slab", partial_grid_shape=(10, 10, 10), material=material)
    constraint = GridCoordinateConstraint(
        object="slab", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    objects, arrays, _, config, _ = place_objects([simple_volume, obj], simple_config, [constraint], key)
    placed = _placed(objects, "slab")

    Nx, Ny, Nz = simple_volume.partial_grid_shape  # type: ignore[misc]
    assert arrays.dispersive_a1 is not None
    assert arrays.dispersive_a1.shape == (1, 3, Nx, Ny, Nz)
    assert arrays.dispersive_a0.shape == (1, 3, Nx, Ny, Nz)
    assert arrays.dispersive_b1.shape == (1, 3, Nx, Ny, Nz)
    # polarization state keeps its (num_poles, 3, ...) shape
    assert arrays.fields.dispersive_x1.shape == (1, 3, Nx, Ny, Nz)

    a1_ref, a0_ref, b1_ref, _, _ = compute_pole_delta_coefficients_per_axis(
        material.dispersion.poles,  # type: ignore[union-attr]
        config.time_step_duration,
    )
    xs, ys, zs = placed.grid_slice
    for ax in range(3):
        assert jnp.allclose(arrays.dispersive_a1[0, ax, xs, ys, zs], a1_ref[0, ax])
        assert jnp.allclose(arrays.dispersive_a0[0, ax, xs, ys, zs], a0_ref[0, ax])
        assert jnp.allclose(arrays.dispersive_b1[0, ax, xs, ys, zs], b1_ref[0, ax])
    # the axis columns genuinely differ (x vs z resonance)
    assert not jnp.allclose(arrays.dispersive_a1[0, 0, xs, ys, zs], arrays.dispersive_a1[0, 2, xs, ys, zs])
    # outside the slab everything is zero
    inside_mask = jnp.zeros((Nx, Ny, Nz), dtype=bool).at[xs, ys, zs].set(True)
    assert jnp.all(arrays.dispersive_b1[0, :, ~inside_mask] == 0.0)


def test_isotropic_dispersion_keeps_1_component_axis(simple_config, simple_volume):
    """Purely isotropic dispersion must keep the memory-saving size-1 component
    axis (regression guard for the per-axis feature)."""
    obj = UniformMaterialObject(name="slab", partial_grid_shape=(10, 10, 10), material=_lorentz_material())
    constraint = GridCoordinateConstraint(
        object="slab", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    _, arrays, _, _, _ = place_objects([simple_volume, obj], simple_config, [constraint], key)
    assert arrays.dispersive_a1.shape[1] == 1


def test_static_multi_material_per_axis_coefficients(simple_config, simple_volume):
    """A Sphere with a per-axis Drude material bakes per-axis coefficients
    through the multi-material indexing path."""

    per_axis_drude = Material(
        permittivity=1.0,
        dispersion=DispersionModel(poles=(DrudePole(plasma_frequency=(1.37e16, 0.0, 0.0), damping=1e14),)),
    )
    materials = {
        "background": Material(permittivity=1.0),
        "drude": per_axis_drude,
    }
    sphere = Sphere(
        name="sphere",
        materials=materials,
        material_name="drude",
        radius=5.0 * simple_config.uniform_spacing(),
    )
    constraint = GridCoordinateConstraint(object="sphere", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[8, 8, 8])
    key = jax.random.PRNGKey(0)
    objects, arrays, _, config, _ = place_objects([simple_volume, sphere], simple_config, [constraint], key)
    placed = _placed(objects, "sphere")

    assert arrays.dispersive_b1 is not None
    assert arrays.dispersive_b1.shape[1] == 3

    a1_ref, _, b1_ref, _, _ = compute_pole_delta_coefficients_per_axis(
        per_axis_drude.dispersion.poles,  # type: ignore[union-attr]
        config.time_step_duration,
    )
    xs, ys, zs = placed.grid_slice
    voxel_mask = placed.get_voxel_mask_for_shape().astype(bool)
    inside_x = arrays.dispersive_b1[0, 0, xs, ys, zs]
    inside_y = arrays.dispersive_b1[0, 1, xs, ys, zs]
    # x axis carries the plasma coupling, y axis has none
    assert jnp.allclose(inside_x[voxel_mask], b1_ref[0, 0])
    assert b1_ref[0, 0] > 0.0
    assert jnp.all(inside_y[voxel_mask] == 0.0)
    # c1 is still non-zero on the y axis (recurrence exists, zero coupling)
    assert jnp.allclose(arrays.dispersive_a1[0, 1, xs, ys, zs][voxel_mask], a1_ref[0, 1])


def test_device_per_axis_dispersive_continuous_and_discrete(simple_config, simple_volume):
    """Devices write per-axis coefficient stacks through both the CONTINUOUS
    interpolation branch and the DISCRETE selection branch."""

    per_axis_drude = Material(
        permittivity=1.0,
        dispersion=DispersionModel(poles=(DrudePole(plasma_frequency=(1.37e16, 0.0, 5.0e15), damping=1e14),)),
    )
    materials = {
        "air": Material(permittivity=1.0),
        "drude": per_axis_drude,
    }
    for transforms in ([], [ClosestIndex()]):
        device = Device(
            name="device",
            partial_grid_shape=(10, 10, 10),
            partial_voxel_grid_shape=(5, 5, 5),
            materials=materials,
            param_transforms=transforms,
        )
        constraint = GridCoordinateConstraint(
            object="device", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
        )
        key = jax.random.PRNGKey(0)
        objects, arrays, params, config, _ = place_objects([simple_volume, device], simple_config, [constraint], key)
        placed_device = _placed(objects, "device")
        xs, ys, zs = placed_device.grid_slice

        drude_params = {name: jnp.ones_like(p) for name, p in params.items()}
        arrays, objects, _ = apply_params(arrays, objects, drude_params, key)

        assert arrays.dispersive_a1.shape[1] == 3
        a1_ref, _, b1_ref, _, _ = compute_pole_delta_coefficients_per_axis(
            per_axis_drude.dispersion.poles,  # type: ignore[union-attr]
            config.time_step_duration,
        )
        for ax in range(3):
            assert jnp.allclose(arrays.dispersive_a1[0, ax, xs, ys, zs], a1_ref[0, ax])
            assert jnp.allclose(arrays.dispersive_b1[0, ax, xs, ys, zs], b1_ref[0, ax])
        # x and z couplings differ by construction
        assert b1_ref[0, 0] != b1_ref[0, 2]


def test_full_tensor_sigma_plus_dispersive_allocates(simple_config, simple_volume):
    """Dispersion combined with an off-diagonal conductivity tensor runs through
    the fully anisotropic update path (which now carries the ADE block)."""
    sigma_tensor = Material(
        permittivity=2.0,
        electric_conductivity=(1.0, 0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0),  # off-diagonal
    )
    disp = _lorentz_material(eps_inf=2.0)
    obj1 = UniformMaterialObject(name="sigma_slab", partial_grid_shape=(6, 6, 6), material=sigma_tensor)
    obj2 = UniformMaterialObject(name="disp", partial_grid_shape=(6, 6, 6), material=disp)
    constraints = [
        GridCoordinateConstraint(object="sigma_slab", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[2, 2, 2]),
        GridCoordinateConstraint(object="disp", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[15, 15, 15]),
    ]
    key = jax.random.PRNGKey(0)
    _, arrays, _, _, _ = place_objects([simple_volume, obj1, obj2], simple_config, constraints, key)
    assert arrays.electric_conductivity is not None
    assert arrays.electric_conductivity.shape[0] == 9
    assert arrays.dispersive_b1 is not None
    # axis-aligned dispersion keeps its natural coupling tier
    assert arrays.dispersive_b1.shape[1] == 1


# ---------------------------------------------------------------------------
# Oriented (off-diagonal) dispersion
# ---------------------------------------------------------------------------


def _oriented_lorentz_material(eps_inf=2.0):
    from fdtdx.dispersion import LorentzPole

    return Material(
        permittivity=eps_inf,
        dispersion=DispersionModel(
            poles=(
                LorentzPole(
                    resonance_frequency=2e15,
                    damping=1e13,
                    delta_epsilon=1.5,
                    orientation=(1.0, 1.0, 0.0),
                ),
            )
        ),
    )


def test_oriented_dispersion_forces_tensor_tiers(simple_config, simple_volume):
    """An oriented pole widens the coupling to 9 components and forces the
    9-component permittivity tier so the fully anisotropic kernel runs."""

    material = _oriented_lorentz_material()
    obj = UniformMaterialObject(name="slab", partial_grid_shape=(10, 10, 10), material=material)
    constraint = GridCoordinateConstraint(
        object="slab", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    objects, arrays, _, config, _ = place_objects([simple_volume, obj], simple_config, [constraint], key)
    placed = _placed(objects, "slab")

    assert arrays.inv_permittivities.shape[0] == 9
    assert arrays.dispersive_a1.shape[1] == 3
    assert arrays.dispersive_b1.shape[1] == 9
    assert arrays.fields.dispersive_x1.shape[1] == 3

    a1_ref, _, b1_ref, _, _ = compute_pole_delta_coefficients_tensor(
        material.dispersion.poles,  # type: ignore[union-attr]
        config.time_step_duration,
    )
    xs, ys, zs = placed.grid_slice
    for entry in range(9):
        assert jnp.allclose(arrays.dispersive_b1[0, entry, xs, ys, zs], b1_ref[0, entry])
    # the coupling tensor genuinely has off-diagonal weight (u = (1,1,0)/sqrt(2))
    assert b1_ref[0, 1] != 0.0
    for ax in range(3):
        assert jnp.allclose(arrays.dispersive_a1[0, ax, xs, ys, zs], a1_ref[0, ax])


def test_oriented_dispersion_reversible_raises(simple_config, simple_volume):
    from fdtdx.config import GradientConfig
    from fdtdx.interfaces.recorder import Recorder

    material = _oriented_lorentz_material()
    obj = UniformMaterialObject(name="slab", partial_grid_shape=(10, 10, 10), material=material)
    constraint = GridCoordinateConstraint(
        object="slab", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    config = simple_config.aset("gradient_config", GradientConfig(method="reversible", recorder=Recorder(modules=[])))
    key = jax.random.PRNGKey(0)
    with pytest.raises(NotImplementedError, match="under active development"):
        place_objects([simple_volume, obj], config, [constraint], key)


def test_oriented_dispersion_nonuniform_grid_raises():
    """The symmetrized off-diagonal interface coupling assumes a uniform grid."""
    from fdtdx.core.grid import RectilinearGrid

    spacings = np.tile([1.0e-7, 1.3e-7], 15)
    edges = jnp.asarray(np.concatenate([[0.0], np.cumsum(spacings)]))
    grid = RectilinearGrid(x_edges=edges, y_edges=edges, z_edges=edges)
    config = SimulationConfig(grid=grid, time=1e-14, backend="cpu")
    volume = SimulationVolume(name="volume", partial_grid_shape=(30, 30, 30), material=_oriented_lorentz_material())
    key = jax.random.PRNGKey(0)
    with pytest.raises(NotImplementedError, match="non-uniform"):
        place_objects([volume], config, [], key)


def test_ccpr_edot_plus_tensor_path_raises(simple_config, simple_volume):
    """A CCPR pole with dE/dt coupling cannot be combined with off-diagonal
    material tensors: the tensor-branch ADE has no implicit c4 solve."""
    from fdtdx.dispersion import CCPRPole

    ccpr_material = Material(
        permittivity=1.0,
        dispersion=DispersionModel(poles=(CCPRPole(pole=complex(-2e13, -1.8e15), residue=complex(3e14, -6e14)),)),
    )
    aniso = Material(permittivity=(2.0, 0.1, 0.0, 0.1, 2.5, 0.0, 0.0, 0.0, 3.0))
    obj1 = UniformMaterialObject(name="ccpr", partial_grid_shape=(6, 6, 6), material=ccpr_material)
    obj2 = UniformMaterialObject(name="aniso", partial_grid_shape=(6, 6, 6), material=aniso)
    constraints = [
        GridCoordinateConstraint(object="ccpr", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[2, 2, 2]),
        GridCoordinateConstraint(object="aniso", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[15, 15, 15]),
    ]
    key = jax.random.PRNGKey(0)
    with pytest.raises(NotImplementedError, match="dE/dt"):
        place_objects([simple_volume, obj1, obj2], simple_config, constraints, key)


def test_oriented_static_multi_material_and_device(simple_config, simple_volume):
    """Oriented coupling tensors bake correctly through the multi-material
    indexing path and the Device apply_params path."""

    oriented = _oriented_lorentz_material(eps_inf=1.0)
    materials = {
        "air": Material(permittivity=1.0),
        "oriented": oriented,
    }
    sphere = Sphere(
        name="sphere",
        materials=materials,
        material_name="oriented",
        radius=5.0 * simple_config.uniform_spacing(),
    )
    constraint = GridCoordinateConstraint(object="sphere", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[8, 8, 8])
    key = jax.random.PRNGKey(0)
    objects, arrays, _, config, _ = place_objects([simple_volume, sphere], simple_config, [constraint], key)
    placed = _placed(objects, "sphere")

    assert arrays.dispersive_b1.shape[1] == 9
    _, _, b1_ref, _, _ = compute_pole_delta_coefficients_tensor(
        oriented.dispersion.poles,  # type: ignore[union-attr]
        config.time_step_duration,
    )
    xs, ys, zs = placed.grid_slice
    voxel_mask = placed.get_voxel_mask_for_shape().astype(bool)
    inside_xy = arrays.dispersive_b1[0, 1, xs, ys, zs]
    assert jnp.allclose(inside_xy[voxel_mask], b1_ref[0, 1])
    assert b1_ref[0, 1] != 0.0

    device = Device(
        name="device",
        partial_grid_shape=(10, 10, 10),
        partial_voxel_grid_shape=(5, 5, 5),
        materials=dict(materials),
        param_transforms=[],
    )
    constraint = GridCoordinateConstraint(
        object="device", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    objects, arrays, params, config, _ = place_objects([simple_volume, device], simple_config, [constraint], key)
    placed_device = _placed(objects, "device")
    xs, ys, zs = placed_device.grid_slice
    full_params = {name: jnp.ones_like(p) for name, p in params.items()}
    arrays, objects, _ = apply_params(arrays, objects, full_params, key)
    assert arrays.dispersive_b1.shape[1] == 9
    assert jnp.allclose(arrays.dispersive_b1[0, 1, xs, ys, zs], b1_ref[0, 1])


@pytest.mark.parametrize("integrator", ["centered_edot", "bilinear"])
def test_min_dispersive_divisor_matches_what_the_kernel_forms(simple_config, simple_volume, integrator):
    """``materials._min_dispersive_divisor`` must describe the divisor ``update_E``
    actually builds from the STORED arrays.

    The helper works from the P-form (c1..c5) in float64 and was deliberately left
    alone by the delta-basis change, so nothing in the type system ties it to the
    kernel any more — only this test does.
    """
    from fdtdx.constants import eta0
    from fdtdx.materials import _min_dispersive_divisor

    # Ag CCPR fit + static conductivity, i.e. c4 != 0 and the divisor non-trivial.
    poles = tuple(
        CCPRPole(pole=q, residue=r, integrator=integrator)
        for q, r in (
            (complex(-1.89e14, 0.0), complex(-1.00e18, 0.0)),
            (complex(-5.46e14, -6.37e15), complex(1.30e15, 1.54e15)),
            (complex(-5.68e14, -3.43e14), complex(1.61e17, 1.00e12)),
        )
    )
    mat = Material(permittivity=3.07, electric_conductivity=1.49e7, dispersion=DispersionModel(poles=poles))
    obj = UniformMaterialObject(name="silver", partial_grid_shape=(10, 10, 10), material=mat)
    constraint = GridCoordinateConstraint(
        object="silver", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    _, arrays, _, config, _ = place_objects([simple_volume, obj], simple_config, [constraint], jax.random.PRNGKey(0))
    sl = (slice(10, 20), slice(10, 20), slice(10, 20))

    c = config.courant_number
    inv_eps = arrays.inv_permittivities[(slice(None), *sl)]
    sigma_E = arrays.electric_conductivity[(slice(None), *sl)]
    kappa = c * sigma_E * eta0 * inv_eps / 2
    c4 = arrays.dispersive_c4[(slice(None), slice(None), *sl)]

    kernel_fwd = 1 + inv_eps * jnp.sum(c4, axis=0) + kappa

    dt = config.time_step_duration
    helper_fwd, _ = _min_dispersive_divisor(mat, dt)

    # isotropic material -> uniform over components and cells inside the slab
    assert jnp.allclose(kernel_fwd, kernel_fwd.min(), rtol=1e-6)
    # The helper is float64; the kernel expression above is rebuilt from the
    # float32 stored arrays. `simple_config` has a very coarse dt, so the pole and
    # conductivity terms are ~60x larger than the divisor they sum to (asserted
    # below), and float32 caps the agreement at ~1e-5 relative. That gap is the
    # test's own arithmetic, not a discrepancy in the helper.
    assert float(kernel_fwd.min()) == pytest.approx(helper_fwd, rel=3e-4)
    # not vacuous: the conductivity and pole terms are individually large and only
    # nearly cancel, so a dropped or mis-signed term would move the divisor a lot
    pole_term = float(jnp.abs(inv_eps * jnp.sum(c4, axis=0)).max())
    assert pole_term > 10.0 * abs(helper_fwd)
    assert float(jnp.abs(kappa).max()) > 10.0 * abs(helper_fwd)


def test_lorentz_divisor_is_exactly_one():
    """Lorentz/Drude poles on a central-difference scheme have c4 = 0, so the
    divisor must be *exactly* 1 — no float drift from the delta-basis rewrite."""
    from fdtdx.materials import _min_dispersive_divisor

    for integrator in ("central", "centered_edot"):
        model = DispersionModel(
            poles=(LorentzPole(resonance_frequency=3e15, damping=1e14, delta_epsilon=2.0, integrator=integrator),)
        )
        mat = Material(permittivity=2.25, dispersion=model)
        assert _min_dispersive_divisor(mat, 1e-18)[0] == 1.0
