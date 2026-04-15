"""Integration tests for dispersive-material array allocation and filling.

Covers the dispersive branches of ``_init_arrays`` (UniformMaterialObject,
StaticMultiMaterialObject) and ``apply_params`` (Device CONTINUOUS/DISCRETE).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.dispersion import DispersionModel, DrudePole, LorentzPole, compute_pole_coefficients
from fdtdx.fdtd.initialization import apply_params, place_objects
from fdtdx.materials import Material
from fdtdx.objects.device.device import Device
from fdtdx.objects.device.parameters.discretization import ClosestIndex
from fdtdx.objects.object import GridCoordinateConstraint
from fdtdx.objects.static_material.sphere import Sphere
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject


def _placed(container, name):
    """Return the placed copy of an object from an ObjectContainer by name."""
    for o in container.objects:
        if o.name == name:
            return o
    raise KeyError(name)


@pytest.fixture
def simple_config():
    return SimulationConfig(resolution=1e-7, time=1e-14, backend="cpu")


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
    assert arrays.dispersive_P_curr is not None
    assert arrays.dispersive_P_prev is not None
    assert arrays.dispersive_c1 is not None
    assert arrays.dispersive_c2 is not None
    assert arrays.dispersive_c3 is not None
    assert arrays.dispersive_P_curr.shape == (1, 3, Nx, Ny, Nz)
    assert arrays.dispersive_P_prev.shape == (1, 3, Nx, Ny, Nz)
    assert arrays.dispersive_c1.shape == (1, 1, Nx, Ny, Nz)
    assert arrays.dispersive_c2.shape == (1, 1, Nx, Ny, Nz)
    assert arrays.dispersive_c3.shape == (1, 1, Nx, Ny, Nz)
    # polarization always starts at zero
    assert jnp.all(arrays.dispersive_P_curr == 0)
    assert jnp.all(arrays.dispersive_P_prev == 0)

    # Coefficient values inside the slab should match compute_pole_coefficients.
    c1_ref, c2_ref, c3_ref = compute_pole_coefficients(
        material.dispersion.poles,
        config.time_step_duration,  # type: ignore[union-attr]
    )
    xs, ys, zs = placed.grid_slice
    assert jnp.allclose(arrays.dispersive_c1[0, 0, xs, ys, zs], c1_ref[0])
    assert jnp.allclose(arrays.dispersive_c2[0, 0, xs, ys, zs], c2_ref[0])
    assert jnp.allclose(arrays.dispersive_c3[0, 0, xs, ys, zs], c3_ref[0])

    # c3 should be zero outside the slab (vacuum cells have no polarization)
    inside_mask = jnp.zeros((Nx, Ny, Nz), dtype=bool).at[xs, ys, zs].set(True)
    assert jnp.all(arrays.dispersive_c3[0, 0][~inside_mask] == 0.0)


def test_no_dispersive_arrays_when_unused(simple_config, simple_volume):
    """A non-dispersive simulation leaves all dispersive arrays as None."""
    material = Material(permittivity=2.25)
    obj = UniformMaterialObject(name="slab", partial_grid_shape=(10, 10, 10), material=material)
    constraint = GridCoordinateConstraint(
        object="slab", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    _, arrays, _, _, _ = place_objects([simple_volume, obj], simple_config, [constraint], key)

    assert arrays.dispersive_P_curr is None
    assert arrays.dispersive_P_prev is None
    assert arrays.dispersive_c1 is None
    assert arrays.dispersive_c2 is None
    assert arrays.dispersive_c3 is None


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
    objects, arrays, _, config, _ = place_objects([simple_volume, obj1, obj2], simple_config, constraints, key)
    placed1 = _placed(objects, "one_pole_slab")
    placed2 = _placed(objects, "three_pole_slab")

    Nx, Ny, Nz = simple_volume.partial_grid_shape  # type: ignore[misc]
    assert arrays.dispersive_c1.shape == (3, 1, Nx, Ny, Nz)
    assert arrays.dispersive_P_curr.shape == (3, 3, Nx, Ny, Nz)

    # Inside the 1-pole slab: pole slot 0 is populated, slots 1 and 2 are zero.
    xs1, ys1, zs1 = placed1.grid_slice
    c1_1p_slot0 = arrays.dispersive_c1[0, 0, xs1, ys1, zs1]
    c1_1p_slot1 = arrays.dispersive_c1[1, 0, xs1, ys1, zs1]
    c1_1p_slot2 = arrays.dispersive_c1[2, 0, xs1, ys1, zs1]
    c3_1p_slot0 = arrays.dispersive_c3[0, 0, xs1, ys1, zs1]
    c3_1p_slot1 = arrays.dispersive_c3[1, 0, xs1, ys1, zs1]
    c3_1p_slot2 = arrays.dispersive_c3[2, 0, xs1, ys1, zs1]
    assert jnp.all(c1_1p_slot0 != 0.0)
    assert jnp.all(c1_1p_slot1 == 0.0)
    assert jnp.all(c1_1p_slot2 == 0.0)
    assert jnp.all(c3_1p_slot0 > 0.0)
    assert jnp.all(c3_1p_slot1 == 0.0)
    assert jnp.all(c3_1p_slot2 == 0.0)

    # Inside the 3-pole slab: all three slots populated.
    xs2, ys2, zs2 = placed2.grid_slice
    for p in range(3):
        c3_slot = arrays.dispersive_c3[p, 0, xs2, ys2, zs2]
        assert jnp.all(c3_slot > 0.0)


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

    assert arrays.dispersive_c1 is not None
    xs_d, ys_d, zs_d = placed_disp.grid_slice
    xs_p, ys_p, zs_p = placed_plain.grid_slice
    # Dispersive slab: c3 non-zero
    assert jnp.all(arrays.dispersive_c3[0, 0, xs_d, ys_d, zs_d] > 0.0)
    # Non-dispersive slab: c3 zero
    assert jnp.all(arrays.dispersive_c3[0, 0, xs_p, ys_p, zs_p] == 0.0)


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
        partial_grid_shape=(14, 14, 14),
        materials=materials,
        material_name="drude",
        radius=5.0 * simple_config.resolution,
    )
    constraint = GridCoordinateConstraint(object="sphere", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[8, 8, 8])
    key = jax.random.PRNGKey(0)
    objects, arrays, _, _, _ = place_objects([simple_volume, sphere], simple_config, [constraint], key)
    placed = _placed(objects, "sphere")

    assert arrays.dispersive_c3 is not None
    assert arrays.dispersive_c3.shape[0] == 1  # one pole total (Drude)

    # Outside the bounding box → strictly zero
    Nx, Ny, Nz = simple_volume.partial_grid_shape  # type: ignore[misc]
    xs, ys, zs = placed.grid_slice
    inside_mask = jnp.zeros((Nx, Ny, Nz), dtype=bool).at[xs, ys, zs].set(True)
    assert jnp.all(arrays.dispersive_c3[0, 0][~inside_mask] == 0.0)

    # Inside the voxel mask → non-zero
    voxel_mask = placed.get_voxel_mask_for_shape().astype(bool)
    inside_slab = arrays.dispersive_c3[0, 0, xs, ys, zs]
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

    assert arrays.dispersive_c1 is not None
    assert arrays.dispersive_c1.shape == (1, 1, 30, 30, 30)

    # Inside the device: coefficients should equal the drude coefficients.
    c1_ref, c2_ref, c3_ref = compute_pole_coefficients(
        materials["drude"].dispersion.poles,  # type: ignore[union-attr]
        config.time_step_duration,
    )
    assert jnp.allclose(arrays.dispersive_c1[0, 0, xs, ys, zs], c1_ref[0])
    assert jnp.allclose(arrays.dispersive_c2[0, 0, xs, ys, zs], c2_ref[0])
    assert jnp.allclose(arrays.dispersive_c3[0, 0, xs, ys, zs], c3_ref[0])

    # All-air: params=0 -> coefficients should be zero (air has no dispersion)
    air_params = {name: jnp.zeros_like(p) for name, p in params.items()}
    arrays2, _, _ = apply_params(arrays, objects, air_params, key)
    assert jnp.all(arrays2.dispersive_c1[0, 0, xs, ys, zs] == 0.0)
    assert jnp.all(arrays2.dispersive_c3[0, 0, xs, ys, zs] == 0.0)

    # Half interpolation: params=0.5 -> coefficients should be half the drude values
    half_params = {name: 0.5 * jnp.ones_like(p) for name, p in params.items()}
    arrays3, _, _ = apply_params(arrays, objects, half_params, key)
    assert jnp.allclose(arrays3.dispersive_c1[0, 0, xs, ys, zs], 0.5 * c1_ref[0])
    assert jnp.allclose(arrays3.dispersive_c3[0, 0, xs, ys, zs], 0.5 * c3_ref[0])


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

    c1_ref, c2_ref, c3_ref = compute_pole_coefficients(
        materials["drude"].dispersion.poles,  # type: ignore[union-attr]
        config.time_step_duration,
    )
    assert jnp.allclose(arrays.dispersive_c1[0, 0, xs, ys, zs], c1_ref[0])
    assert jnp.allclose(arrays.dispersive_c3[0, 0, xs, ys, zs], c3_ref[0])

    # All air: coefficients should be zero
    air_params = {name: jnp.zeros_like(p) for name, p in params.items()}
    arrays2, _, _ = apply_params(arrays, objects, air_params, key)
    assert jnp.all(arrays2.dispersive_c3[0, 0, xs, ys, zs] == 0.0)


def test_fully_anisotropic_plus_dispersive_raises(simple_config, simple_volume):
    """Combining a fully anisotropic permittivity tensor with a dispersive material
    should raise NotImplementedError from _init_arrays."""
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
    key = jax.random.PRNGKey(0)
    with pytest.raises(NotImplementedError, match="fully anisotropic"):
        place_objects([simple_volume, obj1, obj2], simple_config, constraints, key)


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

    c1_ref, _, c3_ref = compute_pole_coefficients(
        materials["drude"].dispersion.poles,  # type: ignore[union-attr]
        config.time_step_duration,
    )
    # 0.5 * (air_coeffs = 0) + 0.5 * (drude_coeffs)
    assert jnp.allclose(arrays_half.dispersive_c1[0, 0, xs, ys, zs], 0.5 * c1_ref[0])
    assert jnp.allclose(arrays_half.dispersive_c3[0, 0, xs, ys, zs], 0.5 * c3_ref[0])
    # All coefficients outside the device slice remain zero
    mask = jnp.zeros(simple_volume.partial_grid_shape, dtype=bool).at[xs, ys, zs].set(True)  # type: ignore[misc]
    assert jnp.all(arrays_half.dispersive_c1[0, 0][~mask] == 0.0)


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
    omega = 2.0 * np.pi * 3e8 / wavelength
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
