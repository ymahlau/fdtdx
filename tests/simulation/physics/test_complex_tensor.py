"""Physics simulation test for complex full-tensor (non-diagonal) permittivity.

``Material.from_complex_permittivity`` maps a complex 3x3 tensor to a real
permittivity tensor plus a real conductivity tensor (sigma_ij = omega eps0
eps''_ij, exact at the reference frequency). This is the first end-to-end
exercise of the fully anisotropic sigma_E branch of the update equations.

Setup: a lossy uniaxial crystal with principal (complex) permittivities
(eps_a, eps_b, eps_b) and the optical axis in the transverse plane.

* Run A (reference, grid-aligned): diagonal complex tensor, source polarized
  at 45 degrees in the xy plane. The wave splits equally into the two
  principal polarizations.
* Run B (rotated): the same crystal rotated 45 degrees about the propagation
  axis, built as the nested 3x3 tuple R eps R^T (complex off-diagonals!),
  with an x-polarized source. Physically identical decomposition.

The transmitted fluxes of both runs must agree, and match the analytic
average of the two principal Fresnel transmissions (with in-medium
absorption over the propagation distance to the detector).

Layout mirrors ``test_dispersion.py`` — 3x3 periodic transverse, PMLs in z,
``UniformPlaneSource`` in +z, one transmission-side Poynting flux detector.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx
from fdtdx.constants import c as c0

_WAVELENGTH = 1e-6
_OMEGA = 2.0 * np.pi * c0 / _WAVELENGTH
_RESOLUTION = 25e-9
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION
_DOMAIN_Z = 5e-6
_Z_CELLS = round(_DOMAIN_Z / _RESOLUTION)

_SOURCE_Z = _PML_CELLS + 2
_INTERFACE_Z = 100
_DET_T_Z = 140
_DIEL_CELLS_Z = _Z_CELLS - _INTERFACE_Z

_SIM_TIME = 120e-15
_TOLERANCE = 0.05

_DT_APPROX = 0.99 * _RESOLUTION / (c0 * np.sqrt(3))
_STEPS_PER_PERIOD = round(_WAVELENGTH / (c0 * _DT_APPROX))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD

# Lossy uniaxial crystal: distinct real parts AND distinct absorption.
_EPS_A = 4.0 + 0.40j
_EPS_B = 2.0 + 0.10j

_SQRT_HALF = float(1.0 / np.sqrt(2.0))


def _rotated_tensor_45deg():
    """R eps R^T for a 45-degree rotation about z, as a nested 3x3 tuple."""
    avg = (_EPS_A + _EPS_B) / 2.0
    off = (_EPS_A - _EPS_B) / 2.0
    return (
        (avg, off, 0.0),
        (off, avg, 0.0),
        (0.0, 0.0, _EPS_B),
    )


def _analytic_transmitted_fraction(eps_complex: complex, distance: float) -> float:
    """Fraction of incident power measured ``distance`` into a semi-infinite
    lossy medium: Fresnel interface transmission times exp(-2 Im(k) d)."""
    n2 = np.sqrt(eps_complex)
    t = 2.0 / (1.0 + n2)
    interface_T = float(np.real(n2) * np.abs(t) ** 2)
    k_imag = float(np.imag(n2)) * _OMEGA / c0
    return interface_T * float(np.exp(-2.0 * k_imag * distance))


def _build_base(polarization):
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_x": "periodic",
            "max_x": "periodic",
            "min_y": "periodic",
            "max_y": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=polarization,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    return objects, constraints, config, volume


def _add_flux_det(name, z_idx, volume, objects, constraints):
    det = fdtdx.PoyntingFluxDetector(
        name=name,
        partial_grid_shape=(None, None, 1),
        direction="+",
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 1)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(z_idx,)),
        ]
    )
    objects.append(det)


def _add_half_space(material, volume, objects, constraints):
    slab = fdtdx.UniformMaterialObject(
        partial_grid_shape=(None, None, _DIEL_CELLS_Z),
        material=material,
    )
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_INTERFACE_Z,)),
        ]
    )
    objects.append(slab)


def _run(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    return arrays


def _mean_flux(arrays, name):
    flux = np.array(arrays.detector_states[name]["poynting_flux"][:, 0])
    return float(np.mean(flux[-_N_AVG_STEPS:]))


def _transmitted_flux(material, polarization):
    obj, con, cfg, vol = _build_base(polarization)
    if material is not None:
        _add_half_space(material, vol, obj, con)
    _add_flux_det("flux_t", _DET_T_Z, vol, obj, con)
    return _mean_flux(_run(obj, con, cfg), "flux_t")


def test_rotated_lossy_uniaxial_crystal_matches_grid_aligned():
    """The 45-degree rotated complex tensor (full sigma_E branch) must
    transmit the same power as the physically identical grid-aligned setup
    (diagonal branch), and both must match the analytic expectation."""
    frequency = c0 / _WAVELENGTH

    # Run A: grid-aligned diagonal crystal, 45-degree polarized source.
    diag_material = fdtdx.Material.from_complex_permittivity(
        (_EPS_A, _EPS_B, _EPS_B),
        frequency=frequency,
    )
    s_diag = _transmitted_flux(diag_material, (_SQRT_HALF, _SQRT_HALF, 0.0))

    # Run B: crystal rotated 45 degrees about z (nested complex 3x3 input),
    # x-polarized source. Exercises the full-tensor eps AND sigma_E update.
    rot_material = fdtdx.Material.from_complex_permittivity(
        _rotated_tensor_45deg(),
        frequency=frequency,
    )
    assert not rot_material.is_diagonally_anisotropic_permittivity
    assert not rot_material.is_diagonally_anisotropic_electric_conductivity
    s_rot = _transmitted_flux(rot_material, (1.0, 0.0, 0.0))

    assert s_diag > 0, f"grid-aligned transmitted flux vanished: {s_diag}"
    assert s_rot > 0, f"rotated-crystal transmitted flux vanished: {s_rot}"

    rel = abs(s_rot - s_diag) / s_diag
    assert rel < _TOLERANCE, (
        f"rotated crystal S={s_rot:.4e} disagrees with grid-aligned S={s_diag:.4e}, rel={rel:.3f} > {_TOLERANCE}"
    )

    # Both must also match the analytic expectation: equal power split into
    # the two principal polarizations, each with its own Fresnel + absorption.
    s_vac = _transmitted_flux(None, (_SQRT_HALF, _SQRT_HALF, 0.0))
    distance = (_DET_T_Z - _INTERFACE_Z) * _RESOLUTION
    t_analytic = 0.5 * (
        _analytic_transmitted_fraction(_EPS_A, distance) + _analytic_transmitted_fraction(_EPS_B, distance)
    )
    t_diag = s_diag / s_vac
    t_rot = s_rot / s_vac
    assert abs(t_diag - t_analytic) / t_analytic < _TOLERANCE, (
        f"grid-aligned T={t_diag:.4f} vs analytic {t_analytic:.4f}"
    )
    assert abs(t_rot - t_analytic) / t_analytic < 2 * _TOLERANCE, (
        f"rotated-crystal T={t_rot:.4f} vs analytic {t_analytic:.4f} "
        "(looser tolerance: off-diagonal Yee averaging is first-order at the interface)"
    )
