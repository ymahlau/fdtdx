"""Hyperbolic (indefinite) medium via per-axis Drude dispersion.

A hyperbolic material has a permittivity tensor with mixed-sign components:
metallic (Re eps < 0) along some axes and dielectric (Re eps > 0) along the
others. In explicit FDTD a *static* negative permittivity is unconditionally
unstable, so the metallic axes must be modeled with dispersion poles that
drive Re eps(omega) negative in-band while eps_inf stays >= 1. Per-axis pole
parameters make this possible: here a Drude plasma frequency acts only on
the x axis, giving

    eps_x(omega) = 1 - omega_p^2 / (omega^2 + i gamma omega)   (metallic)
    eps_y = eps_z = 1                                          (dielectric)

which is an indefinite (type-I-like hyperbolic) tensor for omega < omega_p.

The script:

    1. Plots Re eps_x(omega) and Re eps_y(omega) from
       ``DispersionModel.susceptibility_axes``, marking the hyperbolic band.
    2. Runs two normal-incidence simulations onto a half-space of the
       material — one x-polarized, one y-polarized — plus a vacuum reference.
    3. Logs the transmitted power fractions: the x polarization is strongly
       reflected (metallic response) while the y polarization passes,
       matching the analytic Fresnel values from the per-axis permittivity.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

import fdtdx
from fdtdx.constants import c as c0

WAVELENGTH = 1e-6
OMEGA = 2.0 * np.pi * c0 / WAVELENGTH

# Drude pole on the x axis only: metallic for omega < omega_p.
OMEGA_P = 5.0 * OMEGA
GAMMA = 0.05 * OMEGA
EPS_INF = 1.0

RESOLUTION = 25e-9
PML_CELLS = 10
DOMAIN_XY = 3 * RESOLUTION
DOMAIN_Z = 5e-6
Z_CELLS = int(round(DOMAIN_Z / RESOLUTION))

SOURCE_Z = PML_CELLS + 2
INTERFACE_Z = Z_CELLS // 2
DET_T_Z = INTERFACE_Z + 40

SIM_TIME = 120e-15


def build_model() -> fdtdx.DispersionModel:
    return fdtdx.DispersionModel(
        poles=(fdtdx.DrudePole(plasma_frequency=(OMEGA_P, 0.0, 0.0), damping=GAMMA),)
    )


def plot_permittivity(model: fdtdx.DispersionModel) -> None:
    omegas = np.linspace(0.2 * OMEGA, 1.5 * OMEGA_P, 400)
    eps_x = np.array([model.permittivity_axes(w, eps_inf=EPS_INF)[0] for w in omegas])
    eps_y = np.array([model.permittivity_axes(w, eps_inf=EPS_INF)[1] for w in omegas])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(omegas / OMEGA, eps_x.real, label=r"Re $\varepsilon_x(\omega)$ (Drude axis)")
    ax.plot(omegas / OMEGA, eps_y.real, label=r"Re $\varepsilon_y(\omega) = \varepsilon_z(\omega)$")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(1.0, color="gray", ls="--", lw=0.8, label=r"source $\omega$")
    hyperbolic = (eps_x.real < 0) & (eps_y.real > 0)
    if np.any(hyperbolic):
        band = omegas[hyperbolic] / OMEGA
        ax.axvspan(float(band.min()), float(band.max()), alpha=0.12, color="tab:red", label="hyperbolic band")
    ax.set_xlabel(r"$\omega / \omega_{\mathrm{src}}$")
    ax.set_ylabel(r"Re $\varepsilon$")
    ax.set_title("Per-axis Drude model: indefinite permittivity tensor")
    ax.legend()
    fig.tight_layout()
    fig.savefig("hyperbolic_permittivity.png", dpi=150)
    logger.info("saved hyperbolic_permittivity.png")


def build_scene(polarization, material=None):
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=RESOLUTION),
        time=SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(partial_real_shape=(DOMAIN_XY, DOMAIN_XY, DOMAIN_Z))
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=PML_CELLS,
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

    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=WAVELENGTH),
        direction="+",
        fixed_E_polarization_vector=polarization,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(SOURCE_Z,)),
        ]
    )
    objects.append(source)

    if material is not None:
        slab = fdtdx.UniformMaterialObject(
            partial_grid_shape=(None, None, Z_CELLS - INTERFACE_Z),
            material=material,
        )
        constraints.extend(
            [
                slab.same_size(volume, axes=(0, 1)),
                slab.place_at_center(volume, axes=(0, 1)),
                slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(INTERFACE_Z,)),
            ]
        )
        objects.append(slab)

    det = fdtdx.PoyntingFluxDetector(
        name="flux_t",
        partial_grid_shape=(None, None, 1),
        direction="+",
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 1)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(DET_T_Z,)),
        ]
    )
    objects.append(det)

    return objects, constraints, config


def transmitted_flux(polarization, material=None) -> float:
    objects, constraints, config = build_scene(polarization, material)
    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj, config=config, key=key)
    flux = np.array(arrays.detector_states["flux_t"]["poynting_flux"][:, 0])
    dt = config.time_step_duration
    steps_per_period = int(round(WAVELENGTH / (c0 * dt)))
    return float(np.mean(flux[-10 * steps_per_period :]))


def fresnel_transmission(eps: complex) -> float:
    n2 = np.sqrt(eps)
    t = 2.0 / (1.0 + n2)
    return float(np.real(n2) * np.abs(t) ** 2)


def main() -> None:
    model = build_model()
    plot_permittivity(model)

    eps_axes = model.permittivity_axes(OMEGA, eps_inf=EPS_INF)
    logger.info(f"eps at source frequency: eps_x = {eps_axes[0]:.3f}, eps_y = {eps_axes[1]:.3f}")
    material = fdtdx.Material(permittivity=EPS_INF, dispersion=model)

    s0_x = transmitted_flux((1, 0, 0))
    s0_y = transmitted_flux((0, 1, 0))
    t_x = transmitted_flux((1, 0, 0), material) / s0_x
    t_y = transmitted_flux((0, 1, 0), material) / s0_y

    logger.info(
        f"x polarization (metallic axis):   T = {t_x:.4f}  "
        f"(interface Fresnel: {fresnel_transmission(eps_axes[0]):.4f}; the detector sits 1 um "
        "inside the medium, far past the skin depth, so nearly nothing survives)"
    )
    logger.info(f"y polarization (dielectric axis): T = {t_y:.4f}  (Fresnel: {fresnel_transmission(eps_axes[1]):.4f})")
    logger.info(f"polarization selectivity T_y - T_x = {t_y - t_x:.4f}")


if __name__ == "__main__":
    main()
