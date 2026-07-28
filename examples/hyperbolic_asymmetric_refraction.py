"""Asymmetric refraction by a tilted hyperbolic crystal (oriented poles).

Calcite driven inside its ordinary Reststrahlen band (here 1470 cm^-1,
lambda = 6.8 um) has Re eps < 0 across the ordinary plane and Re eps > 0
along the optic axis. Tilting the optic axis away from the surface normal —
here 25 degrees above the surface, in the plane of incidence — puts a
frequency-dependent off-diagonal component eps_xz(omega) into the lab-frame
tensor. Both ingredients need dispersion: the mixed-sign tensor because a
static negative permittivity is unconditionally unstable in FDTD, and the
tilt because per-axis (diagonal) poles can only describe grid-aligned
optical axes.

The tilt shears the iso-frequency curve and breaks the kx -> -kx mirror
symmetry of the interface: beams incident at +45 and -45 degrees refract to
+23 and +53 degrees — one ordinary, one negative — which no grid-aligned
tensor can produce.

The script:

    1. Builds calcite from a per-axis two-pole Lorentz model and tilts it
       with ``DispersionModel.rotated``.
    2. Plots the lab-frame tensor components eps_xx, eps_zz and eps_xz.
    3. Runs two mirrored 45-degree TM Gaussian beams onto the tilted slab.
    4. Measures each beam's energy-corridor angle inside the slab against
       the analytic group-velocity direction.

Runs in a few minutes on a GPU (about 1.5M cells, fully anisotropic kernel).

Outputs (to the working directory):
    asym_refraction_tensor.png     — eps tensor components over the band
    asym_refraction_field.png      — final H_y snapshot with analytic rays
    asym_refraction_intensity.png  — time-averaged field with analytic rays
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

import fdtdx
from fdtdx.constants import c as c0

DRIVE_CM = 1470.17  # 1/cm, inside the ordinary Reststrahlen band
WAVELENGTH = 1e-2 / DRIVE_CM  # 6.80 um
OMEGA = 2.0 * np.pi * c0 / WAVELENGTH

# Optic axis 25 degrees above the surface, in the plane of incidence.
TILT_RAD = np.radians(65.0)

# Two-pole Lorentz fit (E_u ordinary, A_2u extraordinary) matching the
# tabulated calcite tensor at the drive frequency. Crystal frame: x/y
# ordinary, z extraordinary.
EPS_INF_CRYSTAL = (2.5898, 2.5898, 2.4691)


def _cm(v: float) -> float:
    return 2.0 * np.pi * c0 * v * 100.0  # wavenumber (1/cm) -> rad/s


def build_material():
    crystal_frame = fdtdx.DispersionModel(
        poles=(
            fdtdx.LorentzPole(
                resonance_frequency=_cm(1410.0),
                damping=_cm(10.0),
                delta_epsilon=(0.5416, 0.5416, 0.0),
            ),
            fdtdx.LorentzPole(
                resonance_frequency=_cm(871.0),
                damping=_cm(3.0),
                delta_epsilon=(0.0, 0.0, 0.3745),
            ),
        )
    )
    model = crystal_frame.rotated((0.0, float(TILT_RAD), 0.0))
    ct, st = np.cos(TILT_RAD), np.sin(TILT_RAD)
    r_mat = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    (xx, xy, xz), (yx, yy, yz), (zx, zy, zz) = (r_mat @ np.diag(EPS_INF_CRYSTAL) @ r_mat.T).tolist()
    eps_inf_lab = ((xx, xy, xz), (yx, yy, yz), (zx, zy, zz))
    material = fdtdx.Material(permittivity=eps_inf_lab, dispersion=model)
    return material, eps_inf_lab


MATERIAL, EPS_INF_LAB = build_material()

THETA_DEG = 45.0
RESOLUTION = 0.17e-6
DOMAIN_X = 170e-6
DOMAIN_Y = 1.36e-6  # quasi-2D: y is periodic
DOMAIN_Z = 32e-6
PML_CELLS = 12
SIM_TIME = 800e-15

NX = round(DOMAIN_X / RESOLUTION)
NY = round(DOMAIN_Y / RESOLUTION)
NZ = round(DOMAIN_Z / RESOLUTION)
SLAB_TOP_Z = round(20e-6 / RESOLUTION)
SLAB_BOT_Z = SLAB_TOP_Z - round(9e-6 / RESOLUTION)
SOURCE_Z = NZ - PML_CELLS - 6

BEAM_RADIUS = 17e-6
# Beam A (+45 deg, marching +x) hits the slab at ~40 um, beam B (-45 deg,
# marching -x) at ~125 um: incident, reflected and refracted paths never
# overlap.
SRC_A_X = 31.1e-6
SRC_B_X = 133.9e-6

FRAME_INTERVAL = 6  # ~12 field samples per optical period


def eps_xz_block(omega: float) -> np.ndarray:
    """The 2x2 xz block of the lab-frame permittivity tensor at ``omega``."""
    model = MATERIAL.dispersion
    assert model is not None
    full = model.permittivity_tensor(omega, eps_inf=EPS_INF_LAB)
    return np.array([[full[0, 0], full[0, 2]], [full[2, 0], full[2, 2]]])


def analytic_energy_angle(theta_inc_deg: float) -> float:
    """TM energy-refraction angle into the crystal, in degrees from -z.

    Solves k_perp^T eps^{-1} k_perp = k0^2 with k_perp = (-kz, kx) for the
    downward branch; group velocity g ~ (inv_zz kx - inv_xz kz,
    inv_xx kz - inv_xz kx).
    """
    inv = np.linalg.inv(np.real(eps_xz_block(OMEGA)))
    kx = float(np.sin(np.deg2rad(theta_inc_deg)))
    a = inv[0, 0]
    b = -2.0 * inv[0, 1] * kx
    c = inv[1, 1] * kx**2 - 1.0
    disc = b**2 - 4.0 * a * c
    if disc < 0:
        raise ValueError("TM wave is evanescent in the crystal at this angle")
    for sign in (+1.0, -1.0):
        kz = (-b + sign * np.sqrt(disc)) / (2.0 * a)
        gx = inv[1, 1] * kx - inv[0, 1] * kz
        gz = inv[0, 0] * kz - inv[0, 1] * kx
        if gz < 0:  # energy flows down, into the crystal
            return float(np.degrees(np.arctan2(gx, -gz)))
    raise ValueError("no downward-energy branch found")


def plot_tensor_components() -> None:
    model = MATERIAL.dispersion
    assert model is not None
    freqs_cm = np.linspace(1300.0, 1600.0, 400)
    eps = np.array([model.permittivity_tensor(_cm(f), eps_inf=EPS_INF_LAB) for f in freqs_cm])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(freqs_cm, eps[:, 0, 0].real, label=r"Re $\varepsilon_{xx}$")
    ax.plot(freqs_cm, eps[:, 2, 2].real, label=r"Re $\varepsilon_{zz}$")
    ax.plot(freqs_cm, eps[:, 0, 2].real, label=r"Re $\varepsilon_{xz}$ (from the tilt)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(DRIVE_CM, color="gray", ls="--", lw=0.8, label="drive frequency")
    ax.set_xlabel(r"frequency (cm$^{-1}$)")
    ax.set_ylabel(r"Re $\varepsilon$")
    ax.set_title("Tilted calcite: lab-frame permittivity tensor components")
    ax.legend()
    fig.tight_layout()
    fig.savefig("asym_refraction_tensor.png", dpi=150)
    plt.close(fig)
    logger.info("saved asym_refraction_tensor.png")


def build_scene():
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=RESOLUTION),
        time=SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(partial_real_shape=(DOMAIN_X, DOMAIN_Y, DOMAIN_Z))
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=PML_CELLS,
        override_types={"min_y": "periodic", "max_y": "periodic"},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    slab = fdtdx.UniformMaterialObject(
        name="calcite_slab",
        partial_grid_shape=(None, None, SLAB_TOP_Z - SLAB_BOT_Z),
        material=MATERIAL,
    )
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(SLAB_BOT_Z,)),
        ]
    )
    objects.append(slab)

    # Absorber walls at the slab's x-ends. PMLs amplify rather than absorb
    # the backward waves of hyperbolic media — including a hyperbolic absorber
    # layer itself — so the loss grading ends in a plain conductor inside the
    # PML.
    layer_cells = 12
    wall_specs = [
        (0, 8e3, False),
        (layer_cells, 6e3, True),
        (2 * layer_cells, 1.5e3, True),
        (NX - 3 * layer_cells, 1.5e3, True),
        (NX - 2 * layer_cells, 6e3, True),
        (NX - layer_cells, 8e3, False),
    ]
    for wi, (x_start, sigma, crystal) in enumerate(wall_specs):
        if crystal:
            wall_material = fdtdx.Material(
                permittivity=EPS_INF_LAB,
                electric_conductivity=sigma,
                dispersion=MATERIAL.dispersion,
            )
        else:
            wall_material = fdtdx.Material(permittivity=2.5, electric_conductivity=sigma)
        wall = fdtdx.UniformMaterialObject(
            name=f"slab_absorber_{wi}",
            partial_grid_shape=(layer_cells, None, SLAB_TOP_Z - SLAB_BOT_Z),
            material=wall_material,
        )
        constraints.extend(
            [
                wall.same_size(volume, axes=(1,)),
                wall.place_at_center(volume, axes=(1,)),
                wall.set_grid_coordinates(axes=(0, 2), sides=("-", "-"), coordinates=(x_start, SLAB_BOT_Z)),
            ]
        )
        objects.append(wall)

    # Two mirrored TM beams (E along x, H along y); azimuth_angle tilts the
    # propagation in the xz plane. The source planes must stay clear of the
    # x-PML.
    source_width = round(2.9 * BEAM_RADIUS / RESOLUTION)
    for name, src_x, sgn in (("beam_a", SRC_A_X, +1.0), ("beam_b", SRC_B_X, -1.0)):
        source = fdtdx.GaussianPlaneSource(
            name=name,
            partial_grid_shape=(source_width, None, 1),
            fixed_E_polarization_vector=(1, 0, 0),
            wave_character=fdtdx.WaveCharacter(wavelength=WAVELENGTH),
            radius=BEAM_RADIUS,
            std=0.25,
            direction="-",
            azimuth_angle=float(-sgn * THETA_DEG),
        )
        x_left = round(src_x / RESOLUTION) - source_width // 2
        constraints.extend(
            [
                source.same_size(volume, axes=(1,)),
                source.place_at_center(volume, axes=(1,)),
                source.set_grid_coordinates(axes=(0, 2), sides=("-", "-"), coordinates=(x_left, SOURCE_Z)),
            ]
        )
        objects.append(source)

    det = fdtdx.FieldDetector(
        name="hy_side",
        components=("Hy",),
        partial_grid_shape=(None, 1, None),
        switch=fdtdx.OnOffSwitch(interval=FRAME_INTERVAL),
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 2)),
            det.place_at_center(volume, axes=(0, 2)),
            det.set_grid_coordinates(axes=(1,), sides=("-",), coordinates=(NY // 2,)),
        ]
    )
    objects.append(det)

    return objects, constraints, config


def run() -> np.ndarray:
    objects, constraints, config = build_scene()
    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj, config=config, key=key)
    frames = np.asarray(arrays.detector_states["hy_side"]["fields"]).squeeze()
    assert np.isfinite(frames).all(), "non-finite fields recorded"
    return frames  # (T, NX, NZ)


def measure_corridor_angle(intensity: np.ndarray, hit_x_um: float) -> float:
    """Energy-corridor angle from the intensity ridge inside the slab.

    Follows the ridge row by row from the entry face down, then fits a line
    through the mid-slab rows only: the first ~1.5 um below the face carry a
    bright interface wave, and the deepest rows have little signal left.
    """
    window = round(3e-6 / RESOLUTION)
    skip_face = round(1.5e-6 / RESOLUTION)
    skip_exit = round(2e-6 / RESOLUTION)
    x_prev = hit_x_um * 1e-6 / RESOLUTION
    xs, zs = [], []
    for z in np.arange(SLAB_TOP_Z - 3, SLAB_BOT_Z + 3, -1):
        lo = max(int(round(x_prev)) - window, 0)
        hi = min(int(round(x_prev)) + window, intensity.shape[0])
        row = intensity[lo:hi, z]
        weights = np.clip(row - np.median(row), 0.0, None) ** 2
        if weights.sum() <= 0:
            continue
        x_c = float(np.average(np.arange(lo, hi), weights=weights))
        x_prev = x_c
        if SLAB_TOP_Z - z >= skip_face and z - SLAB_BOT_Z >= skip_exit:
            xs.append(x_c)
            zs.append(float(z))
    slope = np.polyfit(zs, xs, 1)[0]  # dx/dz, z decreasing downward
    return float(np.degrees(np.arctan(-slope)))


def make_plots(frames: np.ndarray, angles: dict[float, float]) -> None:
    late = frames[frames.shape[0] // 2 :]
    scale = float(np.percentile(np.abs(late), 99.5)) + 1e-30
    extent = (0, DOMAIN_X * 1e6, 0, DOMAIN_Z * 1e6)
    z_top = SLAB_TOP_Z * RESOLUTION * 1e6
    z_bot = SLAB_BOT_Z * RESOLUTION * 1e6
    z_src = SOURCE_Z * RESOLUTION * 1e6

    def rays(ax, color_in, colors):
        for sgn, src_x in ((+1.0, SRC_A_X * 1e6), (-1.0, SRC_B_X * 1e6)):
            ts = angles[sgn * THETA_DEG]
            hit = src_x + sgn * (z_src - z_top)
            xb = hit + np.tan(np.deg2rad(ts)) * (z_top - z_bot)
            ax.plot([src_x, hit], [z_src, z_top], ls="--", lw=0.9, color=color_in, alpha=0.7)
            ax.plot(
                [hit, xb],
                [z_top, z_bot],
                ls="--",
                lw=1.3,
                color=colors[sgn],
                label=f"{sgn * THETA_DEG:+.0f}° in → {ts:+.1f}° energy (analytic)",
            )
            ax.plot([xb, xb + sgn * 3.0], [z_bot, z_bot - 3.0], ls=":", lw=0.9, color=color_in, alpha=0.7)

    fig, ax = plt.subplots(figsize=(12, 5.4))
    im = ax.imshow(
        np.clip(frames[-1] / scale, -1, 1).T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=extent,
        aspect="equal",
        interpolation="bilinear",
    )
    for z in (z_top, z_bot):
        ax.axhline(z, color="k", lw=0.9)
    rays(ax, "0.3", {+1.0: "0.1", -1.0: "0.1"})
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("z (μm)")
    ax.set_title(f"Tilted calcite at {DRIVE_CM:.0f} cm$^{{-1}}$: asymmetric refraction of ±45° beams")
    fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.85, label=r"$H_y$ (norm.)")
    fig.tight_layout()
    fig.savefig("asym_refraction_field.png", dpi=150)
    plt.close(fig)
    logger.info("saved asym_refraction_field.png")

    intensity = np.mean(late.astype(np.float64) ** 2, axis=0)
    fig, ax = plt.subplots(figsize=(12, 5.4))
    im = ax.imshow(
        np.sqrt(intensity / intensity.max()).T,
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=1,
        extent=extent,
        aspect="equal",
        interpolation="bilinear",
    )
    for z in (z_top, z_bot):
        ax.axhline(z, color="w", lw=0.9)
    rays(ax, "0.85", {+1.0: "lime", -1.0: "cyan"})
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("z (μm)")
    ax.set_title("Time-averaged field: one beam refracts negatively, the mirrored one positively")
    fig.colorbar(im, ax=ax, ticks=[0, 0.5, 1], shrink=0.85, label="normalized field magnitude")
    fig.tight_layout()
    fig.savefig("asym_refraction_intensity.png", dpi=150)
    plt.close(fig)
    logger.info("saved asym_refraction_intensity.png")


def main() -> None:
    plot_tensor_components()

    eps2 = eps_xz_block(OMEGA)
    logger.info(f"lab-frame eps xz-block at drive:\n{np.array_str(eps2, precision=3, suppress_small=True)}")

    angles = {sgn * THETA_DEG: analytic_energy_angle(sgn * THETA_DEG) for sgn in (+1.0, -1.0)}
    for theta, ts in angles.items():
        kind = "NEGATIVE" if ts * theta < 0 else "positive"
        logger.info(f"analytic: incidence {theta:+.0f} deg -> energy {ts:+.1f} deg ({kind} refraction)")

    frames = run()

    intensity = np.mean(frames[frames.shape[0] // 2 :].astype(np.float64) ** 2, axis=0)
    z_src_um = SOURCE_Z * RESOLUTION * 1e6
    z_top_um = SLAB_TOP_Z * RESOLUTION * 1e6
    for sgn, src_x in ((+1.0, SRC_A_X), (-1.0, SRC_B_X)):
        hit_x_um = src_x * 1e6 + sgn * (z_src_um - z_top_um)
        measured = measure_corridor_angle(intensity, hit_x_um)
        analytic = angles[sgn * THETA_DEG]
        logger.info(
            f"measured: incidence {sgn * THETA_DEG:+.0f} deg -> energy {measured:+.1f} deg "
            f"(analytic {analytic:+.1f} deg, difference {abs(measured - analytic):.1f} deg)"
        )

    make_plots(frames, angles)


if __name__ == "__main__":
    main()
