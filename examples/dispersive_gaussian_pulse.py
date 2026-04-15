"""Gaussian pulse broadening through uniform Lorentz-dispersive silicon.

A short Gaussian pulse propagates along +z inside a silicon domain that fills
the entire simulation volume. Two FieldDetectors (A upstream, B downstream)
record E_x as the pulse passes. The dispersive simulation is compared to a
reference run in which silicon is replaced with a non-dispersive material
whose permittivity matches Re(eps(omega_c)) of the Lorentz model at the
carrier frequency — phase velocity is identical, so the non-dispersive pulse
should propagate without changing its envelope.

Uniform silicon throughout avoids Fresnel reflections at a slab/vacuum
interface, so the only physics shaping the downstream pulse is
group-velocity dispersion from the Lorentz pole. The script:

    1. Runs two simulations with identical sources and grid — one with the
       Lorentz-dispersive Si background, one with a matched non-dispersive
       Si background.
    2. Records the E_x time trace at detectors A (near source) and B (far
       downstream) in each run.
    3. Measures the FWHM temporal width of |E|^2 at each detector.
    4. Uses the measured FWHM at detector A as the initial pulse width
       tau_0 and computes the analytic broadening tau(L)/tau_0
       = sqrt(1 + (L * beta_2 / tau_0^2)^2), with L = (DET_B - DET_A) * dx
       and beta_2 from the Lorentz model's d^2k/d omega^2.
    5. Plots dispersive vs non-dispersive intensity envelopes at both
       detectors and logs the measured vs analytic broadening factor.

The Lorentz parameters are taken from @bruxillensis single-pole fit for
Si (Silicon, Dispersive & Lossless): eps_inf = 7.98737, delta_eps = 3.68799,
omega_0 = 3.93282e15 rad/s, linewidth = 1e8 rad/s. This fit matches silicon's
real refractive index (n ~ 3.48 at 1550 nm) but places the single pole far
above the operating frequency, so beta_2 is small and a short pulse over a
compact propagation distance is needed to produce observable broadening.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

import fdtdx
from fdtdx.constants import c as c0

WAVELENGTH = 1.55e-6
OMEGA_C = 2.0 * np.pi * c0 / WAVELENGTH

EPS_INF = 7.98737492
OMEGA_0 = 3.93282466e15
DELTA_EPS = 3.68799143
GAMMA = 1e8

SIGMA_T = 4.0e-15
SIGMA_F = 1.0 / (2.0 * np.pi * SIGMA_T)

RESOLUTION = 20e-9
PML_CELLS = 10
DOMAIN_XY = 3 * RESOLUTION
DOMAIN_Z = 18e-6
Z_CELLS = int(round(DOMAIN_Z / RESOLUTION))

SOURCE_Z = PML_CELLS + 2
DET_A_Z = 100
DET_B_Z = 800
PROPAGATION_LENGTH = (DET_B_Z - DET_A_Z) * RESOLUTION

SIM_TIME = 350e-15


def silicon_material() -> fdtdx.Material:
    """Silicon with the single-Lorentz dispersion fit."""
    return fdtdx.Material(
        permittivity=EPS_INF,
        dispersion=fdtdx.DispersionModel(
            poles=(
                fdtdx.LorentzPole(
                    resonance_frequency=OMEGA_0,
                    damping=GAMMA,
                    delta_epsilon=DELTA_EPS,
                ),
            )
        ),
    )


def nondispersive_silicon_material() -> fdtdx.Material:
    """Frequency-independent silicon whose permittivity matches
    Re(eps(omega_c)) of the Lorentz model. Matches phase velocity at the
    carrier frequency exactly — the reference run isolates dispersion."""
    model = silicon_material().dispersion
    assert model is not None
    eps_c = EPS_INF + complex(model.susceptibility(OMEGA_C))
    return fdtdx.Material(permittivity=float(eps_c.real))


def build_scene(dispersive: bool):
    config = fdtdx.SimulationConfig(
        resolution=RESOLUTION,
        time=SIM_TIME,
        dtype=jnp.float32,
        courant_factor=0.99,
    )
    objects: list = []
    constraints: list = []

    background = silicon_material() if dispersive else nondispersive_silicon_material()
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(DOMAIN_XY, DOMAIN_XY, DOMAIN_Z),
        material=background,
    )
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

    center_wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH)
    spectral_width = fdtdx.WaveCharacter(frequency=SIGMA_F)

    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=center_wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
        temporal_profile=fdtdx.GaussianPulseProfile(
            spectral_width=spectral_width,
            center_wave=center_wave,
        ),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(SOURCE_Z,)),
        ]
    )
    objects.append(source)

    for name, z_idx in (("pulse_trace_A", DET_A_Z), ("pulse_trace_B", DET_B_Z)):
        detector = fdtdx.FieldDetector(
            name=name,
            partial_grid_shape=(None, None, 1),
            components=("Ex",),
            reduce_volume=True,
            plot=False,
        )
        constraints.extend(
            [
                detector.same_size(volume, axes=(0, 1)),
                detector.place_at_center(volume, axes=(0, 1)),
                detector.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(z_idx,)),
            ]
        )
        objects.append(detector)

    return objects, constraints, config


def run(objects, constraints, config) -> tuple[np.ndarray, np.ndarray]:
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    trace_a = np.asarray(arrays.detector_states["pulse_trace_A"]["fields"][:, 0])
    trace_b = np.asarray(arrays.detector_states["pulse_trace_B"]["fields"][:, 0])
    return trace_a, trace_b


def pulse_width(trace: np.ndarray, dt: float) -> tuple[float, float]:
    """Return (t_peak, sigma) of the Gaussian intensity envelope inferred from
    the FWHM of the smoothed |E(t)|^2 signal.

    FWHM avoids contamination from pulse tails / residual ringing that would
    inflate a naive second-moment computation. For a Gaussian,
    FWHM = 2*sqrt(2*ln 2) * sigma.
    """
    intensity = trace.astype(np.float64) ** 2
    carrier_period = WAVELENGTH / c0
    w = max(int(round(2.0 * carrier_period / dt)) | 1, 5)
    kernel = np.ones(w) / w
    smooth = np.convolve(intensity, kernel, mode="same")

    peak_idx = int(np.argmax(smooth))
    peak_val = float(smooth[peak_idx])
    half = 0.5 * peak_val

    left = peak_idx
    while left > 0 and smooth[left] > half:
        left -= 1
    right = peak_idx
    while right < smooth.size - 1 and smooth[right] > half:
        right += 1

    def _interp(a: int, b: int) -> float:
        if smooth[a] == smooth[b]:
            return float(a)
        return float(a + (half - smooth[a]) / (smooth[b] - smooth[a]))

    t_left = _interp(left, left + 1)
    t_right = _interp(right - 1, right)
    fwhm = (t_right - t_left) * dt
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return peak_idx * dt, sigma


def analytic_beta2(model: fdtdx.DispersionModel, omega: float, eps_inf: float = 1.0) -> float:
    """Numerically compute beta_2 = d^2 k / d omega^2 at omega, using central
    differences of Re(k(omega)) = Re(omega * sqrt(eps_inf + chi(omega)) / c0)."""
    h = 1e-3 * omega

    def k_of(w: float) -> float:
        eps = eps_inf + model.susceptibility(w)
        n = np.sqrt(eps)
        return float(np.real(w * n / c0))

    return (k_of(omega + h) - 2.0 * k_of(omega) + k_of(omega - h)) / (h * h)


def main():
    exp_logger = fdtdx.Logger(experiment_name="dispersive_gaussian_pulse", name=None)

    material = silicon_material()
    assert material.dispersion is not None
    model = material.dispersion
    eps_c = EPS_INF + complex(model.susceptibility(OMEGA_C))
    n_c = float(np.sqrt(eps_c).real)
    beta_2 = analytic_beta2(model, OMEGA_C, eps_inf=EPS_INF)

    logger.info("Gaussian pulse dispersion example (uniform silicon)")
    logger.info(f"  wavelength       : {WAVELENGTH * 1e9:.1f} nm")
    logger.info(f"  carrier omega    : {OMEGA_C:.3e} rad/s")
    logger.info(f"  resonance omega  : {OMEGA_0:.3e} rad/s  ({OMEGA_0 / OMEGA_C:.2f} * omega_c)")
    logger.info(f"  pulse sigma_t    : {SIGMA_T * 1e15:.2f} fs")
    logger.info(f"  propagation L    : {PROPAGATION_LENGTH * 1e6:.2f} um (det A -> det B)")
    logger.info(f"  eps(omega_c)     : {eps_c.real:.4f} + {eps_c.imag:.2e}j")
    logger.info(f"  n(omega_c)       : {n_c:.4f}")
    logger.info(f"  beta_2           : {beta_2:.3e} s^2/m")

    logger.info("Running non-dispersive reference simulation...")
    nd_obj, nd_con, nd_cfg = build_scene(dispersive=False)
    nd_trace_a, nd_trace_b = run(nd_obj, nd_con, nd_cfg)
    dt = nd_cfg.time_step_duration

    logger.info("Running dispersive simulation...")
    d_obj, d_con, d_cfg = build_scene(dispersive=True)
    d_trace_a, d_trace_b = run(d_obj, d_con, d_cfg)

    t_a_nd, sigma_a_nd = pulse_width(nd_trace_a, dt)
    t_b_nd, sigma_b_nd = pulse_width(nd_trace_b, dt)
    t_a_d, sigma_a_d = pulse_width(d_trace_a, dt)
    t_b_d, sigma_b_d = pulse_width(d_trace_b, dt)

    tau_0 = sigma_a_d
    L_D = tau_0**2 / abs(beta_2)
    analytic_ratio = float(np.sqrt(1.0 + (PROPAGATION_LENGTH * beta_2 / tau_0**2) ** 2))
    measured_ratio = sigma_b_d / sigma_a_d if sigma_a_d > 0 else float("nan")
    rel_err = abs(measured_ratio - analytic_ratio) / analytic_ratio
    nondisp_ratio = sigma_b_nd / sigma_a_nd if sigma_a_nd > 0 else float("nan")

    logger.info("Results:")
    logger.info(f"  dispersion len   : {L_D * 1e6:.3f} um  (L / L_D = {PROPAGATION_LENGTH / L_D:.3f})")
    logger.info(f"  non-disp  sigma_A: {sigma_a_nd * 1e15:.3f} fs  (t = {t_a_nd * 1e15:.2f} fs)")
    logger.info(f"  non-disp  sigma_B: {sigma_b_nd * 1e15:.3f} fs  (t = {t_b_nd * 1e15:.2f} fs)")
    logger.info(f"  dispersive sigmaA: {sigma_a_d * 1e15:.3f} fs  (t = {t_a_d * 1e15:.2f} fs)")
    logger.info(f"  dispersive sigmaB: {sigma_b_d * 1e15:.3f} fs  (t = {t_b_d * 1e15:.2f} fs)")
    logger.info(f"  measured t/t0    : {measured_ratio:.4f}")
    logger.info(f"  analytic t/t0    : {analytic_ratio:.4f}")
    logger.info(f"  relative error   : {rel_err * 100:.2f} %")
    logger.info(f"  non-disp  t/t0   : {nondisp_ratio:.4f}  (sanity; should be ~1)")

    exp_logger.write(
        {
            "wavelength_m": WAVELENGTH,
            "omega_c": OMEGA_C,
            "omega_0": OMEGA_0,
            "eps_inf": EPS_INF,
            "delta_eps": DELTA_EPS,
            "gamma": GAMMA,
            "sigma_t": SIGMA_T,
            "propagation_length_m": PROPAGATION_LENGTH,
            "eps_real": float(eps_c.real),
            "eps_imag": float(eps_c.imag),
            "n_c": n_c,
            "beta_2": float(beta_2),
            "tau_0_s": float(tau_0),
            "L_D_m": float(L_D),
            "analytic_ratio": analytic_ratio,
            "sigma_a_nd_s": float(sigma_a_nd),
            "sigma_b_nd_s": float(sigma_b_nd),
            "sigma_a_d_s": float(sigma_a_d),
            "sigma_b_d_s": float(sigma_b_d),
            "t_a_nd_s": float(t_a_nd),
            "t_b_nd_s": float(t_b_nd),
            "t_a_d_s": float(t_a_d),
            "t_b_d_s": float(t_b_d),
            "measured_ratio": float(measured_ratio),
            "nondisp_ratio": float(nondisp_ratio),
            "rel_err": float(rel_err),
        }
    )

    results_dir = exp_logger.cwd / "results"
    results_dir.mkdir(exist_ok=True)
    t_ax = np.arange(nd_trace_a.size) * dt
    np.savez(
        results_dir / "pulse_traces.npz",
        t=t_ax,
        nondisp_A_Ex=nd_trace_a,
        nondisp_B_Ex=nd_trace_b,
        disp_A_Ex=d_trace_a,
        disp_B_Ex=d_trace_b,
        dt=dt,
    )
    logger.info(f"Pulse traces saved to {results_dir / 'pulse_traces.npz'}")

    t_ax_fs = t_ax * 1e15

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax_a.plot(t_ax_fs, nd_trace_a**2, color="tab:blue", label=f"non-disp (sigma={sigma_a_nd * 1e15:.2f} fs)")
    ax_a.plot(t_ax_fs, d_trace_a**2, color="tab:red", label=f"dispersive (sigma={sigma_a_d * 1e15:.2f} fs)")
    ax_a.set_ylabel("|E_x|^2  [a.u.]")
    ax_a.set_title(f"Detector A (z cell = {DET_A_Z}) — upstream reference")
    ax_a.legend(loc="upper right")
    ax_a.grid(True, alpha=0.3)

    ax_b.plot(t_ax_fs, nd_trace_b**2, color="tab:blue", label=f"non-disp (sigma={sigma_b_nd * 1e15:.2f} fs)")
    ax_b.plot(t_ax_fs, d_trace_b**2, color="tab:red", label=f"dispersive (sigma={sigma_b_d * 1e15:.2f} fs)")
    ax_b.set_xlabel("time [fs]")
    ax_b.set_ylabel("|E_x|^2  [a.u.]")
    ax_b.set_title(
        f"Detector B (z cell = {DET_B_Z}) — after {PROPAGATION_LENGTH * 1e6:.1f} um propagation. "
        f"measured t/t0 = {measured_ratio:.3f}, analytic = {analytic_ratio:.3f} (err {rel_err * 100:.1f}%)"
    )
    ax_b.legend(loc="upper right")
    ax_b.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = exp_logger.cwd / "dispersive_gaussian_pulse.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
