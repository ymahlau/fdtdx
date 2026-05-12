"""
1D Photonic Crystal Band Structure — Bloch-FDTD Example

Computes the photonic band structure of a 1D photonic crystal by running
one FDTD simulation per Bloch k-point in the first Brillouin zone:

  1. For each k_z ∈ (0, π/a]:
       - Set up a single unit cell with Bloch BCs in z, periodic in x/y.
       - Inject a broadband Gaussian pulse to excite all modes at once.
       - Wait for the non-resonant (non-propagating) transient to die away;
         only long-lived resonant modes remain ringing in the field.
       - Fourier-transform the recorded E-field time series → resonance peaks.
  2. Collect (k, ω) pairs across all k-points and plot the band diagram.

Structure:
  - Two isotropic materials in alternating layers, 50% duty cycle.
  - Background (air): ε₁ = 1.0
  - High-index (Si):  ε₂ = 12.0
  - Lattice constant: a = 500 nm

Dependencies (beyond fdtdx): numpy, scipy, matplotlib.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # non-interactive; remove if running in a notebook
import matplotlib.pyplot as plt
from loguru import logger
from scipy.signal import find_peaks
import fdtdx

# ── Physical parameters ────────────────────────────────────────────────────────
A = 500e-9        # Lattice constant / unit-cell length in z [m]
EPS_LO = 1.0      # Air permittivity
EPS_HI = 12.0     # Silicon permittivity
DUTY_CYCLE = 0.5  # Fraction of unit cell occupied by the high-index material

# ── Numerical parameters ───────────────────────────────────────────────────────
# Resolution chosen so that DUTY_CYCLE*A is an exact integer number of cells:
#   DUTY_CYCLE * A / RESOLUTION = 0.5 * 500e-9 / 25e-9 = 10 cells  ✓
RESOLUTION = 25e-9   # [m]  → 20 cells per unit cell

# Minimal transverse domain (periodic BCs make the lattice effectively infinite in x/y)
N_TRANSVERSE = 3
TRANSVERSE = N_TRANSVERSE * RESOLUTION  # 75 nm

# ── Source: broadband Gaussian pulse ──────────────────────────────────────────
# The carrier is centred at normalised frequency a/λ = 0.5 (first BZ edge).
# The spectral width is chosen wide enough to cover [0, FREQ_MAX] in a/λ.
# GaussianPulseProfile computes σ_t = 1 / (2π · c / SPECTRAL_WL),
# so smaller SPECTRAL_WL → broader bandwidth.
CENTER_WL = 2.0 * A   # carrier λ → normalised freq a/λ = A/CENTER_WL = 0.5
SPECTRAL_WL = 1.5 * A # σ_f = c / SPECTRAL_WL ≈ 0.67 · c/a; pulse FWHM ~ 1.6 in a/λ

# ── Time and k-sampling ────────────────────────────────────────────────────────
# 500 fs gives ~200 optical periods at center wavelength.
# Frequency resolution: Δ(a/λ) = (a/c) / T_sim ≈ 0.0083  (resolves individual bands).
# The Gaussian pulse is essentially over by ~6·σ_t ≈ 5 fs, leaving ~195 fs of
# clean ringdown for the resonant modes to be resolved by the FFT.
SIM_TIME = 500e-15  # [s]

NUM_K = 15
N_SOURCES = 5   # random z positions; more sources → better mode coverage
# Exclude exact k = 0 (Γ point) so that Bloch BCs always require complex fields.
# Scan from k·a/2π = 1/(2·NUM_K) ≈ 0.033 to 0.5 (X point / BZ edge).
K_NORMS = np.linspace(0.0, 0.5, NUM_K + 1)[1:]  # shape (NUM_K,)

FREQ_MAX = 0.8  # upper limit of the plotted normalised frequency range

C = fdtdx.constants.c  # speed of light [m/s]


# ── Core simulation ────────────────────────────────────────────────────────────

def run_bloch_k(k_norm: float, key: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    """Run one FDTD simulation for a given Bloch wavevector.

    Args:
        k_norm: Normalised Bloch wavevector k_z · a / (2π) ∈ (0, 0.5].
        key:    JAX PRNG key.

    Returns:
        freqs_norm: Array of normalised frequencies f · a / c = a / λ.
        spectrum:   Corresponding |FFT(Ex(t))| spectral amplitudes.
    """
    k_z = k_norm * 2.0 * np.pi / A  # [rad/m]

    # Forward-only simulation; complex fields auto-promoted for non-zero Bloch k.
    config = fdtdx.SimulationConfig(
        time=SIM_TIME,
        resolution=RESOLUTION,
        dtype=jnp.float32,
        courant_factor=0.99,
    )
    dt = config.time_step_duration

    constraints, object_list = [], []

    # ── Simulation volume: exactly one unit cell ───────────────────────────────
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(TRANSVERSE, TRANSVERSE, A),
        material=fdtdx.Material(permittivity=EPS_LO),  # background = air
    )
    object_list.append(volume)

    # ── High-index slab: bottom DUTY_CYCLE fraction of the unit cell ──────────
    slab = fdtdx.UniformMaterialObject(
        partial_real_shape=(TRANSVERSE, TRANSVERSE, DUTY_CYCLE * A),
        partial_grid_shape=(None, None, None),
        material=fdtdx.Material(permittivity=EPS_HI),
    )
    # Align bottom face of slab with bottom face of volume.
    constraints.append(
        slab.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, -1),    # bottom face of slab …
            other_positions=(0, 0, -1),  # … to bottom face of volume
        )
    )
    object_list.append(slab)

    # ── Boundaries ─────────────────────────────────────────────────────────────
    # x / y: standard periodic (k = 0) — structure is uniform transversely.
    # z min / max: Bloch with the current k_z value.
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        boundary_type="periodic",
        override_types={"min_z": "bloch", "max_z": "bloch"},
        bloch_vector=(0.0, 0.0, k_z),
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    object_list.extend(list(bound_dict.values()))

    # ── Broadband sources at random z positions ────────────────────────────────
    # A single source placed at a symmetry point (z = a/4, a/2, or 3a/4) has
    # zero overlap with modes of the opposite parity and cannot excite them.
    # Using several sources at independently random z positions guarantees that
    # every mode — regardless of parity or node structure — is hit by at least
    # one source with non-zero coupling.  The random phases additionally
    # decorrelate the sources so their contributions add incoherently across
    # the spectrum.  A fixed seed makes the run reproducible.
    _rng = np.random.default_rng(seed=12345)
    _z_positions = _rng.uniform(-1.0, 1.0, size=N_SOURCES)   # other_positions range
    _phases      = _rng.uniform(0.0, 2 * np.pi, size=N_SOURCES)

    for _j in range(N_SOURCES):
        _src = fdtdx.UniformPlaneSource(
            partial_grid_shape=(None, None, 1),
            partial_real_shape=(TRANSVERSE, TRANSVERSE, None),
            fixed_E_polarization_vector=(1, 0, 0),
            wave_character=fdtdx.WaveCharacter(
                wavelength=CENTER_WL, phase_shift=float(_phases[_j])
            ),
            amplitude=1.0 / N_SOURCES,   # keep total injected energy constant
            direction="+",
            temporal_profile=fdtdx.GaussianPulseProfile(
                center_wave=fdtdx.WaveCharacter(
                    wavelength=CENTER_WL, phase_shift=float(_phases[_j])
                ),
                spectral_width=fdtdx.WaveCharacter(wavelength=SPECTRAL_WL),
            ),
        )
        constraints.append(
            _src.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 0),
                other_positions=(0, 0, float(_z_positions[_j])),
            )
        )
        object_list.append(_src)

    # ── Column field detector spanning the full z extent ───────────────────────
    # A 1×1×Nz detector records Ex(z, t) at every cell along the propagation
    # axis.  After the simulation we FFT over time at each z cell and sum the
    # spectral power over z.  Because every mode has non-zero amplitude at some
    # z, the summed spectrum cannot miss a mode due to node placement — unlike a
    # point detector, which is blind to any mode with a node at its location.
    # dtype=complex64 because Bloch modes are complex for k_z > 0.
    field_det = fdtdx.FieldDetector(
        name="field_col",
        partial_grid_shape=(None, None, None),
        partial_real_shape=(None, None, None),
        components=("Ex",),
        reduce_volume=False,
        dtype=jnp.complex64,
        switch=fdtdx.OnOffSwitch(interval=1),
        plot=False,
    )
    constraints.extend(field_det.same_position_and_size(volume))
    object_list.append(field_det)

    # ── Resolve geometry and initialise ───────────────────────────────────────
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=constraints,
        key=subkey,
    )
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, subkey)

    # ── Run forward FDTD ───────────────────────────────────────────────────────
    # Note: each k-point triggers a JAX recompilation because the Bloch vector
    # is a frozen (static) field.  For production use consider making k_z a
    # traced JAX array and restructuring accordingly.
    _, arrays = fdtdx.run_fdtd(
        arrays=arrays,
        objects=objects,
        config=config,
        key=subkey,
    )

    # ── Fourier transform ──────────────────────────────────────────────────────
    # state["fields"] shape: (N_recorded, 1, 1, 1, Nz)
    #   axis 0: time steps recorded
    #   axis 1: field component (just Ex → size 1)
    #   axes 2,3: transverse x/y (size 1 each)
    #   axis 4: z cells across the unit cell
    #
    # Steps:
    #   1. Extract real part — valid for lossless media where ω is real.
    #   2. Apply a Hann window in time to suppress spectral leakage.
    #   3. FFT over the time axis at every z cell simultaneously.
    #   4. Sum |FFT|² over z → total spectral energy.  Every mode contributes
    #      at some z cell; summing ensures none are suppressed by node placement.
    # state shape: (N_recorded, 1, Nx, Ny, Nz)
    # The field is uniform in x/y (periodic BCs), so average over those axes.
    ex_zt = np.real(np.array(
        arrays.detector_states["field_col"]["fields"][:, 0, :, :, :]
    )).mean(axis=(1, 2))  # shape (N_recorded, Nz)
    N_t = ex_zt.shape[0]
    window = np.hanning(N_t)[:, None]          # broadcast over z
    spectra_z = np.abs(np.fft.rfft(ex_zt * window, axis=0)) ** 2  # (N_freq, Nz)
    spectrum = spectra_z.sum(axis=1)           # (N_freq,) — summed spectral power
    freqs_norm = np.fft.rfftfreq(N_t, d=dt) * A / C  # a/λ = f·a/c

    return freqs_norm, spectrum


# ── Analytic band structure ────────────────────────────────────────────────────

def analytic_bands(
    eps_hi: float,
    eps_lo: float,
    duty_cycle: float,
    freq_max: float,
    num_points: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 1D photonic crystal band structure via the transfer matrix method.

    For a two-layer unit cell the Bloch condition gives the dispersion relation:

        cos(k_z · a) = cos(φ₁)·cos(φ₂) − ½(n₁/n₂ + n₂/n₁)·sin(φ₁)·sin(φ₂)

    where φⱼ = 2π · nⱼ · (a/λ) · dⱼ/a is the phase accumulated per layer.
    Solutions exist only where the right-hand side lies in [−1, 1]; outside that
    range the frequency falls in a band gap.

    Args:
        eps_hi:     Permittivity of the high-index layer.
        eps_lo:     Permittivity of the low-index layer.
        duty_cycle: Fraction of the unit cell occupied by the high-index layer.
        freq_max:   Upper normalised frequency limit (a/λ) for the sweep.
        num_points: Number of frequency samples used to trace the bands.

    Returns:
        k_norms: Normalised Bloch wavevectors k_z·a/(2π) ∈ [0, 0.5].
        f_norms: Corresponding normalised frequencies a/λ.
    """
    n1 = np.sqrt(eps_hi)
    n2 = np.sqrt(eps_lo)
    d1 = duty_cycle
    d2 = 1.0 - duty_cycle

    f_norms = np.linspace(0.0, freq_max, num_points)

    # Phase per layer: φⱼ = 2π · nⱼ · f_norm · (dⱼ/a)
    phi1 = 2.0 * np.pi * n1 * f_norms * d1
    phi2 = 2.0 * np.pi * n2 * f_norms * d2

    rhs = (
        np.cos(phi1) * np.cos(phi2)
        - 0.5 * (n1 / n2 + n2 / n1) * np.sin(phi1) * np.sin(phi2)
    )

    # Propagating modes exist only where |RHS| ≤ 1
    propagating = np.abs(rhs) <= 1.0
    k_norms = np.where(propagating, np.arccos(np.clip(rhs, -1.0, 1.0)) / (2.0 * np.pi), np.nan)

    return k_norms[propagating], f_norms[propagating]


# ── Peak detection ─────────────────────────────────────────────────────────────

def find_resonances(
    freqs_norm: np.ndarray,
    spectrum: np.ndarray,
    freq_max: float = FREQ_MAX,
    min_height: float = 0.05,
    min_prominence: float = 0.03,
) -> np.ndarray:
    """Return normalised frequencies of spectral peaks above threshold.

    Args:
        freqs_norm:     Normalised frequency axis (a/λ).
        spectrum:       Spectral amplitude |FFT(Ex)|.
        freq_max:       Upper frequency cutoff (normalised).
        min_height:     Minimum peak height relative to spectrum maximum.
        min_prominence: Minimum peak prominence relative to spectrum maximum.

    Returns:
        Array of normalised resonance frequencies.
    """
    mask = (freqs_norm > 0) & (freqs_norm <= freq_max)
    f, s = freqs_norm[mask], spectrum[mask]
    if s.max() < 1e-30:
        return np.array([])
    s_norm = s / s.max()
    peaks, _ = find_peaks(s_norm, height=min_height, prominence=min_prominence)
    return f[peaks]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    exp_logger = fdtdx.Logger(experiment_name="bloch_band_structure", name=None)
    key = jax.random.PRNGKey(0)

    logger.info("1D Photonic Crystal Band Structure")
    logger.info(f"  Lattice:    a = {A*1e9:.0f} nm")
    logger.info(f"  Materials:  eps1 = {EPS_LO:.1f} (air),  eps2 = {EPS_HI:.1f} (Si),  "
                f"{DUTY_CYCLE*100:.0f}% duty cycle")
    logger.info(f"  Grid:       {RESOLUTION*1e9:.0f} nm/cell  ({int(A/RESOLUTION)} cells/unit cell)")
    logger.info(f"  Time:       {SIM_TIME*1e15:.0f} fs")
    logger.info(f"  k-points:   {NUM_K}  (k*a/2pi in [{K_NORMS[0]:.3f}, {K_NORMS[-1]:.3f}])")

    band_data: list[tuple[float, np.ndarray]] = []
    sweep_task = exp_logger.progress.add_task("Bloch k-sweep", total=NUM_K)

    for i, k_norm in enumerate(K_NORMS):
        key, subkey = jax.random.split(key)
        freqs, spec = run_bloch_k(k_norm, subkey)
        peaks = find_resonances(freqs, spec)
        band_data.append((k_norm, peaks))

        label = ", ".join(f"{p:.3f}" for p in peaks[:4]) or "(none found)"
        logger.info(f"  k {i+1:2d}/{NUM_K}  k*a/2pi = {k_norm:.3f}  ->  "
                    f"{len(peaks)} mode(s): {label}")

        exp_logger.write({
            "k_index": i,
            "k_norm": float(k_norm),
            "num_modes": len(peaks),
        })

        exp_logger.progress.update(sweep_task, advance=1)

    # ── Save raw band data ─────────────────────────────────────────────────────
    results_dir = exp_logger.cwd / "results"
    results_dir.mkdir(exist_ok=True)

    all_k = np.concatenate([[k] * len(p) for k, p in band_data if len(p)])
    all_f = np.concatenate([p for _, p in band_data if len(p)])
    ak, af = analytic_bands(EPS_HI, EPS_LO, DUTY_CYCLE, FREQ_MAX)
    np.savez(
        results_dir / "band_data.npz",
        k_norms=np.array([k for k, _ in band_data]),
        all_k=all_k,
        all_f=all_f,
        analytic_k=ak,
        analytic_f=af,
    )
    logger.info(f"Band data saved to {results_dir / 'band_data.npz'}")

    # ── Band diagram plot ──────────────────────────────────────────────────────
    ak, af = analytic_bands(EPS_HI, EPS_LO, DUTY_CYCLE, FREQ_MAX)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(ak, af, c="lightgray", s=4, lw=0, zorder=1, label="analytic (TMM)")
    for k_norm, peaks in band_data:
        if len(peaks):
            ax.scatter(
                [k_norm] * len(peaks), peaks,
                c="royalblue", s=18, alpha=0.85, lw=0, zorder=3,
            )
    ax.scatter([], [], c="royalblue", s=18, label="FDTD")  # legend proxy
    ax.axvline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6, label="BZ edge")
    ax.set_xlabel(r"Bloch wavevector $k_z a\,/\,2\pi$")
    ax.set_ylabel(r"Normalised frequency $a\,/\,\lambda$")
    ax.set_title(
        "1D Photonic Crystal Band Structure\n"
        rf"$\varepsilon_1={EPS_LO:.0f}$, $\varepsilon_2={EPS_HI:.0f}$, "
        rf"{DUTY_CYCLE*100:.0f}% duty cycle"
    )
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, FREQ_MAX)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = exp_logger.cwd / "band_diagram.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Band diagram saved to {plot_path}")


if __name__ == "__main__":
    main()
