# Imports: standard libraries and required packages
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
import fdtdx
from fdtdx.core.physics.modes import compute_mode


def compute_modes_and_neff_complex(arrays, object):
    """
    Compute mode profiles and complex effective refractive index.
    
    Args:
        arrays: Array container with material properties
        object: Source/detector object defining the mode
        
    Returns:
        mode_E: Electric field mode profile
        mode_H: Magnetic field mode profile
        neff_complex: Complex effective refractive index
    """
    grid_slice = object.grid_slice
    if arrays.electric_conductivity is not None:
        inv_permittivities = jax.lax.stop_gradient(
            1./(
                1./arrays.inv_permittivities + \
                1j * arrays.electric_conductivity / object._config.resolution / (2 * jnp.pi * object.wave_characters[0].get_frequency() * fdtdx.constants.eps0)
            )
        )
    else:
        inv_permittivities = jax.lax.stop_gradient(arrays.inv_permittivities)
    inv_permeabilities = jax.lax.stop_gradient(arrays.inv_permeabilities)
    inv_permittivity_slice = inv_permittivities[:, *grid_slice]
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        inv_permeability_slice = inv_permeabilities[:, *grid_slice]
    else:
        inv_permeability_slice = inv_permeabilities
    
    mode_E, mode_H, neff_complex = compute_mode(
        frequency          = object.wave_characters[0].get_frequency(),
        inv_permittivities = inv_permittivity_slice,
        inv_permeabilities = inv_permeability_slice,
        resolution         = object._config.resolution,
        direction          = object.direction,
        mode_index         = object.mode_index,
        filter_pol         = object.filter_pol,
    )
    
    return mode_E, mode_H, neff_complex


def overlap(E0, E1, H0, H1, propagation_axis):
    """
    Calculate overlap integral between two electromagnetic field distributions.
    
    Args:
        E0: Electric field of first mode
        E1: Electric field of second mode
        H0: Magnetic field of first mode
        H1: Magnetic field of second mode
        propagation_axis: Axis of propagation (0 for x, 1 for y, 2 for z)
        
    Returns:
        alpha_coeff: Overlap coefficient
    """
    E0_cross_H1_star = jnp.cross(
        E0,
        jnp.conj(H1),
        axis=0,
    )

    E1_star_cross_H0 = jnp.cross(
        jnp.conj(E1),
        H0,
        axis=0,
    )

    alpha_coeff = jnp.sum((E0_cross_H1_star + E1_star_cross_H0)[propagation_axis])
    alpha_coeff = alpha_coeff / 4.0

    return alpha_coeff


def make_waveguide_simulation(wavelength, mesh, width, height, absorber_thickness, time, key):
    """
    Create a waveguide simulation with PML boundaries.
    
    Args:
        wavelength: Wavelength of light in meters
        mesh: Resolution multiplier (wavelength/mesh)
        width: Width of the waveguide in meters
        height: Height of the waveguide in meters
        absorber_thickness: Thickness of absorber layers in meters
        time: Total simulation time in seconds
        key: JAX random key
        
    Returns:
        arrays: Array container with simulation fields
        new_objects: Simulation objects after applying parameters
        config: Simulation configuration
    """
    # Calculate simulation parameters
    frequency = fdtdx.constants.c / wavelength
    resolution = float(wavelength/mesh)
    courant_factor = float(0.99)

    # List all objects
    object_list = []

    # PML boundary layer configuration
    num_layers_x = 2*mesh
    num_layers_y = mesh
    num_layers_z = mesh

    # PML parameters
    kappa_start = 1.0
    kappa_end = 3.0
    kappa_order = 4
    alpha_start = 0.0
    alpha_end = 0.0
    alpha_order = 1
    sigma_start = 0.0
    sigma_end = -(4+1)*np.log(1e-6)/(2*(fdtdx.constants.eta0/1.44)*(num_layers_x*resolution))
    sigma_order = 4

    # Configure PML boundaries on all sides
    bound_cfg = fdtdx.BoundaryConfig(
        boundary_type_minx="pml",
        boundary_type_maxx="pml",
        boundary_type_miny="pml",
        boundary_type_maxy="pml",
        boundary_type_minz="pml",
        boundary_type_maxz="pml",
        thickness_grid_minx=num_layers_x,
        thickness_grid_maxx=num_layers_x,
        thickness_grid_miny=num_layers_y,
        thickness_grid_maxy=num_layers_y,
        thickness_grid_minz=num_layers_z,
        thickness_grid_maxz=num_layers_z,
        kappa_start_minx=kappa_start,
        kappa_end_minx=kappa_end,
        kappa_start_maxx=kappa_start,
        kappa_end_maxx=kappa_end,
        kappa_start_miny=kappa_start,
        kappa_end_miny=kappa_end,
        kappa_start_maxy=kappa_start,
        kappa_end_maxy=kappa_end,
        kappa_start_minz=kappa_start,
        kappa_end_minz=kappa_end,
        kappa_start_maxz=kappa_start,
        kappa_end_maxz=kappa_end,
        kappa_order_minx=kappa_order,
        kappa_order_maxx=kappa_order,
        kappa_order_miny=kappa_order,
        kappa_order_maxy=kappa_order,
        kappa_order_minz=kappa_order,
        kappa_order_maxz=kappa_order,
        alpha_start_minx=alpha_start,
        alpha_end_minx=alpha_end,
        alpha_start_maxx=alpha_start,
        alpha_end_maxx=alpha_end,
        alpha_start_miny=alpha_start,
        alpha_end_miny=alpha_end,
        alpha_start_maxy=alpha_start,
        alpha_end_maxy=alpha_end,
        alpha_start_minz=alpha_start,
        alpha_end_minz=alpha_end,
        alpha_start_maxz=alpha_start,
        alpha_end_maxz=alpha_end,
        alpha_order_minx=alpha_order,
        alpha_order_maxx=alpha_order,
        alpha_order_miny=alpha_order,
        alpha_order_maxy=alpha_order,
        alpha_order_minz=alpha_order,
        alpha_order_maxz=alpha_order,
        sigma_start_minx=sigma_start,
        sigma_end_minx=sigma_end,
        sigma_start_maxx=sigma_start,
        sigma_end_maxx=sigma_end,
        sigma_start_miny=sigma_start,
        sigma_end_miny=sigma_end,
        sigma_start_maxy=sigma_start,
        sigma_end_maxy=sigma_end,
        sigma_start_minz=sigma_start,
        sigma_end_minz=sigma_end,
        sigma_start_maxz=sigma_start,
        sigma_end_maxz=sigma_end,
        sigma_order_minx=sigma_order,
        sigma_order_maxx=sigma_order,
        sigma_order_miny=sigma_order,
        sigma_order_maxy=sigma_order,
        sigma_order_minz=sigma_order,
        sigma_order_maxz=sigma_order,      
    )

    # Define simulation configuration
    config = fdtdx.SimulationConfig(
        resolution=resolution,
        time=time,
        courant_factor=courant_factor,
    )

    # Define materials
    materials = {
        "air": fdtdx.Material(),
        "background": fdtdx.Material(permittivity=1.44**2),
        "waveguide": fdtdx.Material(permittivity=4.0),
        "absorber": fdtdx.Material(
            permittivity=1.44**2, 
            electric_conductivity=0.104481
        ),
    }

    # Define background volume
    background = fdtdx.SimulationVolume(
        partial_real_shape=(
            30e-6 + 2*resolution*num_layers_x, 
            8e-6 + 2*resolution*num_layers_y, 
            8e-6 + 2*resolution*num_layers_z
        ),
        material=materials["background"],
    )
    object_list.append(background)

    # Define waveguide object
    waveguide = fdtdx.UniformMaterialObject(
        name="Waveguide",
        partial_real_shape=(None, float(width), float(height)),
        material=materials["waveguide"],
        color=fdtdx.colors.PINK,
    )
    object_list.append(waveguide)

    # Define absorber objects on left and right sides of waveguide
    absorber_l = fdtdx.UniformMaterialObject(
        name="Absorber_L",
        partial_real_shape=(None, float(absorber_thickness), float(height)),
        material=materials["absorber"],
        color=fdtdx.colors.ORANGE,
    )
    object_list.append(absorber_l)

    absorber_r = fdtdx.UniformMaterialObject(
        name="Absorber_R",
        partial_real_shape=(None, float(absorber_thickness), float(height)),
        material=materials["absorber"],
        color=fdtdx.colors.ORANGE,
    )
    object_list.append(absorber_r)

    # Define mode plane source
    source = fdtdx.ModePlaneSource(
        mode_index=0,
        filter_pol="te",
        partial_real_shape=(resolution, 6e-6, 6e-6), 
        direction='+',
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength),
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_frequency=frequency, 
            spectral_width=frequency/10
        ),
        name="Source",
        color=fdtdx.colors.GREEN,
    )
    object_list.append(source)

    # Create mode overlap detectors at multiple positions
    mode_overlap_detectors = []
    mode_overlap_detectors_placements = []
    num_detectors = 28
    for i in range(num_detectors):
        mode_overlap_detectors.append(
            fdtdx.ModeOverlapDetector(
                mode_index=0,
                filter_pol="te",
                partial_real_shape=(resolution, 6e-6, 6e-6), 
                direction='+',
                wave_characters=(fdtdx.WaveCharacter(wavelength=wavelength),),
                name=f"Mode Detector {i+1}",
            )
        )
        object_list.append(mode_overlap_detectors[i])
        
        # Place detectors at center in y and z, spaced along x
        mode_overlap_detectors_placements.append(
            mode_overlap_detectors[i].place_at_center(background, axes=(1,2)),
        )
        mode_overlap_detectors_placements.append(
            mode_overlap_detectors[i].place_relative_to(
                source, 
                axes=0, 
                other_positions=0, 
                own_positions=0, 
                margins=(i+1)*1e-6
            ),
        )

    # Create boundary objects from configuration
    boundary_dict, boundary_constraint_list = fdtdx.boundary_objects_from_config(bound_cfg, background)
    object_list.extend(list(boundary_dict.values()))

    # List all placement constraints
    placement_constraints = [
        # Boundary placement constraints
        *boundary_constraint_list,

        # Place waveguide at center in y and z directions
        waveguide.place_at_center(background, axes=(1,2)),

        # Place left absorber adjacent to waveguide
        absorber_l.place_relative_to(
            waveguide, 
            axes=(1,2), 
            other_positions=(-1,-1), 
            own_positions=(+1,-1)
        ),

        # Place right absorber adjacent to waveguide
        absorber_r.place_relative_to(
            waveguide, 
            axes=(1,2), 
            other_positions=(+1,-1), 
            own_positions=(-1,-1)
        ),

        # Place source at center and near the start of the volume
        source.place_at_center(background, axes=(1,2)),
        source.place_relative_to(
            background, 
            axes=0, 
            other_positions=0, 
            own_positions=0, 
            margins=-14e-6
        ),

        # Place mode overlap detectors
        *mode_overlap_detectors_placements,
    ] 

    # Place all objects in the simulation volume
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=placement_constraints,
        key=subkey,
    )

    # Apply parameters to arrays and objects
    arrays, new_objects, _ = fdtdx.apply_params(arrays, objects, params, key)

    return arrays, new_objects, config


def run_simulation(wavelength, mesh, width, height, absorber_thickness, time, key):
    """
    Run a single waveguide simulation and compute loss metrics.
    
    Args:
        wavelength: Wavelength of light in meters
        mesh: Resolution multiplier
        width: Waveguide width in meters
        height: Waveguide height in meters
        absorber_thickness: Absorber thickness in meters
        time: Total simulation time in seconds
        key: JAX random key
        
    Returns:
        mode_loss: Loss calculated from mode effective index (dB/cm)
        fdtd_loss: Loss calculated from FDTD field decay (dB/cm)
    """
    # Create simulation
    key, subkey = jax.random.split(key)
    arrays, new_objects, config = make_waveguide_simulation(
        wavelength, mesh, width, height, absorber_thickness, time, subkey
    )    

    # Run FDTD simulation
    key, subkey = jax.random.split(key)
    _, arrays = fdtdx.run_fdtd(arrays, new_objects, config, subkey)

    # Compute mode properties at first detector position
    Em0, Hm0, neff_complex = compute_modes_and_neff_complex(
        arrays, 
        new_objects["Mode Detector 1"]
    )

    # Calculate mode overlap at all detector positions
    num_detectors = 28
    mode_overlaps = []
    for i in jnp.arange(1, num_detectors+1):
        # Extract field phasors from detector
        Ef = arrays.detector_states[f"Mode Detector {i}"]["phasor"][0, 0, :3]
        Hf = arrays.detector_states[f"Mode Detector {i}"]["phasor"][0, 0, 3:]
        # Calculate overlap with reference mode
        mode_overlap = overlap(Ef, Em0, Hf, Hm0, 0)
        mode_overlaps.append(mode_overlap)
    mode_overlaps = jnp.array(mode_overlaps)

    # Fit linear regression to log of mode overlap vs distance
    x = jnp.arange(1, num_detectors+1)
    y = jnp.log10(jnp.abs(mode_overlaps))
    n = len(x)
    x_mean = jnp.sum(x) / n
    y_mean = jnp.sum(y) / n
    slope = jnp.sum((x - x_mean) * (y - y_mean)) / jnp.sum((x - x_mean)**2)
    ### intercept = y_mean - slope * x_mean

    # Calculate mode loss from complex effective index
    k_eff = jnp.imag(neff_complex)
    lambda_cm = wavelength * 100
    mode_loss = 20 * 2 * jnp.pi * jnp.log10(jnp.e) * k_eff / lambda_cm
    
    # Calculate FDTD loss from fitted slope
    fdtd_loss = -20 * slope * 10000.0

    return mode_loss, fdtd_loss


def plot_results(width_values, mode_losses, fdtd_losses, save_path):
    """
    Create and save a plot of loss vs waveguide width.
    
    Args:
        width_values: Array of waveguide widths in micrometers
        mode_losses: Array of mode losses in dB/cm
        fdtd_losses: Array of FDTD losses in dB/cm
        save_path: Path where the plot should be saved
    """
    plt.figure(figsize=(8, 6))
    
    # Plot mode loss as solid line
    plt.plot(
        width_values,
        mode_losses,
        "-",
        label="Mode Loss",
        linewidth=2,
        color="red",
    )
    
    # Plot FDTD loss as dotted line
    plt.plot(
        width_values,
        fdtd_losses,
        "--",
        label="FDTD Loss",
        linewidth=2,
        color="red",
    )
    
    plt.xlabel('Waveguide Width (μm)', fontsize=14)
    plt.ylabel('Loss (dB/cm)', fontsize=14)
    plt.title('Waveguide Loss vs. Width', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(
        loc="upper right", 
        fontsize=12,
        frameon=True,
        facecolor="white",
        framealpha=0.75,
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved to {save_path}")


def main(
    seed: int,
    save_results: bool,
    make_plot: bool,
):
    """
    Main function to run width sweep analysis.
    
    Args:
        seed: Random seed for reproducibility
        save_results: Whether to save results to file
        make_plot: Whether to create and save a plot of the results
    """
    # Log the random seed for reproducibility
    logger.info(f"{seed=}")

    # Initialize experiment logger for tracking results and saving outputs
    exp_logger = fdtdx.Logger(
        experiment_name="width_sweep_analysis",
        name=None,
    )
    # Create a JAX random key for stochastic operations
    key = jax.random.PRNGKey(seed=seed)

    # Fixed simulation parameters
    wavelength = 1.55e-6
    mesh = 15
    height = 0.4e-6
    absorber_thickness = 0.08e-6
    simulation_time = 400e-15

    # WIDTH values to sweep over
    width_values = np.array([
        0.5e-6, 0.6e-6, 0.7e-6, 0.8e-6, 0.9e-6, 1.0e-6, 1.1e-6, 1.2e-6, 1.5e-6,
        2.0e-6, 2.5e-6, 3.0e-6, 3.5e-6, 4.0e-6
    ])

    logger.info(f"Running simulations for {len(width_values)} different WIDTH values")
    logger.info(f"WIDTH values: {width_values}")
    logger.info(f"Fixed parameters: WAVELENGTH={wavelength}, MESH={mesh}, TIME={simulation_time}, HEIGHT={height}, ABSORBER_THICKNESS={absorber_thickness}")

    # Initialize lists to store results
    fdtd_losses = []
    mode_losses = []

    # Add progress bar for width sweep
    sweep_task_id = exp_logger.progress.add_task("Width Sweep", total=len(width_values))

    logger.info("Starting width sweep simulations...")
    
    # Run simulations for each WIDTH value
    for i, width in enumerate(width_values):
        logger.info(f"\nRunning simulation {i+1}/{len(width_values)} with WIDTH = {width:.1e}")
        
        # Start timer for this simulation
        run_start_time = time.time()
        
        # Run the simulation
        key, subkey = jax.random.split(key)
        mode_loss, fdtd_loss = run_simulation(
            wavelength, mesh, width, height, absorber_thickness, simulation_time, subkey
        )

        runtime_delta = time.time() - run_start_time

        # Store results
        fdtd_losses.append(fdtd_loss)
        mode_losses.append(mode_loss)
        
        logger.info(f"  mode_loss: {mode_loss}")
        logger.info(f"  fdtd_loss: {fdtd_loss}")
        logger.info(f"  runtime: {runtime_delta:.2f}s")

        # Log results for this iteration
        exp_logger.write({
            "iteration": i,
            "width": float(width),
            "mode_loss": float(mode_loss),
            "fdtd_loss": float(fdtd_loss),
            "runtime": runtime_delta,
        })

        exp_logger.progress.update(sweep_task_id, advance=1)

    logger.info("\nAll simulations completed!")
    
    # Convert results to numpy arrays
    fdtd_losses = np.array(fdtd_losses)
    mode_losses = np.array(mode_losses)

    # Save results if requested
    if save_results:
        results_dir = exp_logger.cwd / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Name of file
        filename = f"width_sweep_mesh{mesh}_absorber{absorber_thickness:.2e}.npz"
        
        # Save the results
        np.savez(
            results_dir / filename, 
            widths=width_values*1e6, 
            mode_losses=mode_losses,
            fdtd_losses=fdtd_losses,
        )
        logger.info(f"Results saved to {results_dir / filename}")

    # Log summary statistics
    logger.info(f"\nSummary:")
    logger.info(f"  Width range: {width_values[0]*1e6:.2f} - {width_values[-1]*1e6:.2f} μm")
    logger.info(f"  Mode loss range: {mode_losses.min():.4f} - {mode_losses.max():.4f} dB/cm")
    logger.info(f"  FDTD loss range: {fdtd_losses.min():.4f} - {fdtd_losses.max():.4f} dB/cm")

    # Create and save plot if requested
    if make_plot:
        plot_path = exp_logger.cwd / "width_sweep_loss_plot.png"
        plot_results(
            width_values=width_values*1e6,
            mode_losses=mode_losses,
            fdtd_losses=fdtd_losses,
            save_path=plot_path,
        )


# Entry point: parse command line arguments and run main function
if __name__ == "__main__": 
    seed = 0
    save_results = True
    make_plot = True
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    if len(sys.argv) > 2:
        save_results = sys.argv[2].lower() in ['true', '1', 'yes', 'y']
    if len(sys.argv) > 3:
        make_plot = sys.argv[3].lower() in ['true', '1', 'yes', 'y']
    main(
        seed=seed,
        save_results=save_results,
        make_plot=make_plot,
    )

