import math
import time
import jax
import jax.numpy as jnp
import numpy as np
import pytreeclass as tc
from loguru import logger

from fdtdx.core.physics.metrics import compute_energy
from fdtdx.core.switch import OnOffSwitch
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.materials import Material
from fdtdx.config import SimulationConfig
from fdtdx import constants
from fdtdx.fdtd import ArrayContainer, ParameterContainer, apply_params, run_fdtd, place_objects
from fdtdx.objects import SimulationVolume, Substrate, Waveguide, SimulationObject
from fdtdx.objects.boundaries import BoundaryConfig, boundary_objects_from_config
from fdtdx.objects.detectors import EnergyDetector
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.sources import ModePlaneSource
from fdtdx.objects.sources.linear_polarization import SimplePlaneSource
from fdtdx.utils import Logger, plot_setup
from fdtdx.core.plotting.debug import debug_plot_lines

def main():
    exp_logger = Logger(
        experiment_name="lossy_material",
    )
    key = jax.random.PRNGKey(seed=42)

    wavelength = 1.55e-6
    
    period = constants.wavelength_to_period(wavelength)

    config = SimulationConfig(
        time=200e-15,
        resolution=50e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    constraints = []

    volume = SimulationVolume(
        partial_real_shape=(5e-6, 5e-6, 10e-6),
        material=Material(  # Background material
            permittivity=1.0,
            permeability=1.0,
            electric_conductivity=1e5,
            magnetic_conductivity=0.0,
        )
    )

    bound_cfg = BoundaryConfig(
        boundary_type_maxx="periodic",
        boundary_type_maxy="periodic",
        boundary_type_minx="periodic",
        boundary_type_miny="periodic",
    )
    bound_dict, c_list = boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)

    source = SimplePlaneSource(
        partial_grid_shape=(None, None, 1),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=WaveCharacter(wavelength=1.550e-6),
        direction="-",
        # switch=OnOffSwitch(start_after_periods=2, on_for_periods=2, period=period),
    )
    constraints.extend(
        [
            source.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 1),
                other_positions=(0, 0, 1),
                grid_margins=(0, 0, -15),
            ),
        ]
    )
    
    phasor_detector = PhasorDetector(
        name="Phasor",
        wave_characters=[source.wave_character],
        plot=False,
        exact_interpolation=True,
        switch=OnOffSwitch(start_time=100e-15),
    )
    constraints.extend(phasor_detector.same_position_and_size(volume))
    
    video_energy_detector = EnergyDetector(
        name="Energy Video",
        as_slices=True,
        switch=OnOffSwitch(interval=5, start_time=100e-15),
        exact_interpolation=True,
    )
    constraints.extend(video_energy_detector.same_position_and_size(volume))

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = place_objects(
        volume=volume,
        config=config,
        constraints=constraints,
        key=subkey,
    )
    logger.info(tc.tree_summary(arrays, depth=1))
    print(tc.tree_diagram(config, depth=4))

    exp_logger.savefig(
        exp_logger.cwd,
        "setup.png",
        plot_setup(
            config=config,
            objects=objects,
            exclude_object_list=[
                # video_energy_detector,
            ],
        ),
    )

    apply_params(arrays, objects, params, key)

    def sim_fn(
        params: ParameterContainer,
        arrays: ArrayContainer,
        key: jax.Array,
    ):
        arrays, new_objects, info = apply_params(arrays, objects, params, key)

        final_state = run_fdtd(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )
        _, arrays = final_state

        new_info = {
            **info,
        }
        return arrays, new_info

    compile_start_time = time.time()
    jit_task_id = exp_logger.progress.add_task("JIT", total=None)
    jitted_loss = jax.jit(sim_fn, donate_argnames=["arrays"]).lower(params, arrays, key).compile()
    compile_delta_time = time.time() - compile_start_time
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)

    run_start_time = time.time()
    key, subkey = jax.random.split(key)
    arrays, info = jitted_loss(params, arrays, subkey)

    runtime_delta = time.time() - run_start_time
    info["runtime"] = runtime_delta
    info["compile time"] = compile_delta_time

    logger.info(f"{compile_delta_time=}")
    logger.info(f"{runtime_delta=}")

    exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=arrays.detector_states)
    
    # get numerical amplitude measure
    phasors = arrays.detector_states[phasor_detector.name]["phasor"][0][0].mean(axis=(1, 2))
    p_E = phasors[:3]
    p_H = phasors[3:]
    energy = compute_energy(
        E=p_E,
        H=p_H,
        inv_permittivity=arrays.inv_permittivities.mean(axis=(0, 1)),
        inv_permeability=arrays.inv_permeabilities,  # scalar
    )
    # isolate decay region
    decay_vals = energy[175:25:-1]
    # decay_vals = energy[70:10:-1]
    log_vals = np.asarray(jnp.log(decay_vals))
    x = np.arange(len(decay_vals))
    coeffs = np.polyfit(x, log_vals, deg=1)
    alpha = coeffs[0]
    scale = np.exp(coeffs[1])
    fit = scale * np.exp(alpha * x)
    
    logger.info(f"Measured decay constant: {alpha}")

    # compare fit and original
    debug_plot_lines({"measure": decay_vals, "fit": fit})
    
    

if __name__ == '__main__':
    main()
