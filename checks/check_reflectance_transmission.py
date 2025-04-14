import math
import time

import jax
import jax.numpy as jnp
import optax
import pytreeclass as tc
from loguru import logger

from fdtdx.core.physics.metrics import poynting_flux
from fdtdx.core.plotting.debug import debug_plot_lines
from fdtdx.core.switch import OnOffSwitch
from fdtdx.fdtd import (
    ArrayContainer,
    ParameterContainer,
    apply_params,
    place_objects,
    reversible_fdtd
)
from fdtdx.config import  SimulationConfig
from fdtdx import constants
from fdtdx.materials import Material
from fdtdx.objects import  SimulationVolume, UniformMaterialObject
from fdtdx.objects.boundaries import BoundaryConfig, boundary_objects_from_config
from fdtdx.objects.detectors import EnergyDetector, PoyntingFluxDetector
from fdtdx.objects.detectors.field import FieldDetector
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.sources import SimplePlaneSource
from fdtdx.core import WaveCharacter
from fdtdx.utils import Logger, plot_setup


def main():
    exp_logger = Logger(
        experiment_name="reflectance_transmission",
    )
    key = jax.random.PRNGKey(seed=42)

    wavelength = 1.55e-6
    angle_degree = 15
    angle_radians = angle_degree * math.pi / 180
    size_xy = wavelength / math.sin(angle_radians)
    resolution = size_xy / 100
    logger.info(f"{angle_degree=}")
    logger.info(f"{size_xy=}")
    logger.info(f"{resolution=}")
    
    period = constants.wavelength_to_period(wavelength)

    config = SimulationConfig(
        time=100e-15,
        resolution=resolution,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    constraints = []

    volume = SimulationVolume(
        partial_real_shape=(size_xy, size_xy, 2 * size_xy),
        material=Material(  # Background material
            permittivity=1.0,
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
        azimuth_angle=angle_degree,
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
    
    # substrate = UniformMaterialObject(
    #     material=Material(permittivity=2.5),
    #     partial_real_shape=(None, None, size_xy)
    # )
    # constraints.append(
    #     substrate.place_relative_to(
    #         volume,
    #         axes=(0, 1, 2),
    #         own_positions=(0, 0, -1),
    #         other_positions=(0, 0, -1),
    #     )
    # )
    
    
    fields_below_source = FieldDetector(
        name="fields",
        partial_grid_shape=(1, 1, 1),
        switch=OnOffSwitch(
            start_after_periods=10,
            period=period,
        ),
        exact_interpolation=True,
        reduce_volume=True,
    )
    constraints.extend(
        [
            fields_below_source.place_relative_to(
                source,
                axes=(0, 1, 2),
                own_positions=(0, 0, 1),
                other_positions=(0, 0, 1),
                grid_margins=(0, 0, -5),
            ),
        ]
    )
    
    freqs_below_source = PhasorDetector(
        name="freqs",
        wave_characters=[source.wave_character],
        partial_grid_shape=(1, 1, 1),
        switch=OnOffSwitch(
            start_after_periods=10,
            period=period,
        ),
        exact_interpolation=True,
        reduce_volume=True,
    )
    constraints.extend(
        [
            freqs_below_source.place_relative_to(
                source,
                axes=(0, 1, 2),
                own_positions=(0, 0, 1),
                other_positions=(0, 0, 1),
                grid_margins=(0, 0, -5),
            ),
        ]
    )

    video_energy_detector = EnergyDetector(
        name="Energy Video",
        as_slices=True,
        switch=OnOffSwitch(interval=3),
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

        final_state = reversible_fdtd(
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
    
    # compute frequency-transform of Ex component
    pf = arrays.detector_states[fields_below_source.name]["fields"]
    pfx = pf[:, 0]
    signal_length = pfx.shape[0]
    sampling_frequency = 1 / config.time_step_duration
    desired_frequency = source.wave_character.frequency
    t = jnp.arange(signal_length) / sampling_frequency
    freqs = jnp.fft.fftfreq(signal_length, 1 / sampling_frequency)
    freqs_vals = jnp.fft.fft(pfx)
    
    target_idx = jnp.abs(freqs - desired_frequency).argmin() + 1
    desired_freqs_val = freqs_vals[target_idx]
    
    amplitude = 2 * jnp.abs(desired_freqs_val) / signal_length
    phase = jnp.angle(desired_freqs_val)
    
    recs = amplitude * jnp.sin(2 * jnp.pi * freqs[target_idx] * t + phase)

    # compute specific frequency 
    angular_freq = 2 * jnp.pi * desired_frequency
    state = 0
    for t_step in range(len(pfx)):
        time_passed = t_step * config.time_step_duration
        cur_phase_angle = time_passed * angular_freq
        phasor = jnp.exp(1j * cur_phase_angle)
        new_phasor = pfx[t_step] * phasor
        state = state + new_phasor
    
    amplitude2 = 2 * jnp.abs(state) / signal_length
    phase2 = jnp.angle(state)
    recs2 = amplitude2 * jnp.cos(2 * jnp.pi * desired_frequency * t - phase2)
    
    # compute specific frequency using phasor detector
    phasors = arrays.detector_states[freqs_below_source.name]["phasor"].squeeze()
    phasors_E = phasors[:3]
    phasors_H = phasors[3:]
    
    state3 = phasors_E[0]
    amplitude3 = jnp.abs(state3)
    phase3 = jnp.angle(state3)
    recs3 = amplitude3 * jnp.cos(2 * jnp.pi * desired_frequency * t - phase3)
    
    debug_plot_lines({"Ex": pfx - pfx.mean(), "recs": recs, "recs2": recs2, "recs3": recs3})
    
    fields_E = pf[:, :3]
    fields_H = pf[:, 3:]
    flux = poynting_flux(fields_E, fields_H, axis=1)
    avg_flux = jnp.mean(flux, axis=0)
    
    mean_flux = 0.5 * jnp.real(poynting_flux(phasors_E, phasors_H))
    
    logger.info(f"Average flux measured at every time step: {avg_flux}")
    logger.info(f"Average flux computed through frequency transform: {mean_flux}")
    
    # measure poynting flux angle below source
    pf_norm = -mean_flux / optax.global_norm(mean_flux)
    prop_vec = jnp.asarray([0, 0, 1], dtype=jnp.float32)
    measured_prop_angle = jnp.arccos(jnp.dot(prop_vec, pf_norm)) * 180 / jnp.pi
    difference = angle_degree - measured_prop_angle
    logger.info(f"{measured_prop_angle=} Degree, {difference=}")
    


if __name__ == "__main__":
   main()


