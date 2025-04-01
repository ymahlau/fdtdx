import jax
import jax.numpy as jnp
import pytreeclass as tc
from loguru import logger

from fdtdx.core.switch import OnOffSwitch
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.materials import Material
from fdtdx.config import SimulationConfig
from fdtdx import constants
from fdtdx.fdtd import reversible_fdtd, ArrayContainer, ParameterContainer, apply_params, place_objects
from fdtdx.objects import SimulationVolume, Substrate, Waveguide, SimulationObject
from fdtdx.objects.boundaries import BoundaryConfig, boundary_objects_from_config
from fdtdx.objects.detectors import EnergyDetector
from fdtdx.objects.sources import ModePlaneSource
from fdtdx.utils import Logger, plot_setup

def main():
    exp_logger = Logger(
        experiment_name="check_waveguide_modes",
        name=None,
    )
    key = jax.random.PRNGKey(seed=42)

    wavelength = 1.55e-6
    period = constants.wavelength_to_period(wavelength)

    config = SimulationConfig(
        time=50e-15,
        resolution=20e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    all_time_steps = list(range(config.time_steps_total))
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    placement_constraints = []

    volume = SimulationVolume(
        partial_real_shape=(2e-6, 2e-6, 2e-6),
    )

    bound_cfg = BoundaryConfig.from_uniform_bound(thickness=10)
    _, c_list = boundary_objects_from_config(bound_cfg, volume)
    placement_constraints.extend(c_list)

    substrate = Substrate(
        partial_real_shape=(None, None, 0.8e-6),
        material=Material(permittivity=constants.relative_permittivity_silica),
    )
    placement_constraints.append(
        substrate.place_relative_to(
            volume,
            axes=2,
            own_positions=-1,
            other_positions=-1,
        )
    )
    
    waveguide_in = Waveguide(
        partial_real_shape=(None, 0.4e-6, 0.4e-6),
        material=Material(permittivity=constants.relative_permittivity_silicon),
    )
    placement_constraints.extend(
        [
            waveguide_in.place_at_center(
                substrate,
                axes=1,
            ),
            waveguide_in.place_above(substrate),
        ]
    )
    
    source = ModePlaneSource(
        partial_grid_shape=(1, None, None),
        wave_character=WaveCharacter(wavelength=wavelength),
        direction="+",
        mode_index=0,
    )
    placement_constraints.extend(
        [
            source.place_relative_to(
                waveguide_in,
                axes=(0,),
                other_positions=(0,),
                own_positions=(0,),
            )
        ]
    )
    
    energy_last_step = EnergyDetector(
        name="energy_last_step",
        as_slices=True,
        switch=OnOffSwitch(fixed_on_time_steps=[-1]),
    )
    placement_constraints.extend([*energy_last_step.same_position_and_size(volume)])

    exclude_object_list: list[SimulationObject] = [energy_last_step]
    video_detector = EnergyDetector(
        name="video",
        as_slices=True,
        switch=OnOffSwitch(interval=10),
        exact_interpolation=True,
        plot_interpolation="nearest",
    )
    placement_constraints.extend([*video_detector.same_position_and_size(volume)])
    exclude_object_list.append(video_detector)
    
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = place_objects(
        volume=volume,
        config=config,
        constraints=placement_constraints,
        key=subkey,
    )

    logger.info(tc.tree_summary(arrays, depth=2))
    print(tc.tree_diagram(config, depth=4))
    
    exp_logger.savefig(
        exp_logger.cwd,
        "setup.png",
        plot_setup(
            config=config,
            objects=objects,
            exclude_object_list=exclude_object_list,
        ),
    )
    
    _, tmp, _ = apply_params(arrays, objects, params, key)
    tmp.sources[0].plot(  # type: ignore
        exp_logger.cwd / "figures" / "mode.png"
    )
    
    @jax.jit
    def sim_func(
        params: ParameterContainer,
        arrays: ArrayContainer,
        key: jax.Array,
    ):
        arrays, new_objects, info = apply_params(arrays, objects, params, key)

        _, arrays = reversible_fdtd(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )
        
        return arrays

    key, subkey = jax.random.split(key)
    arrays = sim_func(params, arrays, subkey)
    exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=arrays.detector_states)
    
    

if __name__ == '__main__':
    main()
