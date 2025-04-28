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
from fdtdx.fdtd.backward import full_backward
from fdtdx.interfaces.modules import DtypeConversion
from fdtdx.interfaces.recorder import Recorder
from fdtdx.materials import Material
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx import constants
from fdtdx.fdtd import ArrayContainer, ParameterContainer, apply_params, run_fdtd, place_objects
from fdtdx.objects import SimulationVolume, Substrate, Waveguide, SimulationObject
from fdtdx.objects.boundaries import BoundaryConfig, boundary_objects_from_config
from fdtdx.objects.detectors import EnergyDetector
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.device.device import DiscreteDevice
from fdtdx.objects.device.parameters.mapping import DiscreteParameterMapping
from fdtdx.objects.sources import ModePlaneSource
from fdtdx.objects.sources.linear_polarization import SimplePlaneSource
from fdtdx.utils import Logger, plot_setup
from fdtdx.core.plotting.debug import debug_plot_2d, debug_plot_lines


def main():
    exp_logger = Logger(
        experiment_name="check_gradients",
    )
    key = jax.random.PRNGKey(seed=42)

    wavelength = 1.55e-6
    
    period = constants.wavelength_to_period(wavelength)

    config = SimulationConfig(
        time=150e-15,
        resolution=50e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=GradientConfig(
            recorder=Recorder(
                modules=[DtypeConversion(dtype=jnp.float16),]
            ),
            # method="checkpointed",
            # num_checkpoints=20,
        )
    )

    period_steps = round(period / config.time_step_duration)
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    constraints = []

    volume = SimulationVolume(
        partial_real_shape=(5e-6, 5e-6, 10e-6),
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
    
    device = DiscreteDevice(
        partial_real_shape=(2e-6, 2e-6, 2e-6),
        partial_voxel_real_shape=(1e-7, 1e-7, 1e-7),
        material={
            "air": Material(),
            "silicon": Material(permittivity=12.25),
        },
        parameter_mapping=DiscreteParameterMapping()
    )
    constraints.extend([
        device.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, 0),
        )
    ])
    
    energy_detector = EnergyDetector(
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(0.3e-7, 0.3e-7, None),
        name="target",
    )
    constraints.extend([
        energy_detector.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, -1),
            grid_margins=(0, 0, 15),
        )
    ])
    
    video_energy_detector = EnergyDetector(
        name="Energy Video",
        as_slices=True,
        switch=OnOffSwitch(interval=5),
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
                video_energy_detector,
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

        _, arrays = run_fdtd(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )
        
        target_energy = arrays.detector_states[energy_detector.name]["energy"]
        objective = target_energy.mean()
        
        return objective, (arrays,)
    
    grad_fn = jax.jit(jax.value_and_grad(sim_fn, has_aux=True))
    
    (val, (aux,)), grad = grad_fn(params, arrays, key)
    
    exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=aux.detector_states)
    debug_plot_2d(grad[device.name].mean(axis=-1), tmp_dir=exp_logger.cwd / "figures", filename="checkpointed_grad")
    


if __name__ == '__main__':
    main()
