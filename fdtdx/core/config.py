import math
from typing import Literal
import pytreeclass as tc
import jax.numpy as jnp

from fdtdx.core.physics import constants
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field
from fdtdx.interfaces.recorder import Recorder


class GradientConfig(ExtendedTreeClass):
    
    def __init__(
        self,
        recorder: Recorder | None = None,  # if not none, use invertible diff
        num_checkpoints: int | None = None
    ):
        self.recorder = recorder
        self.num_checkpoints = num_checkpoints
        if self.recorder is not None and self.num_checkpoints is not None:
            raise Exception(f"Cannot use both invertible and checkpointing autodiff!")
        if self.recorder is None and self.num_checkpoints is None:
            raise Exception(f"Need either recorder or checkpoints to define autograd!")   
    

@extended_autoinit
class SimulationConfig(ExtendedTreeClass):
    time: float
    resolution: float
    backend: Literal["gpu", "tpu", "cpu"] = frozen_field(default="gpu")
    dtype: jnp.dtype = frozen_field(default=jnp.float32)
    courant_factor: float = 0.99
    gradient_config: GradientConfig | None = None
    
    @property
    def courant_number(self) -> float:
        return self.courant_factor / math.sqrt(3)

    @property
    def time_step_duration(self) -> float:
        return self.courant_number * self.resolution / constants.c

    @property
    def time_steps_total(self) -> int:
        return round(self.time / self.time_step_duration)
    
    @property
    def max_travel_distance(self) -> float:
        return constants.c * self.time
    
    @property
    def only_forward(self) -> bool:
        return self.gradient_config is None
    
    @property
    def invertible_optimization(self) -> bool:
        if self.gradient_config is None:
            return False
        return self.gradient_config.recorder is not None
    

DUMMY_SIMULATION_CONFIG = SimulationConfig(
    time=-1,
    resolution=-1,
)



    
    
    