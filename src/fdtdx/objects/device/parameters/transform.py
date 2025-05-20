from abc import ABC, abstractmethod
from typing import Self

import jax

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_private_field
from fdtdx.materials import Material
from fdtdx.typing import ParameterSpecs, ParameterType


@extended_autoinit
class ParameterTransformation(ExtendedTreeClass, ABC):
    _input_specs: dict[str, ParameterSpecs] | ParameterSpecs = frozen_private_field()
    _output_specs: dict[str, ParameterSpecs] | ParameterSpecs = frozen_private_field()
    _materials: dict[str, Material] = frozen_private_field()
    _config: SimulationConfig = frozen_private_field()
    _matrix_voxel_grid_shape: tuple[int, int, int] = frozen_private_field()
    _single_voxel_size: tuple[float, float, float] = frozen_private_field()

    def init_module(
        self: Self,
        config: SimulationConfig,
        materials: dict[str, Material],
        input_specs: dict[str, ParameterSpecs] | ParameterSpecs,
        matrix_voxel_grid_shape: tuple[int, int, int],
        single_voxel_size: tuple[float, float, float],
    ) -> Self:
        self = self.aset("_config", config)
        self = self.aset("_materials", materials)
        self = self.aset("_input_specs", input_specs)
        self = self.aset("_matrix_voxel_grid_shape", matrix_voxel_grid_shape)
        self = self.aset("_single_voxel_size", single_voxel_size)
        output_specs = self.get_output_specs(input_specs)
        self = self.aset("_output_specs", output_specs)
        return self

    @abstractmethod
    def __call__(
        self,
        params: dict[str, jax.Array] | jax.Array,
        **kwargs,
    ) -> dict[str, jax.Array] | jax.Array:
        del params
        raise NotImplementedError()

    @abstractmethod
    def get_output_specs(
        self,
        input_specs: dict[str, ParameterSpecs] | ParameterSpecs,
    ) -> dict[str, ParameterSpecs] | ParameterSpecs:
        del input_specs
        raise NotImplementedError()


class SameShapeTypeParameterTransform(ParameterTransformation, ABC):
    def get_output_specs(
        self,
        input_specs: dict[str, ParameterSpecs] | ParameterSpecs,
    ) -> dict[str, ParameterSpecs] | ParameterSpecs:
        return input_specs


class SameShapeContinousParameterTransform(ParameterTransformation, ABC):
    def get_output_specs(
        self,
        input_specs: dict[str, ParameterSpecs] | ParameterSpecs,
    ) -> dict[str, ParameterSpecs] | ParameterSpecs:
        err_msg = (
            f"Before ContinousParameterTransform({self.__class__}) needs to be a transform "
            + "outputting continous parameters"
        )
        if isinstance(input_specs, ParameterSpecs):
            if input_specs.type != ParameterType.CONTINUOUS:
                raise Exception(err_msg)
        else:
            for v in input_specs.values():
                if v.type != ParameterType.CONTINUOUS:
                    raise Exception(err_msg)
        return input_specs


class SameShapeDiscreteParameterTransform(ParameterTransformation, ABC):
    def get_output_specs(
        self,
        input_specs: dict[str, ParameterSpecs] | ParameterSpecs,
    ) -> dict[str, ParameterSpecs] | ParameterSpecs:
        err_msg = (
            f"Before DiscreteParameterTransform({self.__class__}) needs to be a transform "
            + "outputting discrete parameters"
        )
        if isinstance(input_specs, ParameterSpecs):
            if input_specs.type != ParameterType.DISCRETE and input_specs.type != ParameterType.BINARY:
                raise Exception(err_msg)
        else:
            for v in input_specs.values():
                if v.type != ParameterType.DISCRETE and v.type != ParameterType.BINARY:
                    raise Exception(err_msg)
        return input_specs


class SameShapeBinaryParameterTransform(ParameterTransformation, ABC):
    def get_output_specs(
        self,
        input_specs: dict[str, ParameterSpecs] | ParameterSpecs,
    ) -> dict[str, ParameterSpecs] | ParameterSpecs:
        err_msg = (
            f"Before BinaryParameterTransform({self.__class__}) needs to be a transform "
            + "outputting binary parameters"
        )
        if isinstance(input_specs, ParameterSpecs):
            if input_specs.type != ParameterType.BINARY:
                raise Exception(err_msg)
        else:
            for v in input_specs.values():
                if v.type != ParameterType.BINARY:
                    raise Exception(err_msg)
        return input_specs
