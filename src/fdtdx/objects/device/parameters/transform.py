from abc import ABC, abstractmethod
from typing import Self, Sequence

import jax

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_private_field
from fdtdx.materials import Material
from fdtdx.typing import ParameterType


@autoinit
class ParameterTransformation(TreeClass, ABC):
    _input_type: dict[str, ParameterType] = frozen_private_field()
    _input_shape: dict[str, tuple[int, ...]] = frozen_private_field()
    _output_type: dict[str, ParameterType] = frozen_private_field()
    _output_shape: dict[str, tuple[int, ...]] = frozen_private_field()
    _materials: dict[str, Material] = frozen_private_field()
    _config: SimulationConfig = frozen_private_field()
    _matrix_voxel_grid_shape: tuple[int, int, int] = frozen_private_field()
    _single_voxel_size: tuple[float, float, float] = frozen_private_field()

    # settings
    _check_single_array: bool = frozen_private_field(default=False)
    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(default=None)
    _all_arrays_2d: bool = frozen_private_field(default=False)

    def init_module(
        self: Self,
        config: SimulationConfig,
        materials: dict[str, Material],
        matrix_voxel_grid_shape: tuple[int, int, int],
        single_voxel_size: tuple[float, float, float],
        output_shape: dict[str, tuple[int, ...]],
    ) -> Self:
        self = self.aset("_config", config, create_new_ok=True)
        self = self.aset("_materials", materials, create_new_ok=True)
        self = self.aset("_matrix_voxel_grid_shape", matrix_voxel_grid_shape, create_new_ok=True)
        self = self.aset("_single_voxel_size", single_voxel_size, create_new_ok=True)

        self = self.aset("_output_shape", output_shape, create_new_ok=True)
        input_shape = self.get_input_shape(output_shape)
        self = self.aset("_input_shape", input_shape, create_new_ok=True)
        return self

    def init_type(
        self,
        input_type: dict[str, ParameterType],
    ) -> Self:
        # given input type
        self = self.aset("_input_type", input_type, create_new_ok=True)
        # compute output type
        output_type = self.get_output_type(input_type)
        self = self.aset("_output_type", output_type, create_new_ok=True)
        return self

    def get_output_type(
        self,
        input_type: dict[str, ParameterType],
    ) -> dict[str, ParameterType]:
        # checks
        if self._check_single_array and len(input_type) != 1:
            raise Exception(
                f"ParameterTransform {self.__class__} expects input to be a single array, but got: {input_type}"
            )
        if self._fixed_input_type is not None:
            for v in input_type.values():
                err_msg = (
                    f"ParameterTransform {self.__class__} expects all input types to be {self._fixed_input_type}"
                    f", but got {input_type}"
                )
                if isinstance(self._fixed_input_type, Sequence):
                    if v not in self._fixed_input_type:
                        raise Exception(err_msg)
                elif v != self._fixed_input_type:
                    raise Exception(err_msg)
        # implementation
        output_type = self._get_output_type_impl(input_type)
        return output_type

    def get_input_shape(
        self,
        output_shape: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        # checks
        if self._all_arrays_2d:
            for v in output_shape.values():
                err_msg = (
                    f"ParameterTransform {self.__class__} expects to work with 2d arrays, so exactly one axis of the "
                    f"3d array needs to have size of 1, but got: {output_shape}"
                )
                if len(v) != 3 or 1 not in v:
                    raise Exception(err_msg)
                if sum([n != 1 for n in v]) != 2:
                    raise Exception(err_msg)

        # implementation
        input_shape = self._get_input_shape_impl(output_shape)
        return input_shape

    @abstractmethod
    def _get_input_shape_impl(
        self,
        output_shape: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        raise NotImplementedError()

    @abstractmethod
    def _get_output_type_impl(
        self,
        input_type: dict[str, ParameterType],
    ) -> dict[str, ParameterType]:
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        raise NotImplementedError()


@autoinit
class SameShapeTypeParameterTransform(ParameterTransformation, ABC):
    def _get_input_shape_impl(
        self,
        output_shape: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        return output_shape

    def _get_output_type_impl(
        self,
        input_type: dict[str, ParameterType],
    ) -> dict[str, ParameterType]:
        return input_type
