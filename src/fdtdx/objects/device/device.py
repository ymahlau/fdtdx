from abc import ABC
from typing import Self, Sequence

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, field, frozen_field, frozen_private_field
from fdtdx.core.jax.utils import check_specs
from fdtdx.core.misc import expand_matrix, is_float_divisible
from fdtdx.core.plotting.colors import PINK
from fdtdx.materials import Material
from fdtdx.objects.device.parameters.transform import ParameterTransformation
from fdtdx.objects.object import OrderableObject
from fdtdx.typing import (
    INVALID_SHAPE_3D,
    UNDEFINED_SHAPE_3D,
    ParameterType,
    PartialGridShape3D,
    PartialRealShape3D,
    SliceTuple3D,
)


@autoinit
class Device(OrderableObject, ABC):
    """Abstract base class for devices with optimizable permittivity distributions.

    This class defines the common interface and functionality for both discrete and
    continuous devices that can be optimized through gradient-based methods.

    Attributes:
        materials (dict[str, Material]): Dictionary of materials to be used in the device.
        param_transforms (Sequence[ParameterTransformation]): A Sequence of parameter transformation to be applied to
            the parameters when mapping them to simulation materials.
        color (tuple[float, float, float] | None, optional): Color of the object when plotted. Defaults to Pink.
        partial_voxel_grid_shape (PartialGridShape3D, optional): Size of the material voxels used within the device in
            metrical units (meter). Note that this is independent of the simulation voxel size. Defaults to undefined
            shape. For all three axes, either the voxel grid or real shape needs to be defined.
        partial_voxel_real_shape (PartialRealShape3D, optional): Size of the material voxels used within the device in
            simulation voxels. Defaults to undefined shape. For all three axes, either the voxel grid or real shape
            needs to be defined.
    """

    materials: dict[str, Material] = field()
    param_transforms: Sequence[ParameterTransformation] = field()
    color: tuple[float, float, float] | None = frozen_field(default=PINK)
    partial_voxel_grid_shape: PartialGridShape3D = frozen_field(default=UNDEFINED_SHAPE_3D)
    partial_voxel_real_shape: PartialRealShape3D = frozen_field(default=UNDEFINED_SHAPE_3D)

    _single_voxel_grid_shape: tuple[int, int, int] = frozen_private_field(default=INVALID_SHAPE_3D)

    @property
    def matrix_voxel_grid_shape(self) -> tuple[int, int, int]:
        """Calculate the shape of the voxel matrix in grid coordinates.

        Returns:
            tuple[int, int, int]: Tuple of (x,y,z) dimensions representing how many voxels fit in each direction
                of the grid shape when divided by the single voxel shape.
        """
        return (
            round(self.grid_shape[0] / self.single_voxel_grid_shape[0]),
            round(self.grid_shape[1] / self.single_voxel_grid_shape[1]),
            round(self.grid_shape[2] / self.single_voxel_grid_shape[2]),
        )

    @property
    def single_voxel_grid_shape(self) -> tuple[int, int, int]:
        """Get the shape of a single voxel in grid coordinates.

        Returns:
            tuple[int, int, int]: Tuple of (x,y,z) dimensions for one voxel.
        """
        if self._single_voxel_grid_shape == INVALID_SHAPE_3D:
            raise Exception(f"{self} is not initialized yet")
        return self._single_voxel_grid_shape

    @property
    def single_voxel_real_shape(self) -> tuple[float, float, float]:
        """Calculate the shape of a single voxel in real (physical) coordinates.

        Returns:
            tuple[float, float, float]: Tuple of (x,y,z) dimensions in real units, computed by multiplying
                the grid shape by the simulation resolution.
        """
        grid_shape = self.single_voxel_grid_shape
        return (
            grid_shape[0] * self._config.resolution,
            grid_shape[1] * self._config.resolution,
            grid_shape[2] * self._config.resolution,
        )

    @property
    def output_type(self) -> ParameterType:
        if not self.param_transforms:
            return ParameterType.CONTINUOUS
        out_type = self.param_transforms[-1]._output_type
        if isinstance(out_type, dict) and len(out_type) == 1:
            out_type = list(out_type.values())[0]
        if not isinstance(out_type, ParameterType):
            raise Exception(
                "Output of Parameter transformation sequence (last module) needs to be a single array, but got: "
                f"{out_type}"
            )
        return out_type

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)
        # determine voxel shape
        voxel_grid_shape = []
        for axis in range(3):
            partial_grid = self.partial_voxel_grid_shape[axis]
            partial_real = self.partial_voxel_real_shape[axis]
            if partial_grid is not None and partial_real is not None:
                raise Exception(f"Multi-Material voxels overspecified in axis: {axis=}")
            if partial_grid is not None:
                voxel_grid_shape.append(partial_grid)
            elif partial_real is not None:
                voxel_grid_shape.append(round(partial_real / config.resolution))
            else:
                raise Exception(f"Multi-Material voxels not specified in axis: {axis=}")

        self = self.aset("_single_voxel_grid_shape", tuple(voxel_grid_shape))

        # sanity checks on the voxel shape
        for axis in range(3):
            float_div = is_float_divisible(
                self.single_voxel_real_shape[axis],
                self._config.resolution,
            )
            if not float_div:
                raise Exception(f"Not divisible: {self.single_voxel_real_shape[axis]=}, {self._config.resolution=}")
            if self.grid_shape[axis] % self.matrix_voxel_grid_shape[axis] != 0:
                raise Exception(
                    f"Due to discretization, matrix got skewered for {axis=}. "
                    f"{self.grid_shape=}, {self.matrix_voxel_grid_shape=}"
                )

        # init parameter transformations
        # We need to go once backwards through the transformations to determine the shape of the latent parameters
        # then we need to go forward through the transformations again to determine the parameter type of the
        # output
        new_t_list: list[ParameterTransformation] = []
        cur_shape = {"params": self.matrix_voxel_grid_shape}
        for transform in self.param_transforms[::-1]:
            t_new = transform.init_module(
                config=config,
                materials=self.materials,
                matrix_voxel_grid_shape=self.matrix_voxel_grid_shape,
                single_voxel_size=self.single_voxel_real_shape,
                output_shape=cur_shape,
            )
            new_t_list.append(t_new)
            cur_shape = t_new._input_shape

        # init shape of transformations by going backwards through new list
        module_list: list[ParameterTransformation] = []
        cur_input_type = {"params": ParameterType.CONTINUOUS}
        for transform in new_t_list[::-1]:
            t_new = transform.init_type(
                input_type=cur_input_type,
            )
            module_list.append(t_new)
            cur_input_type = t_new._output_type

        # set own input shape dtype
        self = self.aset("param_transforms", module_list)
        if self.output_type == ParameterType.CONTINUOUS and len(self.materials) != 2:
            raise Exception(
                f"Need exactly two materials in device when parameter mapping outputs continuous permittivity indices, "
                f"but got {self.materials}"
            )
        return self

    def init_params(
        self,
        key: jax.Array,
    ) -> dict[str, jax.Array] | jax.Array:
        if len(self.param_transforms) > 0:
            shapes = self.param_transforms[0]._input_shape
        else:
            shapes = self.matrix_voxel_grid_shape
        if not isinstance(shapes, dict):
            shapes = {"params": shapes}
        params = {}
        for k, cur_shape in shapes.items():
            key, subkey = jax.random.split(key)
            p = jax.random.uniform(
                key=subkey,
                shape=cur_shape,
                minval=0,  # parameter always live between 0 and 1
                maxval=1,
                dtype=jnp.float32,
            )
            params[k] = p
        if len(params) == 1:
            params = list(params.values())[0]
        return params

    def __call__(
        self,
        params: dict[str, jax.Array] | jax.Array,
        expand_to_sim_grid: bool = False,
        **transform_kwargs,
    ) -> jax.Array:
        if not isinstance(params, dict):
            params = {"params": params}
        # walk through modules
        for transform in self.param_transforms:
            check_specs(params, transform._input_shape)
            params = transform(params, **transform_kwargs)
            check_specs(params, transform._output_shape)
        if len(params) == 1:
            params = list(params.values())[0]
        else:
            raise Exception(
                "The parameter mapping should return a single array of indices. If using a continuous device, please"
                " make sure that the latent transformations abide to this rule."
            )
        if expand_to_sim_grid:
            params = expand_matrix(
                matrix=params,
                grid_points_per_voxel=self.single_voxel_grid_shape,
            )
        return params
