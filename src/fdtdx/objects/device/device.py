from abc import ABC
from typing import Self, Sequence

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import extended_autoinit, field, frozen_field
from fdtdx.core.jax.utils import check_specs
from fdtdx.core.misc import expand_matrix, is_float_divisible
from fdtdx.core.plotting.colors import PINK
from fdtdx.materials import Material
from fdtdx.objects.device.parameters.transform import ParameterTransformation
from fdtdx.objects.object import OrderableObject
from fdtdx.typing import (
    INVALID_SHAPE_3D,
    UNDEFINED_SHAPE_3D,
    GridShape3D,
    ParameterSpecs,
    ParameterType,
    PartialGridShape3D,
    PartialRealShape3D,
    RealShape3D,
    SliceTuple3D,
)


@extended_autoinit
class Device(OrderableObject, ABC):
    """Abstract base class for devices with optimizable permittivity distributions.

    This class defines the common interface and functionality for both discrete and
    continuous devices that can be optimized through gradient-based methods.

    Attributes:
        name: Optional name identifier for the device
        dtype: Data type for device parameters, defaults to float32
        color: RGB color tuple for visualization, defaults to pink
    """

    materials: dict[str, Material] = frozen_field(kind="KW_ONLY")
    param_transforms: Sequence[ParameterTransformation] = frozen_field(kind="KW_ONLY")
    color: tuple[float, float, float] | None = frozen_field(default=PINK)
    partial_voxel_grid_shape: PartialGridShape3D = field(default=UNDEFINED_SHAPE_3D)
    partial_voxel_real_shape: PartialRealShape3D = field(default=UNDEFINED_SHAPE_3D)
    _single_voxel_grid_shape: GridShape3D = field(default=INVALID_SHAPE_3D, init=False)

    @property
    def matrix_voxel_grid_shape(self) -> GridShape3D:
        """Calculate the shape of the voxel matrix in grid coordinates.

        Returns:
            Tuple of (x,y,z) dimensions representing how many voxels fit in each direction
            of the grid shape when divided by the single voxel shape.
        """
        return (
            round(self.grid_shape[0] / self.single_voxel_grid_shape[0]),
            round(self.grid_shape[1] / self.single_voxel_grid_shape[1]),
            round(self.grid_shape[2] / self.single_voxel_grid_shape[2]),
        )

    @property
    def single_voxel_grid_shape(self) -> GridShape3D:
        """Get the shape of a single voxel in grid coordinates.

        Returns:
            Tuple of (x,y,z) dimensions for one voxel.

        Raises:
            Exception: If the object has not been initialized yet.
        """
        if self._single_voxel_grid_shape == INVALID_SHAPE_3D:
            raise Exception(f"{self} is not initialized yet")
        return self._single_voxel_grid_shape

    @property
    def single_voxel_real_shape(self) -> RealShape3D:
        """Calculate the shape of a single voxel in real (physical) coordinates.

        Returns:
            Tuple of (x,y,z) dimensions in real units, computed by multiplying
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
        out_specs = self.param_transforms[-1]._output_specs
        if not isinstance(out_specs, ParameterSpecs):
            raise Exception(f"Output of Parameter transformation needs to be a single array, but got: {out_specs}")
        return out_specs.type

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
                raise Exception(f"Not divisible: {self.single_voxel_real_shape[axis]=}, " f"{self._config.resolution=}")
            if self.grid_shape[axis] % self.matrix_voxel_grid_shape[axis] != 0:
                raise Exception(
                    f"Due to discretization, matrix got skewered for {axis=}. "
                    f"{self.grid_shape=}, {self.matrix_voxel_grid_shape=}"
                )

        # init parameter transformations
        cur_specs = ParameterSpecs(
            shape=self.matrix_voxel_grid_shape,
            type=ParameterType.CONTINUOUS,
        )
        new_t_list = []
        for transform in self.param_transforms:
            t_new = transform.init_module(
                config=config,
                materials=self.materials,
                input_specs=cur_specs,
                matrix_voxel_grid_shape=self.matrix_voxel_grid_shape,
            )
            new_t_list.append(t_new)
            cur_specs = t_new._output_specs

        # set own input shape dtype
        self = self.aset("param_transforms", new_t_list)
        if self.output_type == ParameterType.CONTINUOUS and len(self.materials) != 2:
            raise Exception(
                f"Need exactly two materials in device when parameter mapping outputs continous permittivity indices, "
                f"but got {self.materials}"
            )
        return self

    def init_params(
        self,
        key: jax.Array,
    ) -> dict[str, jax.Array] | jax.Array:
        if len(self.param_transforms) > 0:
            shape_dtypes = self.param_transforms[0]._input_specs
        else:
            shape_dtypes = jax.ShapeDtypeStruct(
                self.matrix_voxel_grid_shape,
                jnp.float32,
            )
        if not isinstance(shape_dtypes, dict):
            shape_dtypes = {"dummy": shape_dtypes}
        params = {}
        for k, sd in shape_dtypes.items():
            key, subkey = jax.random.split(key)
            p = jax.random.uniform(
                key=subkey,
                shape=sd.shape,
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
        **module_kwargs,
    ) -> jax.Array:
        # walk through modules
        for transform in self.param_transforms:
            check_specs(params, transform._input_specs)
            params = transform(params, **module_kwargs)
            check_specs(params, transform._output_specs)
        if not isinstance(params, jax.Array):
            raise Exception(
                "The parameter mapping should return a single array of indices. If using a continous device, please"
                " make sure that the latent transformations abide to this rule."
            )
        if expand_to_sim_grid:
            params = expand_matrix(
                matrix=params,
                grid_points_per_voxel=self.single_voxel_grid_shape,
            )
        return params
