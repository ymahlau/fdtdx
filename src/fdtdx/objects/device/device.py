from abc import ABC
from typing import Self

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import extended_autoinit, field, frozen_field
from fdtdx.core.misc import expand_matrix, is_float_divisible
from fdtdx.core.plotting.colors import PINK
from fdtdx.materials import ContinuousMaterialRange, Material
from fdtdx.objects.device.parameters.mapping import DiscreteParameterMapping, LatentParameterMapping
from fdtdx.objects.object import OrderableObject
from fdtdx.typing import (
    INVALID_SHAPE_3D,
    UNDEFINED_SHAPE_3D,
    GridShape3D,
    PartialGridShape3D,
    PartialRealShape3D,
    RealShape3D,
    SliceTuple3D,
)


@extended_autoinit
class BaseDevice(OrderableObject, ABC):
    """Abstract base class for devices with optimizable permittivity distributions.

    This class defines the common interface and functionality for both discrete and
    continuous devices that can be optimized through gradient-based methods.

    Attributes:
        name: Optional name identifier for the device
        dtype: Data type for device parameters, defaults to float32
        color: RGB color tuple for visualization, defaults to pink
    """

    material: dict[str, Material] | ContinuousMaterialRange = frozen_field(kind="KW_ONLY")  # type: ignore
    color: tuple[float, float, float] = frozen_field(default=PINK)
    parameter_mapping: DiscreteParameterMapping | LatentParameterMapping = frozen_field(kind="KW_ONLY")  # type:ignore
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

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        """Place the multi-material object on the simulation grid.

        Validates and sets up the voxel grid dimensions based on the provided
        partial grid or real shape specifications. Ensures the voxel sizes are
        compatible with the grid resolution.

        Args:
            grid_slice_tuple: Tuple of slices defining object position on grid.
            config: Simulation configuration object.
            key: JAX random key.

        Returns:
            Self: Updated instance with validated grid placement.

        Raises:
            Exception: If voxel specifications are invalid or incompatible with grid.
        """
        self = super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)

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
        return self

    def init_params(
        self,
        key: jax.Array,
    ) -> dict[str, jax.Array] | jax.Array:
        shape_dtypes = self.parameter_mapping._input_shape_dtypes
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
                dtype=sd.dtype,
            )
            params[k] = p
        if len(params) == 1:
            params = list(params.values())[0]
        return params

    def get_material_mapping(
        self,
        params: dict[str, jax.Array] | jax.Array,
    ) -> jax.Array:
        indices = self.parameter_mapping(params)
        if not isinstance(indices, jax.Array):
            raise Exception(
                "The parameter mapping should return a single array of indices. If using a continous device, please"
                " make sure that the latent transformations abide to this rule."
            )
        return indices

    def get_expanded_material_mapping(
        self,
        params: dict[str, jax.Array] | jax.Array,
    ) -> jax.Array:
        indices = self.get_material_mapping(params)
        expanded_indices = expand_matrix(
            matrix=indices,
            grid_points_per_voxel=self.single_voxel_grid_shape,
        )
        return expanded_indices


@extended_autoinit
class DiscreteDevice(BaseDevice):
    """A device with discrete material states.

    This class represents a simulation object whose permittivity distribution can be
    optimized through gradient-based methods, with discrete transitions between materials.
    The permittivity values are controlled by parameters that are mapped through constraints
    to produce the final device structure.

    Attributes:
        name: Optional name identifier for the device
        constraint_mapping: Maps optimization parameters to permittivity values
        dtype: Data type for device parameters, defaults to float32
        color: RGB color tuple for visualization, defaults to pink
    """

    material: dict[str, Material] = frozen_field(kind="KW_ONLY")  # type: ignore
    parameter_mapping: DiscreteParameterMapping = frozen_field(kind="KW_ONLY")  # type:ignore

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        mapping = self.parameter_mapping.init_modules(
            config=config,
            material=self.material,
            output_shape_dtype=jax.ShapeDtypeStruct(
                shape=self.matrix_voxel_grid_shape,
                dtype=jnp.int32,
            ),
        )
        self = self.aset("parameter_mapping", mapping)
        return self


@extended_autoinit
class ContinuousDevice(BaseDevice):
    material: ContinuousMaterialRange = frozen_field(kind="KW_ONLY")  # type: ignore
    parameter_mapping: LatentParameterMapping = frozen_field(
        kind="KW_ONLY",
        default=LatentParameterMapping(latent_transforms=[]),
    )

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        mapping = self.parameter_mapping.init_modules(
            config=config,
            material=self.material,
            output_shape_dtypes=jax.ShapeDtypeStruct(
                shape=self.matrix_voxel_grid_shape,
                dtype=self._config.dtype,
            ),
        )
        self = self.aset("parameter_mapping", mapping)
        return self

