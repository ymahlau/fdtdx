from abc import ABC, abstractmethod
from typing import Self

import jax

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import extended_autoinit, field, frozen_field
from fdtdx.core.jax.typing import (
    INVALID_SHAPE_3D,
    UNDEFINED_SHAPE_3D,
    GridShape3D,
    PartialGridShape3D,
    PartialRealShape3D,
    RealShape3D,
    SliceTuple3D,
)
from fdtdx.core.misc import is_float_divisible
from fdtdx.materials import ContinuousMaterialRange, Material
from fdtdx.objects.object import SimulationObject


@extended_autoinit
class MultiMaterialObject(SimulationObject, ABC):
    """A simulation object with configurable material permittivities.

    This class represents objects with multiple permittivity values arranged in a grid.
    It handles voxelization of the object onto the simulation grid and provides methods
    to access and manipulate the permittivity distribution.

    Attributes:
        materials: Dictionary mapping material names to material values.
        partial_voxel_grid_shape: Shape of voxels in grid coordinates (optional).
        partial_voxel_real_shape: Shape of voxels in real coordinates (optional).
        _single_voxel_grid_shape: Internal shape of a single voxel in grid coordinates.
    """
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
    
    
    @abstractmethod
    def get_voxel_mask_for_shape(self) -> jax.Array:
        """Get a binary mask of the objects shape. Everything voxel not in the mask, will not be updated by
        this object. For example, can be used to approximate a round shape.
        The mask is calculated in device voxel size, not in simulation voxels.

        Returns:
            jax.Array: Binary mask representing the voxels occupied by the object
        """
        raise NotImplementedError()



@extended_autoinit
class DiscreteMultiMaterialObject(MultiMaterialObject, ABC):
    materials: dict[str, Material] = frozen_field(kind="KW_ONLY")  # type: ignore
    
    @abstractmethod
    def get_material_mapping(self) -> jax.Array:
        """Returns an array, which represents the material index at every voxel. Specifically, it returns the 
        index of the ordered material list.

        Returns:
            jax.Array: Index array
        """
        raise NotImplementedError()


@extended_autoinit
class ContinuousMultiMaterialObject(MultiMaterialObject, ABC):
    material_range: ContinuousMaterialRange = frozen_field(kind="KW_ONLY")  # type:ignore
    
    @abstractmethod
    def get_material_mapping(self) -> jax.Array:
        """Returns an array, which represents the material at every voxel. Specifically, it returns a value between 
        0 and 1 representing the linear interpolation between the start and end material of the continous material
        range.

        Returns:
            jax.Array: Index array
        """
        raise NotImplementedError()

