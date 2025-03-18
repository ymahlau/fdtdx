from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, field, frozen_field, frozen_private_field
from fdtdx.materials import ContinuousMaterialRange, Material
from fdtdx.typing import (
    INVALID_SHAPE_3D,
    UNDEFINED_SHAPE_3D,
    GridShape3D, 
    PartialGridShape3D, 
    PartialRealShape3D,
    RealShape3D, 
    SliceTuple3D
)
from fdtdx.core.misc import expand_matrix, is_float_divisible
from fdtdx.core.plotting.colors import PINK
from fdtdx.objects.object import OrderableObject
from fdtdx.objects.device.parameters.mapping import DiscreteParameterMapping


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

    name: str = frozen_field(default=None, kind="KW_ONLY")  # type: ignore
    material: dict[str, Material] | ContinuousMaterialRange = frozen_field(kind="KW_ONLY")  # type: ignore
    color: tuple[float, float, float] = frozen_field(default=PINK)
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
    def init_params(
        self,
        key: jax.Array,
    ) -> dict[str, jax.Array] | jax.Array:
        """Initializes optimization parameters for the device.

        Args:
            key: JAX random key for parameter initialization

        Returns:
            Dictionary mapping parameter names to their initial values
        """
        raise NotImplementedError()
        

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
    
    def init_params(
        self,
        key: jax.Array,
    ) -> dict[str, jax.Array] | jax.Array:
        """Initializes optimization parameters for the device.

        Creates random initial parameters between 0 and 1 for each input shape
        defined in the constraint mapping interface.

        Args:
            key: JAX random key for parameter initialization

        Returns:
            Dictionary mapping parameter names to their initial values
        """
        shape_dtypes = self.parameter_mapping._input_shape_dtypes
        if not isinstance(shape_dtypes, dict):
            shape_dtypes = {'dummy': shape_dtypes}
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


# @extended_autoinit
# class ContinuousDevice(ContinuousMultiMaterialObject, BaseDeviceInterface):
#     """A device that supports continuous transitions between materials.

#     This class represents a simulation object whose permittivity distribution can be
#     optimized with continuous transitions between different materials. The permittivity
#     values are controlled by parameters that are mapped through constraints to produce
#     the final device structure with smooth material transitions.

#     Attributes:
#         name: Optional name identifier for the device
#         dtype: Data type for device parameters, defaults to float32
#         color: RGB color tuple for visualization, defaults to pink
#     """

#     color: tuple[float, float, float] = frozen_field(default=PINK)
#     constraint_mapping: ConstraintMapping = frozen_private_field()
    
#     def get_voxel_mask_for_shape(self) -> jax.Array:
#         return jnp.ones(shape=self.matrix_voxel_grid_shape, dtype=jnp.bool)

#     def place_on_grid(
#         self: Self,
#         grid_slice_tuple: SliceTuple3D,
#         config: SimulationConfig,
#         key: jax.Array,
#     ) -> Self:
#         """Places the device on the simulation grid and initializes constraints.

#         Args:
#             grid_slice_tuple: Tuple of slices defining device position on grid
#             config: Simulation configuration parameters
#             key: JAX random key for initialization

#         Returns:
#             Self with updated grid position and initialized constraint mapping
#         """
#         self = super().place_on_grid(
#             grid_slice_tuple=grid_slice_tuple,
#             config=config,
#             key=key,
#         )

#         # Create constraint mapping for continuous material transitions
#         constraint_mapping = ConstraintMapping(
#             modules=[
#                 StandardToInversePermittivityRange(),
#                 ContinuousPermittivityTransition(),
#             ]
#         )

#         # Initialize the constraint mapping
#         mapping = constraint_mapping.init_modules(
#             config=config,
#             materials=self.materials,
#             output_interface=ConstraintInterface(
#                 shapes={"out": self.matrix_voxel_grid_shape},
#                 type="inv_permittivity",
#             ),
#         )
#         self = self.aset("constraint_mapping", mapping)
#         return self

#     def init_params(
#         self,
#         key: jax.Array,
#         initial_values: dict[str, jax.Array] | None = None,
#     ) -> dict[str, jax.Array]:
#         """Initializes optimization parameters for the device.

#         Creates parameters either from provided initial values or randomly between 0 and 1
#         for each input shape defined in the constraint mapping interface.

#         Args:
#             key: JAX random key for parameter initialization
#             initial_values: Optional dictionary of initial parameter values

#         Returns:
#             Dictionary mapping parameter names to their initial values

#         Raises:
#             Exception: If initial_values shapes don't match required shapes
#         """
#         shapes = self.constraint_mapping._input_interface.shapes

#         # If initial values are provided, validate and use them
#         if initial_values is not None:
#             for k, s in shapes.items():
#                 if k not in initial_values:
#                     raise Exception(f"Missing initial value for parameter {k}")
#                 if initial_values[k].shape != s:
#                     raise Exception(
#                         f"Shape mismatch for parameter {k}: " f"expected {s}, got {initial_values[k].shape}"
#                     )
#             return initial_values

#         # Otherwise initialize randomly
#         params = {}
#         for k, s in shapes.items():
#             key, subkey = jax.random.split(key)
#             p = jax.random.uniform(
#                 key=subkey,
#                 shape=s,
#                 minval=0,  # parameter always live between 0 and 1
#                 maxval=1,
#                 dtype=self._config.dtype,
#             )
#             params[k] = p
#         return params

#     def set_params_from_array(
#         self,
#         array: jax.Array,
#     ) -> dict[str, jax.Array]:
#         """Sets device parameters from a JAX array.

#         Converts the input array into the required parameter format. The array values
#         should be in the range [0,1] as they will be mapped to permittivity values
#         through the constraint mapping.

#         Args:
#             array: JAX array with values in [0,1] range matching the device's matrix shape

#         Returns:
#             Dictionary containing the array as parameters in the required format

#         Raises:
#             Exception: If array shape doesn't match device's matrix shape
#         """
#         expected_shape = self.matrix_voxel_grid_shape
#         if array.shape != expected_shape:
#             raise Exception(f"Array shape {array.shape} doesn't match required shape {expected_shape}")

#         # Ensure values are in [0,1] range
#         array = jnp.clip(array, 0.0, 1.0)

#         # Convert to parameter dictionary format
#         params = {"out": array.astype(self._config.dtype)}

#         return params

#     def get_inv_permittivity(
#         self,
#         prev_inv_permittivity: jax.Array,
#         params: dict[str, jax.Array] | None,
#     ) -> tuple[jax.Array, dict]:
#         """Computes inverse permittivity distribution from optimization parameters.

#         Maps the optimization parameters through constraints to produce the final
#         inverse permittivity distribution for the device with continuous material
#         transitions.

#         Args:
#             prev_inv_permittivity: Previous inverse permittivity values (unused)
#             params: Dictionary of optimization parameters, cannot be None

#         Returns:
#             Tuple containing:
#                 - Array of inverse permittivity values
#                 - Dictionary with additional computation info

#         Raises:
#             Exception: If params is None
#         """
#         del prev_inv_permittivity
#         if params is None:
#             raise Exception("Device params cannot be None")
#         quantized_array = self.constraint_mapping(params)
#         extended_params = expand_matrix(
#             matrix=quantized_array,
#             grid_points_per_voxel=self.single_voxel_grid_shape,
#             add_channels=False,
#         )
#         return extended_params, {}




###################################################################################
    # def get_inv_permittivity(
    #     self,
    #     prev_inv_permittivity: jax.Array,
    #     params: dict[str, jax.Array] | None,
    # ) -> tuple[jax.Array, dict]:
    #     """Computes inverse permittivity distribution from optimization parameters.

    #     Maps the optimization parameters through constraints to produce the final
    #     inverse permittivity distribution for the device.

    #     Args:
    #         prev_inv_permittivity: Previous inverse permittivity values (unused)
    #         params: Dictionary of optimization parameters, cannot be None

    #     Returns:
    #         Tuple containing:
    #             - Array of inverse permittivity values
    #             - Dictionary with additional computation info

    #     Raises:
    #         Exception: If params is None
    #     """
    #     del prev_inv_permittivity
    #     if params is None:
    #         raise Exception("Device params cannot be None")
    #     quantized_array = self.constraint_mapping(params)
    #     extended_params = expand_matrix(
    #         matrix=quantized_array,
    #         grid_points_per_voxel=self.single_voxel_grid_shape,
    #         add_channels=False,
    #     )
    #     return extended_params, {}

    
    # def get_indices(
    #     self,
    #     params: dict[str, jax.Array],
    # ) -> jax.Array:
    #     """Computes material indices from optimization parameters.

    #     Maps parameters through constraints to determine which allowed permittivity
    #     value is used at each point in the device.

    #     Args:
    #         params: Dictionary of optimization parameters

    #     Returns:
    #         Array of indices into allowed_inverse_permittivities
    #     """
    #     quantized_array = self.constraint_mapping(params)[..., None]
    #     index_mask = quantized_array == self.allowed_inverse_permittivities
    #     raw_indices = jnp.arange(len(self.allowed_inverse_permittivities))[None, None, None, :]
    #     indices_3d = (index_mask * raw_indices).sum(axis=-1)
    #     return indices_3d