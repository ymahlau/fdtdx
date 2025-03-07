from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.constraints.mapping import ConstraintMapping
from fdtdx.constraints.module import (
    ConstraintInterface,
    ContinuousPermittivityTransition,
    StandardToInversePermittivityRange,
)
from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field, frozen_private_field
from fdtdx.core.jax.typing import SliceTuple3D
from fdtdx.core.misc import expand_matrix
from fdtdx.core.plotting.colors import PINK
from fdtdx.objects.multi_material.multi_material import MultiMaterial


@extended_autoinit
class BaseDevice(MultiMaterial, ABC):
    """Abstract base class for devices with optimizable permittivity distributions.

    This class defines the common interface and functionality for both discrete and
    continuous devices that can be optimized through gradient-based methods.

    Attributes:
        name: Optional name identifier for the device
        dtype: Data type for device parameters, defaults to float32
        color: RGB color tuple for visualization, defaults to pink
    """

    name: str = frozen_field(default=None, kind="KW_ONLY")  # type: ignore
    dtype: jnp.dtype = frozen_field(
        default=jnp.float32,
        kind="KW_ONLY",
    )
    color: tuple[float, float, float] = frozen_field(default=PINK)

    @abstractmethod
    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        """Places the device on the simulation grid and initializes constraints.

        Args:
            grid_slice_tuple: Tuple of slices defining device position on grid
            config: Simulation configuration parameters
            key: JAX random key for initialization

        Returns:
            Self with updated grid position and initialized constraint mapping
        """
        return super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )

    @abstractmethod
    def init_params(
        self,
        key: jax.Array,
    ) -> dict[str, jax.Array]:
        """Initializes optimization parameters for the device.

        Args:
            key: JAX random key for parameter initialization

        Returns:
            Dictionary mapping parameter names to their initial values
        """
        pass

    @abstractmethod
    def get_inv_permittivity(
        self,
        prev_inv_permittivity: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:
        """Computes inverse permittivity distribution from optimization parameters.

        Args:
            prev_inv_permittivity: Previous inverse permittivity values
            params: Dictionary of optimization parameters

        Returns:
            Tuple containing:
                - Array of inverse permittivity values
                - Dictionary with additional computation info
        """
        pass

    def get_inv_permeability(
        self,
        prev_inv_permeability: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:
        """Returns unchanged inverse permeability values.

        Device optimization only modifies permittivity, not permeability.

        Args:
            prev_inv_permeability: Previous inverse permeability values
            params: Device parameters (unused)

        Returns:
            Tuple containing:
                - Unchanged inverse permeability array
                - Empty info dictionary
        """
        del params
        return prev_inv_permeability, {}


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

    constraint_mapping: ConstraintMapping = frozen_field(kind="KW_ONLY")  # type: ignore

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        """Places the device on the simulation grid and initializes constraints.

        Args:
            grid_slice_tuple: Tuple of slices defining device position on grid
            config: Simulation configuration parameters
            key: JAX random key for initialization

        Returns:
            Self with updated grid position and initialized constraint mapping
        """
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        mapping = self.constraint_mapping.init_modules(
            config=config,
            permittivity_config=self.permittivity_config,
            output_interface=ConstraintInterface(
                shapes={"out": self.matrix_voxel_grid_shape},
                type="inv_permittivity",
            ),
        )
        self = self.aset("constraint_mapping", mapping)
        return self

    def init_params(
        self,
        key: jax.Array,
    ) -> dict[str, jax.Array]:
        """Initializes optimization parameters for the device.

        Creates random initial parameters between 0 and 1 for each input shape
        defined in the constraint mapping interface.

        Args:
            key: JAX random key for parameter initialization

        Returns:
            Dictionary mapping parameter names to their initial values
        """
        shapes = self.constraint_mapping._input_interface.shapes
        params = {}
        for k, s in shapes.items():
            key, subkey = jax.random.split(key)
            p = jax.random.uniform(
                key=subkey,
                shape=s,
                minval=0,  # parameter always live between 0 and 1
                maxval=1,
                dtype=self._config.dtype,
            )
            params[k] = p
        return params

    def get_inv_permittivity(
        self,
        prev_inv_permittivity: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:
        """Computes inverse permittivity distribution from optimization parameters.

        Maps the optimization parameters through constraints to produce the final
        inverse permittivity distribution for the device.

        Args:
            prev_inv_permittivity: Previous inverse permittivity values (unused)
            params: Dictionary of optimization parameters, cannot be None

        Returns:
            Tuple containing:
                - Array of inverse permittivity values
                - Dictionary with additional computation info

        Raises:
            Exception: If params is None
        """
        del prev_inv_permittivity
        if params is None:
            raise Exception("Device params cannot be None")
        quantized_array = self.constraint_mapping(params)
        extended_params = expand_matrix(
            matrix=quantized_array,
            grid_points_per_voxel=self.single_voxel_grid_shape,
            add_channels=False,
        )
        return extended_params, {}

    def get_indices(
        self,
        params: dict[str, jax.Array],
    ) -> jax.Array:
        """Computes material indices from optimization parameters.

        Maps parameters through constraints to determine which allowed permittivity
        value is used at each point in the device.

        Args:
            params: Dictionary of optimization parameters

        Returns:
            Array of indices into allowed_inverse_permittivities
        """
        quantized_array = self.constraint_mapping(params)[..., None]
        index_mask = quantized_array == self.allowed_inverse_permittivities
        raw_indices = jnp.arange(len(self.allowed_inverse_permittivities))[None, None, None, :]
        indices_3d = (index_mask * raw_indices).sum(axis=-1)
        return indices_3d


@extended_autoinit
class ContinuousDevice(BaseDevice):
    """A device that supports continuous transitions between materials.

    This class represents a simulation object whose permittivity distribution can be
    optimized with continuous transitions between different materials. The permittivity
    values are controlled by parameters that are mapped through constraints to produce
    the final device structure with smooth material transitions.

    Attributes:
        name: Optional name identifier for the device
        dtype: Data type for device parameters, defaults to float32
        color: RGB color tuple for visualization, defaults to pink
    """

    color: tuple[float, float, float] = frozen_field(default=PINK)
    constraint_mapping: ConstraintMapping = frozen_private_field()

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        """Places the device on the simulation grid and initializes constraints.

        Args:
            grid_slice_tuple: Tuple of slices defining device position on grid
            config: Simulation configuration parameters
            key: JAX random key for initialization

        Returns:
            Self with updated grid position and initialized constraint mapping
        """
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )

        # Create constraint mapping for continuous material transitions
        constraint_mapping = ConstraintMapping(
            modules=[
                StandardToInversePermittivityRange(),
                ContinuousPermittivityTransition(),
            ]
        )

        # Initialize the constraint mapping
        mapping = constraint_mapping.init_modules(
            config=config,
            permittivity_config=self.permittivity_config,
            output_interface=ConstraintInterface(
                shapes={"out": self.matrix_voxel_grid_shape},
                type="inv_permittivity",
            ),
        )
        self = self.aset("constraint_mapping", mapping)
        return self

    def init_params(
        self,
        key: jax.Array,
        initial_values: dict[str, jax.Array] | None = None,
    ) -> dict[str, jax.Array]:
        """Initializes optimization parameters for the device.

        Creates parameters either from provided initial values or randomly between 0 and 1
        for each input shape defined in the constraint mapping interface.

        Args:
            key: JAX random key for parameter initialization
            initial_values: Optional dictionary of initial parameter values

        Returns:
            Dictionary mapping parameter names to their initial values

        Raises:
            Exception: If initial_values shapes don't match required shapes
        """
        shapes = self.constraint_mapping._input_interface.shapes

        # If initial values are provided, validate and use them
        if initial_values is not None:
            for k, s in shapes.items():
                if k not in initial_values:
                    raise Exception(f"Missing initial value for parameter {k}")
                if initial_values[k].shape != s:
                    raise Exception(
                        f"Shape mismatch for parameter {k}: " f"expected {s}, got {initial_values[k].shape}"
                    )
            return initial_values

        # Otherwise initialize randomly
        params = {}
        for k, s in shapes.items():
            key, subkey = jax.random.split(key)
            p = jax.random.uniform(
                key=subkey,
                shape=s,
                minval=0,  # parameter always live between 0 and 1
                maxval=1,
                dtype=self._config.dtype,
            )
            params[k] = p
        return params

    def set_params_from_array(
        self,
        array: jax.Array,
    ) -> dict[str, jax.Array]:
        """Sets device parameters from a JAX array.

        Converts the input array into the required parameter format. The array values
        should be in the range [0,1] as they will be mapped to permittivity values
        through the constraint mapping.

        Args:
            array: JAX array with values in [0,1] range matching the device's matrix shape

        Returns:
            Dictionary containing the array as parameters in the required format

        Raises:
            Exception: If array shape doesn't match device's matrix shape
        """
        expected_shape = self.matrix_voxel_grid_shape
        if array.shape != expected_shape:
            raise Exception(f"Array shape {array.shape} doesn't match required shape {expected_shape}")

        # Ensure values are in [0,1] range
        array = jnp.clip(array, 0.0, 1.0)

        # Convert to parameter dictionary format
        params = {"out": array.astype(self._config.dtype)}

        return params

    def get_inv_permittivity(
        self,
        prev_inv_permittivity: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:
        """Computes inverse permittivity distribution from optimization parameters.

        Maps the optimization parameters through constraints to produce the final
        inverse permittivity distribution for the device with continuous material
        transitions.

        Args:
            prev_inv_permittivity: Previous inverse permittivity values (unused)
            params: Dictionary of optimization parameters, cannot be None

        Returns:
            Tuple containing:
                - Array of inverse permittivity values
                - Dictionary with additional computation info

        Raises:
            Exception: If params is None
        """
        del prev_inv_permittivity
        if params is None:
            raise Exception("Device params cannot be None")
        quantized_array = self.constraint_mapping(params)
        extended_params = expand_matrix(
            matrix=quantized_array,
            grid_points_per_voxel=self.single_voxel_grid_shape,
            add_channels=False,
        )
        return extended_params, {}
