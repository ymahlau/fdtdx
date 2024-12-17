from abc import ABC, abstractmethod
from typing import Literal, Self

import jax
import jax.numpy as jnp

from fdtdx.core.config import SimulationConfig
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field, frozen_private_field
from fdtdx.core.jax.ste import straight_through_estimator


@extended_autoinit
class ConstraintInterface(ExtendedTreeClass):
    """Interface specification for constraint module inputs/outputs.

    Defines the type and shapes of arrays that a constraint module accepts or produces.

    Attributes:
        type: The type of constraint interface - one of:
            - "latent": Raw latent parameters
            - "index": Discrete material indices
            - "inv_permittivity": Inverse permittivity values
        shapes: Dictionary mapping array names to their expected shapes
    """

    type: Literal["latent", "index", "inv_permittivity"] = frozen_field()
    shapes: dict[str, tuple[int, ...]] = frozen_field()


def check_interface_compliance(
    arrays: dict[str, jax.Array],
    interface: ConstraintInterface,
):
    """Validates that arrays match the expected interface specification.

    Args:
        arrays: Dictionary of arrays to validate
        interface: Interface specification to validate against

    Raises:
        Exception: If array keys don't match interface or shapes don't match
    """
    for k, arr in arrays.items():
        if k not in interface.shapes:
            raise Exception(
                f"Keys Differ, Interface Error: \n{[f'{k}: {a.shape}' for k, a in arrays.items()]}, {interface=}"
            )
        shape = interface.shapes[k]
        if arr.shape != shape:
            raise Exception(f"Wrong shape: Expected {shape=}, not {arr.shape=}")


@extended_autoinit
class ConstraintModule(ExtendedTreeClass, ABC):
    """Abstract base class for constraint modules.

    Constraint modules transform parameters between different representations while
    enforcing physical and fabrication constraints. They form a chain of transformations
    from latent parameters to final inverse permittivity values.

    Attributes:
        _permittivity_config: Dictionary mapping material names to permittivity values
        _config: Global simulation configuration
        _output_interface: Interface specification for module outputs
        _input_interface: Interface specification for module inputs
    """

    _permittivity_config: dict[str, float] = frozen_private_field()
    _config: SimulationConfig = frozen_private_field()
    _output_interface: ConstraintInterface = frozen_private_field()
    _input_interface: ConstraintInterface = frozen_private_field()

    @property
    def _ordered_permittivity_tuples(self) -> list[tuple[str, float]]:
        kv = list(self._permittivity_config.items())
        kv_sorted = sorted(kv, key=lambda x: x[1])
        return kv_sorted

    @property
    def _allowed_permittivities(self) -> jax.Array:
        name_val_list = self._ordered_permittivity_tuples
        perms = jnp.asarray([v[1] for v in name_val_list], dtype=jnp.float32)
        return perms

    @property
    def _allowed_inverse_permittivities(self):
        return 1.0 / self._allowed_permittivities

    @property
    def _permittivity_names(self) -> list[str]:
        name_val_list = self._ordered_permittivity_tuples
        names = [v[0] for v in name_val_list]
        return names

    @abstractmethod
    def transform(
        self,
        input_params: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        del input_params
        raise NotImplementedError()

    @abstractmethod
    def input_interface(
        self,
        output_interface: ConstraintInterface,
    ) -> ConstraintInterface:
        del output_interface
        raise NotImplementedError()

    def init_module(
        self: Self,
        config: SimulationConfig,
        permittivity_config: dict[str, float],
        output_interface: ConstraintInterface,
    ) -> Self:
        self = self.aset("_config", config)
        self = self.aset("_permittivity_config", permittivity_config)
        self = self.aset("_output_interface", output_interface)
        input_interface = self.input_interface(self._output_interface)
        self = self.aset("_input_interface", input_interface)
        return self


@extended_autoinit
class StandardToInversePermittivityRange(ConstraintModule):
    """Maps standard [0,1] range to inverse permittivity range.

    Linearly maps values from [0,1] to the range between minimum and maximum
    inverse permittivity values allowed by the material configuration.
    """

    def transform(
        self,
        input_params: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        max_inv_perm = self._allowed_inverse_permittivities.max()
        min_inv_perm = self._allowed_inverse_permittivities.min()

        result = {}
        for k, v in input_params.items():
            mapped = v * (max_inv_perm - min_inv_perm) + min_inv_perm
            result[k] = mapped
        return result

    def input_interface(
        self,
        output_interface: ConstraintInterface,
    ) -> ConstraintInterface:
        if output_interface.type != "latent":
            raise Exception("Range Conversion only works on latent Parameters!")
        return output_interface


@extended_autoinit
class StandardToCustomRange(ConstraintModule):
    """Maps standard [0,1] range to custom range [min_value, max_value].

    Linearly maps values from [0,1] to a custom range specified by min_value
    and max_value parameters.

    Attributes:
        min_value: Minimum value of target range
        max_value: Maximum value of target range
    """

    min_value: float = frozen_field(default=0)
    max_value: float = frozen_field(default=1)

    def transform(
        self,
        input_params: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        result = {}
        for k, v in input_params.items():
            mapped = v * (self.max_value - self.min_value) + self.min_value
            result[k] = mapped
        return result

    def input_interface(
        self,
        output_interface: ConstraintInterface,
    ) -> ConstraintInterface:
        if output_interface.type != "latent":
            raise Exception("Range Conversion only works on latent Parameters!")
        return output_interface


@extended_autoinit
class StandardToPlusOneMinusOneRange(StandardToCustomRange):
    """Maps standard [0,1] range to [-1,1] range.

    Special case of StandardToCustomRange that maps to [-1,1] range.
    Used for symmetric value ranges around zero.

    Attributes:
        min_value: Fixed to -1
        max_value: Fixed to 1
    """

    min_value: float = frozen_field(default=-1, init=False)
    max_value: float = frozen_field(default=1, init=False)


@extended_autoinit
class ClosestIndex(ConstraintModule):
    """Maps continuous values to nearest allowed material indices.

    For each input value, finds the index of the closest allowed inverse
    permittivity value. Uses straight-through gradient estimation to maintain
    differentiability.
    """

    def transform(
        self,
        input_params: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        result = {}
        for k, v in input_params.items():
            dist = jnp.abs(v[..., None] - self._allowed_inverse_permittivities)
            discrete = jnp.argmin(dist, axis=-1)
            result[k] = straight_through_estimator(v, discrete)
        return result

    def input_interface(
        self,
        output_interface: ConstraintInterface,
    ) -> ConstraintInterface:
        if output_interface.type != "index":
            raise Exception("After ClosestIndex a Module using indices has to follow!")
        return ConstraintInterface(
            type="latent",
            shapes=output_interface.shapes,
        )


@extended_autoinit
class IndicesToInversePermittivities(ConstraintModule):
    """Maps material indices to their inverse permittivity values.

    Converts discrete material indices into their corresponding inverse
    permittivity values from the allowed materials list. Uses straight-through
    gradient estimation to maintain differentiability.
    """

    def transform(
        self,
        input_params: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        result = {}
        for k, v in input_params.items():
            out = self._allowed_inverse_permittivities[v.astype(jnp.int32)]
            out = out.astype(self._config.dtype)
            result[k] = straight_through_estimator(v, out)
        return result

    def input_interface(
        self,
        output_interface: ConstraintInterface,
    ) -> ConstraintInterface:
        if output_interface.type != "inv_permittivity":
            raise Exception(
                "After IndicesToInversePermittivities can only follow a module using" "Inverse permittivities"
            )
        return ConstraintInterface(
            type="index",
            shapes=output_interface.shapes,
        )
