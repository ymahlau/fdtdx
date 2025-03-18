from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_private_field
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.materials import Material, compute_allowed_permittivities


class Discretization(ExtendedTreeClass, ABC):
    _material: dict[str, Material] = frozen_private_field()
    _config: SimulationConfig = frozen_private_field()
    _output_shape_dtype: jax.ShapeDtypeStruct = frozen_private_field()
    _input_shape_dtypes: dict[str, jax.ShapeDtypeStruct] = frozen_private_field()
    
    def init_module(
        self: Self,
        config: SimulationConfig,
        material: dict[str, Material],
        output_shape_dtype: jax.ShapeDtypeStruct,
    ) -> Self:
        self = self.aset("_config", config)
        self = self.aset("_material", material)
        input_shape_dtypes = self._compute_input_shape_dtypes(output_shape_dtype)
        self = self.aset("_output_shape_dtype", output_shape_dtype)
        self = self.aset("_input_shape_dtypes", input_shape_dtypes)
        return self
    
    @abstractmethod
    def _compute_input_shape_dtypes(
        self,
        output_shape_dtype: jax.ShapeDtypeStruct,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        raise NotImplementedError()
    
    @abstractmethod
    def __call__(
        self,
        values: dict[str, jax.Array],
    ) -> jax.Array:
        raise NotImplementedError()


@extended_autoinit
class ClosestIndex(Discretization):
    """Maps continuous latent values to nearest allowed material indices.

    For each input value, finds the index of the closest allowed inverse
    permittivity value. Uses straight-through gradient estimation to maintain
    differentiability.
    """

    def __call__(
        self,
        input_params: dict[str, jax.Array],
    ) -> jax.Array:
        if len(input_params) != 1:
            raise Exception(f"Closest Index cannot be used with latent parameters that contain multiple entries")
        arr = input_params['latent']
        allowed_inv_perms = 1 / jnp.asarray(compute_allowed_permittivities(self._material))
        dist = jnp.abs(arr[..., None] - allowed_inv_perms)
        discrete = jnp.argmin(dist, axis=-1)
        result = straight_through_estimator(arr, discrete)
        return result
    
    def _compute_input_shape_dtypes(
        self,
        output_shape_dtype: jax.ShapeDtypeStruct,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        return {
            'latent': jax.ShapeDtypeStruct(
                shape=output_shape_dtype,
                dtype=self._config.dtype,
            )
        }
        
        
# @extended_autoinit
# class IndicesToInversePermittivities(ConstraintModule):
#     """Maps material indices to their inverse permittivity values.

#     Converts discrete material indices into their corresponding inverse
#     permittivity values from the allowed materials list. Uses straight-through
#     gradient estimation to maintain differentiability.
#     """

#     def transform(
#         self,
#         input_params: dict[str, jax.Array],
#     ) -> dict[str, jax.Array]:
#         result = {}
#         for k, v in input_params.items():
#             out = self._allowed_inverse_permittivities[v.astype(jnp.int32)]
#             out = out.astype(self._config.dtype)
#             result[k] = straight_through_estimator(v, out)
#         return result

#     def input_interface(
#         self,
#         output_interface: ConstraintInterface,
#     ) -> ConstraintInterface:
#         if output_interface.type != "inv_permittivity":
#             raise Exception(
#                 "After IndicesToInversePermittivities can only follow a module using" "Inverse permittivities"
#             )
#         return ConstraintInterface(
#             type="index",
#             shapes=output_interface.shapes,
#         )