from abc import ABC, abstractmethod
from typing import Self

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field, frozen_private_field
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.core.misc import get_air_name
from fdtdx.materials import Material, compute_allowed_permittivities, compute_ordered_names
from fdtdx.objects.device.parameters.binary_transform import dilate_jax


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
    
    def _compute_input_shape_dtypes(
        self,
        output_shape_dtype: jax.ShapeDtypeStruct,
    ) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            shape=output_shape_dtype.shape,
            dtype=self._config.dtype,
        )
    
    @abstractmethod
    def __call__(
        self,
        input_params: dict[str, jax.Array] | jax.Array,
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
        input_params: dict[str, jax.Array] | jax.Array,
    ) -> jax.Array:
        if not isinstance(input_params, jax.Array):
            raise Exception(f"Closest Index cannot be used with latent parameters that contain multiple entries")
        arr = input_params
        allowed_inv_perms = 1 / jnp.asarray(compute_allowed_permittivities(self._material))
        dist = jnp.abs(arr[..., None] - allowed_inv_perms)
        discrete = jnp.argmin(dist, axis=-1)
        result = straight_through_estimator(arr, discrete)
        return result
        

@extended_autoinit
class BrushConstraint2D(Discretization):
    """Applies 2D brush-based constraints to ensure minimum feature sizes.

    Implements the brush-based constraint method described in:
    https://pubs.acs.org/doi/10.1021/acsphotonics.2c00313

    This ensures minimum feature sizes and connectivity in 2D designs by using
    morphological operations with a brush kernel.

    Attributes:
        brush: JAX array defining the brush kernel for morphological operations.
        axis: Axis along which to apply the 2D constraint (perpendicular plane).
    """

    brush: jax.Array = frozen_field()
    axis: int = frozen_field()

    def __call__(
        self,
        input_params: dict[str, jax.Array] | jax.Array,
    ) -> jax.Array:
        if not isinstance(input_params, jax.Array):
            raise Exception(f"BrushConstraint2D cannot be used with latent parameters that contain multiple entries")
        if len(self._material) > 2:
            raise Exception("BrushConstraint2D currently only implemented for single material and air")
        s = input_params.shape
        if len(s) != 3:
            raise Exception(
                f"BrushConstraint2D Generator can only work with 2D-Arrays, got {s=}"
            )
        if s[self.axis] != 1:
            raise Exception(
                f"BrushConstraint2D Generator needs array size 1 in axis, but got {s=}"
            )
        arr_2d = jnp.take(
            input_params,
            jnp.asarray(0),
            axis=self.axis,
        )

        cur_result = 1 - self._generator(arr_2d)

        air_name = get_air_name(self._material)
        ordered_name_list = compute_ordered_names(self._material)
        air_idx = ordered_name_list.index(air_name)
        if air_idx != 0:
            cur_result = 1 - cur_result
        cur_result = jnp.expand_dims(cur_result, axis=self.axis)
        result = straight_through_estimator(input_params, cur_result)
        return result

    def _generator(
        self,
        arr: jax.Array,
    ) -> jax.Array:
        touches_void = jnp.zeros_like(arr, dtype=jnp.bool)
        touches_solid = jnp.zeros_like(touches_void)

        def cond_fn(arrs):
            touch_v, touch_s = arrs[0], arrs[1]
            pixel_existing_solid = dilate_jax(touch_s, self.brush)
            pixel_existing_void = dilate_jax(touch_v, self.brush)
            return ~jnp.all(pixel_existing_solid | pixel_existing_void)

        def body_fn(sv_arrs: tuple[jax.Array, jax.Array]):
            # see Algorithm 1 in paper
            touch_v, touch_s = sv_arrs[0], sv_arrs[1]
            # compute touches and pixel arrays
            pixel_existing_solid = dilate_jax(touch_s, self.brush)
            pixel_existing_void = dilate_jax(touch_v, self.brush)
            touch_impossible_solid = dilate_jax(pixel_existing_void, self.brush)
            touch_impossible_void = dilate_jax(pixel_existing_solid, self.brush)
            touch_valid_solid = ~touch_impossible_solid & ~touch_s
            touch_valid_void = ~touch_impossible_void & ~touch_v
            pixel_possible_solid = dilate_jax(touch_s | touch_valid_solid, self.brush)
            pixel_possible_void = dilate_jax(touch_v | touch_valid_void, self.brush)
            pixel_required_solid = ~pixel_existing_solid & ~pixel_possible_void
            pixel_required_void = ~pixel_existing_void & ~pixel_possible_solid
            touch_resolving_solid = dilate_jax(pixel_required_solid, self.brush) & touch_valid_solid
            touch_resolving_void = dilate_jax(pixel_required_void, self.brush) & touch_valid_void
            touch_free_solid = ~dilate_jax(pixel_possible_void | pixel_existing_void, self.brush) & touch_valid_solid
            touch_free_void = ~dilate_jax(pixel_possible_solid | pixel_existing_solid, self.brush) & touch_valid_void

            # case 1
            def select_all_free_touches():
                new_v = touch_v | touch_free_void
                new_s = touch_s | touch_free_solid
                return new_v, new_s

            # case 2
            def select_best_resolving_touch():
                values_solid = jnp.where(touch_resolving_solid, arr, -jnp.inf)
                values_void = jnp.where(touch_resolving_void, -arr, -jnp.inf)

                def select_void():
                    max_idx = jnp.argmax(values_void)
                    new_v = touch_v.flatten().at[max_idx].set(True).reshape(touch_s.shape)
                    return new_v, touch_s

                def select_solid():
                    max_idx = jnp.argmax(values_solid)
                    new_s = touch_s.flatten().at[max_idx].set(True).reshape(touch_v.shape)
                    return touch_v, new_s

                return jax.lax.cond(
                    jnp.max(values_solid) > jnp.max(values_void),
                    select_solid,
                    select_void,
                )

            # case 3
            def select_best_valid_touch():
                values_solid = jnp.where(touch_valid_solid, arr, -jnp.inf)
                values_void = jnp.where(touch_valid_void, -arr, -jnp.inf)

                def select_void():
                    max_idx = jnp.argmax(values_void)
                    new_v = touch_v.flatten().at[max_idx].set(True).reshape(touch_s.shape)
                    return new_v, touch_s

                def select_solid():
                    max_idx = jnp.argmax(values_solid)
                    new_s = touch_s.flatten().at[max_idx].set(True).reshape(touch_v.shape)
                    return touch_v, new_s

                return jax.lax.cond(
                    jnp.max(values_solid) > jnp.max(values_void),
                    select_solid,
                    select_void,
                )

            # case 2 and 3
            def case_2_and_3_function():
                resolving_exists = jnp.any(touch_resolving_solid | touch_resolving_void)

                return jax.lax.cond(
                    resolving_exists,
                    select_best_resolving_touch,
                    select_best_valid_touch,
                )

            free_touches_exist = jnp.any(touch_free_solid | touch_free_void)
            new_v, new_s = jax.lax.cond(
                free_touches_exist,
                select_all_free_touches,
                case_2_and_3_function,
            )
            return new_v, new_s

        arrs = (touches_void, touches_solid)

        res_arrs = eqxi.while_loop(
            cond_fun=cond_fn,
            body_fun=body_fn,
            init_val=arrs,
            kind="lax",
        )
        pixel_existing_solid = dilate_jax(res_arrs[1], self.brush)
        return pixel_existing_solid


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