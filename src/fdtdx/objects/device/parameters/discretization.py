import math
from typing import Literal, Self, Sequence

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from loguru import logger

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.core.misc import get_background_material_name
from fdtdx.materials import Material, compute_allowed_permittivities, compute_ordered_names
from fdtdx.objects.device.parameters.binary_transform import dilate_jax
from fdtdx.objects.device.parameters.transform import ParameterTransformation
from fdtdx.objects.device.parameters.utils import compute_allowed_indices, nearest_index
from fdtdx.typing import ParameterType


@autoinit
class ClosestIndex(ParameterTransformation):
    """
    Maps continuous latent values to nearest allowed material indices.

    For each input value, finds the index of the closest allowed inverse
    permittivity value. Uses straight-through gradient estimation to maintain
    differentiability. If mapping_from_inverse_permittivities is set to False (default),
    then the transform only quantizes the latent parameters to the closest integer value.
    """

    mapping_from_inverse_permittivities: bool = frozen_field(default=False)
    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=ParameterType.CONTINUOUS
    )

    def _get_input_shape_impl(
        self,
        output_shape: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        return output_shape

    def _get_output_type_impl(
        self,
        input_type: dict[str, ParameterType],
    ) -> dict[str, ParameterType]:
        if len(self._materials) <= 1:
            raise Exception(f"Invalid materials (need two or more): {self._materials}")
        elif len(self._materials) == 2:
            output_type = ParameterType.BINARY
        else:
            output_type = ParameterType.DISCRETE
        result = {k: output_type for k in input_type.keys()}
        return result

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs

        def transform_arr(arr: jax.Array) -> jax.Array:
            if self.mapping_from_inverse_permittivities:
                allowed_inv_perms = 1 / jnp.asarray(compute_allowed_permittivities(self._materials))
                dist = jnp.abs(arr[..., None] - allowed_inv_perms)
                discrete = jnp.argmin(dist, axis=-1)
            else:
                discrete = jnp.clip(jnp.round(arr), 0, len(self._materials) - 1)
            return straight_through_estimator(arr, discrete)

        result = {}
        for k, v in params.items():
            result[k] = transform_arr(v)
        return result


@autoinit
class BrushConstraint2D(ParameterTransformation):
    """Applies 2D brush-based constraints to ensure minimum feature sizes.

    Implements the brush-based constraint method described in:
    https://pubs.acs.org/doi/10.1021/acsphotonics.2c00313

    This ensures minimum feature sizes and connectivity in 2D designs by using
    morphological operations with a brush kernel.

    Attributes:
        brush (jax.Array): Array defining the brush kernel for morphological operations.
        axis (int): Axis along which to apply the 2D constraint (perpendicular plane).
        background_material (str | None, optional): Name of the background material in the material
            dictionary of the device. If None, the material with the lowest permittivity
            is used. Defaults to None.
    """

    brush: jax.Array = frozen_field()
    axis: int = frozen_field()
    background_material: str | None = frozen_field(default=None)

    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=ParameterType.CONTINUOUS
    )
    _check_single_array: bool = frozen_private_field(default=True)
    _all_arrays_2d: bool = frozen_private_field(default=True)

    def _get_input_shape_impl(
        self,
        output_shape: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        return output_shape

    def _get_output_type_impl(
        self,
        input_type: dict[str, ParameterType],
    ) -> dict[str, ParameterType]:
        if len(self._materials) != 2:
            raise Exception(
                f"BrushConstraint2D currently only implemented for exactly two materials, but got {self._materials}"
            )
        return {k: ParameterType.BINARY for k in input_type.keys()}

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs

        single_key = list(params.keys())[0]
        param_arr = params[single_key]
        s = param_arr.shape
        if s[self.axis] != 1:
            raise Exception(f"BrushConstraint2D Generator needs array size 1 in axis, but got {s=}")
        arr_2d = jnp.take(
            param_arr,
            jnp.asarray(0),
            axis=self.axis,
        )

        if self.background_material is None:
            background_name = get_background_material_name(self._materials)
        else:
            background_name = self.background_material

        ordered_name_list = compute_ordered_names(self._materials)
        background_idx = ordered_name_list.index(background_name)
        if background_idx != 0:
            arr_2d = -arr_2d
        cur_result = self._generator(arr_2d)
        if background_idx != 0:
            cur_result = 1 - cur_result

        cur_result = jnp.expand_dims(cur_result, axis=self.axis)
        result = straight_through_estimator(param_arr, cur_result)
        return {single_key: result}

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


def circular_brush(
    diameter: float,
    size: int | None = None,
) -> jax.Array:
    """Creates a circular binary mask/brush for morphological operations.

    Args:
        diameter (float): Diameter of the circle in grid units.
        size (int | None, optional): Optional size of the output array. If None, uses ceil(diameter) rounded
            up to next odd number.

    Returns:
        jax.Array: Binary array containing a circular mask where True indicates points
        within the circle diameter.
    """
    if size is None:
        s = math.ceil(diameter)
        if s % 2 == 0:
            s += 1
        size = s
    xy = jnp.stack(jnp.meshgrid(*map(jnp.arange, (size, size)), indexing="xy"), axis=-1) - jnp.asarray((size / 2) - 0.5)
    euc_dist = jnp.sqrt((xy**2).sum(axis=-1))
    # the less EQUAL here is important, because otherwise design may be infeasible due to discretization errors
    mask = euc_dist <= (diameter / 2)
    return mask


@autoinit
class PillarDiscretization(ParameterTransformation):
    """Constraint module for mapping pillar structures to allowed configurations.

    Maps arbitrary pillar structures to the nearest allowed configurations based on
    material constraints and geometry requirements. Ensures structures meet fabrication
    rules like single polymer columns and no trapped air holes.

    Attributes:
        axis (int): Axis along which to enforce pillar constraints (0=x, 1=y, 2=z).
        single_polymer_columns (bool): If True, restrict to single polymer columns.
        distance_metric (Literal["euclidean", "permittivity_differences_plus_average_permittivity"], optional):
            Method to compute distances between material distributions:
            - "euclidean": Standard Euclidean distance between permittivity values.
            - "permittivity_differences_plus_average_permittivity": Weighted combination
              of permittivity differences and average permittivity values, optimized
              for material distribution comparisons.
            Defaults to "permittivity_differences_plus_average_permittivity".
        background_material (str | None, optional):
            Name of the background material in the materials dictionary of the corresponding device.
            If None, the material with lowest permittivity is used. Defaults to None.
    """

    axis: int = frozen_field()
    single_polymer_columns: bool = frozen_field()
    distance_metric: Literal["euclidean", "permittivity_differences_plus_average_permittivity"] = frozen_field(
        default="permittivity_differences_plus_average_permittivity",
    )
    background_material: str | None = frozen_field(default=None)

    _allowed_indices: jax.Array = frozen_private_field()
    _check_single_array: bool = frozen_private_field(default=True)
    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=ParameterType.CONTINUOUS
    )

    def _get_input_shape_impl(
        self,
        output_shape: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        return output_shape

    def _get_output_type_impl(
        self,
        input_type: dict[str, ParameterType],
    ) -> dict[str, ParameterType]:
        if len(self._materials) <= 1:
            raise Exception(f"Invalid materials (need two or more): {self._materials}")
        elif len(self._materials) == 2:
            output_type = ParameterType.BINARY
        else:
            output_type = ParameterType.DISCRETE
        return {k: output_type for k in input_type.keys()}

    def init_module(
        self: Self,
        config: SimulationConfig,
        materials: dict[str, Material],
        matrix_voxel_grid_shape: tuple[int, int, int],
        single_voxel_size: tuple[float, float, float],
        output_shape: dict[str, tuple[int, ...]],
    ) -> Self:
        self = super().init_module(
            config=config,
            materials=materials,
            matrix_voxel_grid_shape=matrix_voxel_grid_shape,
            single_voxel_size=single_voxel_size,
            output_shape=output_shape,
        )

        if self.background_material is None:
            background_name = get_background_material_name(self._materials)
        else:
            background_name = self.background_material
        ordered_name_list = compute_ordered_names(self._materials)
        background_idx = ordered_name_list.index(background_name)

        allowed_columns = compute_allowed_indices(
            num_layers=matrix_voxel_grid_shape[self.axis],
            indices=list(range(len(materials))),
            fill_holes_with_index=[background_idx],
            single_polymer_columns=self.single_polymer_columns,
        )
        self = self.aset("_allowed_indices", allowed_columns, create_new_ok=True)
        logger.info(f"{allowed_columns=}")
        logger.info(f"{allowed_columns.shape=}")
        return self

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs

        single_key = list(params.keys())[0]
        params_arr = params[single_key]

        allowed_inv_perms = 1 / jnp.asarray(compute_allowed_permittivities(self._materials))
        nearest_allowed_index = nearest_index(
            values=params_arr,
            allowed_values=allowed_inv_perms,
            axis=self.axis,
            distance_metric=self.distance_metric,
            allowed_indices=self._allowed_indices,
            return_distances=False,
        )
        result_index = self._allowed_indices[nearest_allowed_index]
        if self.axis == 2:
            pass  # no transposition needed
        elif self.axis == 1:
            result_index = jnp.transpose(result_index, axes=(0, 2, 1))
        elif self.axis == 0:
            result_index = jnp.transpose(result_index, axes=(2, 0, 1))
        else:
            raise Exception(f"invalid axis: {self.axis}")
        result_index = straight_through_estimator(params_arr, result_index)
        return {single_key: result_index}
