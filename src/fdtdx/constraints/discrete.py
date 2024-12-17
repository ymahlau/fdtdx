import math

import equinox.internal as eqxi
import jax
import jax.numpy as jnp

from fdtdx.constraints.binary_transform import (
    binary_median_filter,
    connect_holes_and_structures,
    dilate_jax,
    remove_floating_polymer,
)
from fdtdx.constraints.module import ConstraintInterface, ConstraintModule
from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.core.misc import PaddingConfig, get_air_name


@extended_autoinit
class RemoveFloatingMaterial(ConstraintModule):
    """Finds all material that floats in the air and sets their permittivity to air.

    This constraint module identifies regions of material that are not connected to any
    substrate or boundary and converts them to air. This helps ensure physically
    realizable designs by eliminating floating/disconnected material regions.

    The module only works with binary material systems (2 permittivities) where one
    material represents air.
    """

    def transform(
        self,
        input_params: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Transforms input parameters by removing floating material regions.

        Args:
            input_params: Dictionary mapping parameter names to their values as JAX arrays.
                Expected to contain a single array with material indices.

        Returns:
            Dictionary with same structure as input but with floating material regions
            converted to air index.

        Raises:
            NotImplementedError: If more than 2 permittivities are specified.
            Exception: If input contains more than one array.
        """
        if len(self._allowed_permittivities) != 2:
            raise NotImplementedError("Remove floating material currently only implemented for single material")
        air_name = get_air_name(self._permittivity_config)
        air_idx = self._permittivity_names.index(air_name)
        arr = list(input_params.values())[0]
        is_material_matrix = arr != air_idx
        result = remove_floating_polymer(is_material_matrix)
        # result = result.astype(arr.dtype)
        result = straight_through_estimator(arr, result)
        return {list(input_params.keys())[0]: result}

    def input_interface(
        self,
        output_interface: ConstraintInterface,
    ) -> ConstraintInterface:
        """Validates and processes the input interface specification.

        Args:
            output_interface: Interface specification from the next module in the chain.

        Returns:
            The validated interface specification.

        Raises:
            Exception: If output interface has multiple shapes or wrong type.
            NotImplementedError: If more than 2 permittivities are specified.
        """
        if len(output_interface.shapes) != 1:
            raise Exception(
                f"RemoveFloatingMaterial expects a single array as output, but got {output_interface.shapes}"
            )
        if output_interface.type != "index":
            raise Exception("RemoveFloatingMaterial can only be followed by a module using indices")
        if len(self._allowed_permittivities) != 2:
            raise NotImplementedError("RemoveFloatingMaterial currently only implemented for single material")
        return output_interface


@extended_autoinit
class ConnectHolesAndStructures(ConstraintModule):
    """Connects floating polymer regions and ensures air holes connect to outside.

    This constraint module ensures physical realizability of designs by:
    1. Either connecting floating polymer regions to the substrate or removing them
    2. Ensuring all air holes are connected to the outside (no trapped air)

    The bottom (lower z) is treated as the substrate reference.

    Attributes:
        fill_material: Name of material to use for filling gaps when connecting regions.
            Required when working with more than 2 materials.
    """

    fill_material: str | None = frozen_field(default=None)

    def input_interface(
        self,
        output_interface: ConstraintInterface,
    ) -> ConstraintInterface:
        if len(self._allowed_permittivities) > 2 and self.fill_material is None:
            raise Exception(
                "ConnectHolesAndStructures: Need to specify fill material when working with more than a single material"
            )
        if output_interface.type != "index":
            raise Exception("ConnectHolesAndStructures can only be followed by a module using indices")
        return output_interface

    def transform(
        self,
        input_params: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        arr = list(input_params.values())[0]
        air_name = get_air_name(self._permittivity_config)
        air_idx = self._permittivity_names.index(air_name)
        is_material_matrix = arr != air_idx
        feasible_material_matrix = connect_holes_and_structures(is_material_matrix)

        result = jnp.empty_like(arr)
        # set air
        result = jnp.where(
            feasible_material_matrix,
            -1,  # this is set below
            air_idx,
        )
        # material where previously was material
        result = jnp.where(feasible_material_matrix & is_material_matrix, arr, result)

        # material, where previously was air
        fill_name = self.fill_material
        if fill_name is None:
            if len(self._allowed_permittivities) > 2:
                raise Exception("This should never happen")
            fill_name = self._permittivity_names[1 - air_idx]
        fill_idx = self._permittivity_names.index(fill_name)
        result = jnp.where(
            feasible_material_matrix & ~is_material_matrix,
            fill_idx,
            result,
        )
        return {list(input_params.keys())[0]: result}


def circular_brush(
    diameter: float,
    size: int | None = None,
) -> jax.Array:
    """Creates a circular binary mask/brush for morphological operations.

    Args:
        diameter: Diameter of the circle in grid units.
        size: Optional size of the output array. If None, uses ceil(diameter) rounded
            up to next odd number.

    Returns:
        Binary JAX array containing a circular mask where True indicates points
        within the circle diameter.
    """
    if size is None:
        s = math.ceil(diameter)
        if s % 2 == 0:
            s += 1
        size = s
    xy = jnp.stack(jnp.meshgrid(*map(jnp.arange, (size, size)), indexing="xy"), axis=-1) - jnp.asarray((size / 2) - 0.5)
    euc_dist = jnp.sqrt((xy**2).sum(axis=-1))
    # the less EQUAL here is important, because otherwise design may be infeasible
    # due to discretization errors
    mask = euc_dist <= (diameter / 2)
    return mask


@extended_autoinit
class BrushConstraint2D(ConstraintModule):
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

    def transform(
        self,
        input_params: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        result = {}
        for k, v in input_params.items():
            arr_2d = jnp.take(
                v,
                jnp.asarray(0),
                axis=self.axis,
            )

            # with jax.disable_jit():
            cur_result = 1 - self._generator(arr_2d)

            air_name = get_air_name(self._permittivity_config)
            air_idx = self._permittivity_names.index(air_name)
            if air_idx != 0:
                cur_result = 1 - cur_result
            cur_result = jnp.expand_dims(cur_result, axis=self.axis)
            result[k] = straight_through_estimator(v, cur_result)
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
        return pixel_existing_solid.astype(jnp.int32)

    def input_interface(
        self,
        output_interface: ConstraintInterface,
    ) -> ConstraintInterface:
        if len(self._allowed_permittivities) > 2:
            raise Exception("BrushConstraint2D currently only implemented for single material and air")
        if output_interface.type != "index":
            raise Exception("BrushConstraint2D has to be followed by a module using indices")
        for s in output_interface.shapes.values():
            if len(s) != 3:
                raise Exception(
                    f"BrushConstraint2D Generator can only work with 2D-Arrays, got {output_interface.shapes=}"
                )
            if s[self.axis] != 1:
                raise Exception(
                    f"BrushConstraint2D Generator needs array size 1 in axis, but got {output_interface.shapes=}"
                )
        return ConstraintInterface(
            type="latent",
            shapes=output_interface.shapes,
        )


BOTTOM_Z_PADDING_CONFIG_REPEAT = PaddingConfig(
    modes=("edge", "edge", "edge", "edge", "constant", "edge"), widths=(20,), values=(1,)
)

BOTTOM_Z_PADDING_CONFIG = PaddingConfig(
    modes=("constant", "constant", "constant", "constant", "constant", "constant"),
    widths=(10,),
    values=(1, 0, 1, 1, 1, 0),
)


@extended_autoinit
class BinaryMedianFilterModule(ConstraintModule):
    """Performs 3D binary median filtering on the design.

    Applies a 3D median filter to smooth and clean up binary material distributions.
    This helps remove small features and noise while preserving larger structures.

    Attributes:
        padding_cfg: Configuration for padding behavior at boundaries.
        kernel_sizes: 3-tuple of kernel sizes for each dimension.
        num_repeats: Number of times to apply the filter consecutively.
    """

    padding_cfg: PaddingConfig = frozen_field()
    kernel_sizes: tuple[int, int, int] = frozen_field()
    num_repeats: int = frozen_field(default=1)

    def input_interface(
        self,
        output_interface: ConstraintInterface,
    ) -> ConstraintInterface:
        if output_interface.type != "index":
            raise Exception("BinaryMedianFilterModule has to be followed by a module using indices")
        if len(self._allowed_permittivities) != 2:
            raise Exception("BinaryMedianFilterModule only works for single materials")
        for s in output_interface.shapes.values():
            if len(s) != 3:
                raise Exception(f"Parameter shape has to have 3 dims, got {output_interface.shapes=}")
        return output_interface

    def transform(
        self,
        input_params: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        result = {}
        for k, v in input_params.items():
            cur_arr = v
            for _ in range(self.num_repeats):
                cur_arr = binary_median_filter(
                    arr_3d=cur_arr,
                    kernel_sizes=self.kernel_sizes,
                    padding_cfg=self.padding_cfg,
                )
            result[k] = straight_through_estimator(v, cur_arr)
        return result
