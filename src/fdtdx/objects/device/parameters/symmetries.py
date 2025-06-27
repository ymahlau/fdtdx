import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field
from fdtdx.objects.device.parameters.transform import SameShapeTypeParameterTransform


@autoinit
class DiagonalSymmetry2D(SameShapeTypeParameterTransform):
    """
    Enforce symmetries by effectively havling the parameter space. The symmetry is transposing by rotating the image
    and taking the mean of original and transpose.

    Attributes:
        min_min_to_max_max (bool): if true, the symmetry axes is from (x_min, y_min) to (x_max, y_max).
            If false, the other diagonal is used.
    """

    min_min_to_max_max: bool = frozen_field()

    _all_arrays_2d: bool = frozen_private_field(default=True)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        result = {}
        for k, v in params.items():
            # convert to 2d
            vertical_axis = v.shape.index(1)
            v_2d = v.squeeze(vertical_axis)

            # enforce symmetry
            if self.min_min_to_max_max:
                other = v_2d.T
            else:
                other = v_2d[::-1, ::-1].T
            cur_mean = (v_2d + other) / 2

            # expand dims again
            result[k] = jnp.expand_dims(cur_mean, vertical_axis)
        return result
