import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field
from fdtdx.objects.device.parameters.transform import SameShapeTypeParameterTransform


@autoinit
class DiagonalSymmetry2D(SameShapeTypeParameterTransform):
    """
    Enforce diagonal symmetry by effectively halving the parameter space.
    The symmetry is achieved by transposing the image and taking the mean
    of the original and transpose.

    This creates a design that is symmetric across one of the two diagonals.
    """

    #: If true, the symmetry axis is from (x_min, y_min) to (x_max, y_max).
    #: If false, the other diagonal (from (x_min, y_max) to (x_max, y_min)) is used.
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


@autoinit
class HorizontalSymmetry2D(SameShapeTypeParameterTransform):
    """
    Enforce horizontal (x-axis) mirror symmetry.

    This creates a design that is symmetric across a vertical line through
    the center, i.e., the left half mirrors the right half.
    The symmetry is enforced by averaging the array with its horizontally
    flipped version.
    """

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

            # enforce symmetry: flip along x-axis (axis 0)
            flipped = v_2d[::-1, :]
            cur_mean = (v_2d + flipped) / 2

            # expand dims again
            result[k] = jnp.expand_dims(cur_mean, vertical_axis)
        return result


@autoinit
class VerticalSymmetry2D(SameShapeTypeParameterTransform):
    """
    Enforce vertical (y-axis) mirror symmetry.

    This creates a design that is symmetric across a horizontal line through
    the center, i.e., the top half mirrors the bottom half.
    The symmetry is enforced by averaging the array with its vertically
    flipped version.
    """

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

            # enforce symmetry: flip along y-axis (axis 1)
            flipped = v_2d[:, ::-1]
            cur_mean = (v_2d + flipped) / 2

            # expand dims again
            result[k] = jnp.expand_dims(cur_mean, vertical_axis)
        return result


@autoinit
class PointSymmetry2D(SameShapeTypeParameterTransform):
    """
    Enforce 180-degree rotational (point) symmetry.

    This creates a design that is symmetric under 180-degree rotation about
    its center. The symmetry is enforced by averaging the array with its
    180-degree rotated version.
    """

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

            # enforce symmetry: 180-degree rotation (flip both axes)
            rotated = v_2d[::-1, ::-1]
            cur_mean = (v_2d + rotated) / 2

            # expand dims again
            result[k] = jnp.expand_dims(cur_mean, vertical_axis)
        return result


# =============================================================================
# 3D Symmetry Transforms
# =============================================================================


@autoinit
class HorizontalSymmetry3D(SameShapeTypeParameterTransform):
    """
    Enforce horizontal mirror symmetry in 3D along the x or y axis.

    This creates a design that is symmetric across a plane perpendicular to
    the specified axis. The symmetry is enforced by averaging the array with
    its flipped version along the chosen axis.
    """

    #: The axis to mirror across. Can be 'x' (axis 0) or 'y' (axis 1).
    #: Defaults to 'x'.
    mirror_axis: str = frozen_field(default="x")

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        # Determine which axis to flip
        if self.mirror_axis == "x":
            axis = 0
        elif self.mirror_axis == "y":
            axis = 1
        else:
            raise ValueError(f"mirror_axis must be 'x' or 'y', got '{self.mirror_axis}'")

        result = {}
        for k, v in params.items():
            # enforce symmetry: flip along the specified axis
            flipped = jnp.flip(v, axis=axis)
            cur_mean = (v + flipped) / 2
            result[k] = cur_mean
        return result


@autoinit
class VerticalSymmetry3D(SameShapeTypeParameterTransform):
    """
    Enforce vertical (z-axis) mirror symmetry in 3D.

    This creates a design that is symmetric across a horizontal plane
    perpendicular to the z-axis (axis 2). The top half mirrors the bottom half.
    The symmetry is enforced by averaging the array with its vertically
    flipped version.
    """

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        result = {}
        for k, v in params.items():
            # enforce symmetry: flip along z-axis (axis 2)
            flipped = jnp.flip(v, axis=2)
            cur_mean = (v + flipped) / 2
            result[k] = cur_mean
        return result


@autoinit
class PointSymmetry3D(SameShapeTypeParameterTransform):
    """
    Enforce 180-degree rotational (point) symmetry in 3D.

    This creates a design that is symmetric under 180-degree rotation about
    the center point of the volume. The symmetry is enforced by averaging
    the array with its fully reversed version (flipped along all three axes).
    """

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        result = {}
        for k, v in params.items():
            # enforce symmetry: 180-degree rotation (flip all axes)
            rotated = v[::-1, ::-1, ::-1]
            cur_mean = (v + rotated) / 2
            result[k] = cur_mean
        return result


@autoinit
class DiagonalSymmetry3D(SameShapeTypeParameterTransform):
    """
    Enforce diagonal symmetry in 3D across one of six possible diagonal planes.

    The diagonal planes are defined by which two axes are swapped (transposed):
    - 'xy': Diagonal in the xy-plane (swaps x and y, z unchanged)
    - 'xz': Diagonal in the xz-plane (swaps x and z, y unchanged)
    - 'yz': Diagonal in the yz-plane (swaps y and z, x unchanged)

    For each plane, there are two diagonals controlled by `min_min_to_max_max`:
    - True: The diagonal from (min, min) to (max, max) in that plane
    - False: The anti-diagonal from (min, max) to (max, min) in that plane

    Note: The two dimensions being swapped must be equal in size.
    """

    #: The plane in which the diagonal lies. One of 'xy', 'xz', or 'yz'.
    #: Defaults to 'xy' for backwards compatibility.
    diagonal_plane: str = frozen_field(default="xy")

    #: If true, the symmetry is across the main diagonal (min,min → max,max).
    #: If false, the anti-diagonal (min,max → max,min) is used.
    min_min_to_max_max: bool = frozen_field(default=True)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs

        # Determine transpose axes and flip axes based on diagonal_plane
        if self.diagonal_plane == "xy":
            # Swap x (0) and y (1), keep z (2)
            transpose_axes = (1, 0, 2)
            flip_axes = (0, 1)  # Flip x and y for anti-diagonal
        elif self.diagonal_plane == "xz":
            # Swap x (0) and z (2), keep y (1)
            transpose_axes = (2, 1, 0)
            flip_axes = (0, 2)  # Flip x and z for anti-diagonal
        elif self.diagonal_plane == "yz":
            # Swap y (1) and z (2), keep x (0)
            transpose_axes = (0, 2, 1)
            flip_axes = (1, 2)  # Flip y and z for anti-diagonal
        else:
            raise ValueError(f"diagonal_plane must be 'xy', 'xz', or 'yz', got '{self.diagonal_plane}'")

        result = {}
        for k, v in params.items():
            if self.min_min_to_max_max:
                # Main diagonal: just transpose
                other = jnp.transpose(v, axes=transpose_axes)
            else:
                # Anti-diagonal: flip both relevant axes, then transpose
                flipped = jnp.flip(jnp.flip(v, axis=flip_axes[0]), axis=flip_axes[1])
                other = jnp.transpose(flipped, axes=transpose_axes)
            cur_mean = (v + other) / 2
            result[k] = cur_mean
        return result
