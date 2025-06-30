from typing import Sequence

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field
from fdtdx.objects.device.parameters.transform import SameShapeTypeParameterTransform
from fdtdx.typing import ParameterType


def tanh_projection(x: jax.Array, beta: float, eta: float) -> jax.Array:
    """
    Adapted from the meep repository:
    https://github.com/NanoComp/meep/blob/master/python/adjoint/filters.py

    Sigmoid projection filter.
    Ref: F. Wang, B. S. Lazarov, & O. Sigmund, On projection methods,
    convergence and robust formulations in topology optimization.
    Structural and Multidisciplinary Optimization, 43(6), pp. 767-784 (2011).

    Args:
        x (jax.Array): design weights to be filtered.
        beta (float): thresholding parameter in the range [0, inf]. Determines the
            degree of binarization of the output.
        eta (float): threshold point in range [0, 1]

    Returns:
        jax.Array: The filtered design weights.
    """

    def beta_inf_case():
        # Note that backpropagating through here can produce NaNs. So we
        # manually specify the step function to keep the gradient clean.
        return jnp.where(x > eta, 1.0, 0.0)

    def beta_zero_case():
        # this is mathematically not really accurate, but makes sense for optimization
        return jnp.clip(x, 0, 1)

    def other_case():
        dividend = jnp.tanh(beta * eta) + jnp.tanh(beta * (x - eta))
        divisor = jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta))
        return dividend / divisor

    index = (beta == 0) + 2 * ((beta != 0) & ~jnp.isinf(beta))
    result = jax.lax.switch(index, (beta_inf_case, beta_zero_case, other_case))
    return result


def smoothed_projection(
    rho_filtered: jax.Array,
    beta: float,
    eta: float,
    resolution: float,
) -> jax.Array:
    """
    This function is adapted from the Meep repository:
    https://github.com/NanoComp/meep/blob/master/python/adjoint/filters.py

    The details of this projection are described in the paper by Alec Hammond:
    https://arxiv.org/pdf/2503.20189

    Project using subpixel smoothing, which allows for β→∞.
    This technique integrates out the discontinuity within the projection
    function, allowing the user to smoothly increase β from 0 to ∞ without
    losing the gradient. Effectively, a level set is created, and from this
    level set, first-order subpixel smoothing is applied to the interfaces (if
    any are present).

    In order for this to work, the input array must already be smooth (e.g. by
    filtering).

    While the original approach involves numerical quadrature, this approach
    performs a "trick" by assuming that the user is always infinitely projecting
    (β=∞). In this case, the expensive quadrature simplifies to an analytic
    fill-factor expression. When to use this fill factor requires some careful
    logic.

    For one, we want to make sure that the user can indeed project at any level
    (not just infinity). So in these cases, we simply check if in interface is
    within the pixel. If not, we revert to the standard filter plus project
    technique.

    If there is an interface, we want to make sure the derivative remains
    continuous both as the interface leaves the cell, *and* as it crosses the
    center. To ensure this, we need to account for the different possibilities.

    Args:
        rho_filtered (jax.Array): The (2D) input design parameters (already filered e.g.
            with a conic filter).
        beta (float): The thresholding parameter in the range [0, inf]. Determines the
            degree of binarization of the output.
        eta (float): The threshold point in the range [0, 1].
        resolution (float): resolution of the design grid (not the Meep grid resolution).
    Returns:
        jax.Array: The projected and smoothed output.

    Example:
        >>> Lx = 2; Ly = 2
        >>> resolution = 50
        >>> eta_i = 0.5; eta_e = 0.75
        >>> lengthscale = 0.1
        >>> filter_radius = get_conic_radius_from_eta_e(lengthscale, eta_e)
        >>> Nx = onp.round(Lx * resolution) + 1
        >>> Ny = onp.round(Ly * resolution) + 1
        >>> rho = onp.random.rand(Nx, Ny)
        >>> beta = npa.inf
        >>> rho_filtered = conic_filter(rho, filter_radius, Lx, Ly, resolution)
        >>> rho_projected = smoothed_projection(rho_filtered, beta, eta_i, resolution)
    """
    # sanity checks
    assert rho_filtered.ndim == 2
    # Note that currently, the underlying assumption is that the smoothing
    # kernel is a circle, which means dx = dy.
    dx = dy = 1 / resolution
    R_smoothing = 0.55 * dx

    rho_projected = tanh_projection(rho_filtered, beta=beta, eta=eta)

    # Compute the spatial gradient (using finite differences) of the *filtered*
    # field, which will always be smooth and is the key to our approach. This
    # gradient essentially represents the normal direction pointing the the
    # nearest inteface.
    rho_filtered_grad = jnp.gradient(rho_filtered)
    rho_filtered_grad_helper = (rho_filtered_grad[0] / dx) ** 2 + (rho_filtered_grad[1] / dy) ** 2

    # Note that a uniform field (norm=0) is problematic, because it creates
    # divide by zero issues and makes backpropagation difficult, so we sanitize
    # and determine where smoothing is actually needed. The value where we don't
    # need smoothings doesn't actually matter, since all our computations our
    # purely element-wise (no spatial locality) and those pixels will instead
    # rely on the standard projection. So just use 1, since it's well behaved.
    nonzero_norm = jnp.abs(rho_filtered_grad_helper) > 0

    rho_filtered_grad_norm = jnp.sqrt(jnp.where(nonzero_norm, rho_filtered_grad_helper, 1))
    rho_filtered_grad_norm_eff = jnp.where(nonzero_norm, rho_filtered_grad_norm, 1)

    # The distance for the center of the pixel to the nearest interface
    d = (eta - rho_filtered) / rho_filtered_grad_norm_eff

    # Only need smoothing if an interface lies within the voxel. Since d is
    # actually an "effective" d by this point, we need to ignore values that may
    # have been sanitized earlier on.
    needs_smoothing = nonzero_norm & (jnp.abs(d) < R_smoothing)

    # The fill factor is used to perform simple, first-order subpixel smoothing.
    # We use the (2D) analytic expression that comes when assuming the smoothing
    # kernel is a circle. Note that because the kernel contains some
    # expressions that are sensitive to NaNs, we have to use the "double where"
    # trick to avoid the Nans in the backward trace. This is a common problem
    # with array-based AD tracers, apparently. See here:
    # https://github.com/google/jax/issues/1052#issuecomment-5140833520
    d_R = d / R_smoothing
    F = jnp.where(needs_smoothing, 0.5 - 15 / 16 * d_R + 5 / 8 * d_R**3 - 3 / 16 * d_R**5, 1.0)
    # F(-d)
    F_minus = jnp.where(needs_smoothing, 0.5 + 15 / 16 * d_R - 5 / 8 * d_R**3 + 3 / 16 * d_R**5, 1.0)

    # Determine the upper and lower bounds of materials in the current pixel (before projection).
    rho_filtered_minus = rho_filtered - R_smoothing * rho_filtered_grad_norm_eff * F
    rho_filtered_plus = rho_filtered + R_smoothing * rho_filtered_grad_norm_eff * F_minus

    # Finally, we project the extents of our range.
    rho_minus_eff_projected = tanh_projection(rho_filtered_minus, beta=beta, eta=eta)
    rho_plus_eff_projected = tanh_projection(rho_filtered_plus, beta=beta, eta=eta)

    # Only apply smoothing to interfaces
    rho_projected_smoothed = (1 - F) * rho_minus_eff_projected + F * rho_plus_eff_projected
    return jnp.where(
        needs_smoothing,
        rho_projected_smoothed,
        rho_projected,
    )


@autoinit
class TanhProjection(SameShapeTypeParameterTransform):
    """
    Tanh projection filter.

    This needs the steepness parameter $\\beta$ as a keyword-argument in
    apply_params

    Ref: F. Wang, B. S. Lazarov, & O. Sigmund, On projection methods,
    convergence and robust formulations in topology optimization.
    Structural and Multidisciplinary Optimization, 43(6), pp. 767-784 (2011).

    Attributes:
        projection_midpoint (float, optional): Midpoint of the TanhProjection. Defaults to 0.5.

    Notes:
        The call method requires a beta parameter as a keyword argument passed to the parameter transformation
    """

    projection_midpoint: float = frozen_field(default=0.5)

    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=ParameterType.CONTINUOUS
    )

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        if "beta" not in kwargs:
            raise Exception("TanhProjection needs the beta parameter as additional keyword argument!")
        beta = kwargs["beta"]

        result = {}
        for k, v in params.items():
            result[k] = tanh_projection(v, beta, self.projection_midpoint)

        return result


@autoinit
class SubpixelSmoothedProjection(SameShapeTypeParameterTransform):
    """
    This function is adapted from the Meep repository:
    https://github.com/NanoComp/meep/blob/master/python/adjoint/filters.py

    The details of this projection are described in the paper by Alec Hammond:
    https://arxiv.org/pdf/2503.20189

    Project using subpixel smoothing, which allows for β→∞.
    This technique integrates out the discontinuity within the projection
    function, allowing the user to smoothly increase β from 0 to ∞ without
    losing the gradient. Effectively, a level set is created, and from this
    level set, first-order subpixel smoothing is applied to the interfaces (if
    any are present).

    In order for this to work, the input array must already be smooth (e.g. by
    filtering).

    While the original approach involves numerical quadrature, this approach
    performs a "trick" by assuming that the user is always infinitely projecting
    (β=∞). In this case, the expensive quadrature simplifies to an analytic
    fill-factor expression. When to use this fill factor requires some careful
    logic.

    For one, we want to make sure that the user can indeed project at any level
    (not just infinity). So in these cases, we simply check if in interface is
    within the pixel. If not, we revert to the standard filter plus project
    technique.

    If there is an interface, we want to make sure the derivative remains
    continuous both as the interface leaves the cell, *and* as it crosses the
    center. To ensure this, we need to account for the different possibilities.

    Attributes:
        projection_midpoint (float, optional): midpoint of the tanh projection. Defaults to 0.5

    Notes:
        The call method requires a beta parameter as a keyword argument passed to the parameter transformation
    """

    projection_midpoint: float = frozen_field(default=0.5)

    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=ParameterType.CONTINUOUS
    )
    _check_single_array: bool = frozen_private_field(default=True)
    _all_arrays_2d: bool = frozen_private_field(default=True)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        if "beta" not in kwargs:
            raise Exception("SubpixelSmoothedProjection needs the beta parameter as additional keyword argument!")
        beta = kwargs["beta"]

        result = {}
        for k, v in params.items():
            # shape sanity checks
            vertical_axis = v.shape.index(1)
            first_axis = 0 if vertical_axis != 0 else 1
            second_axis = 2 if vertical_axis != 2 else 1
            if self._single_voxel_size[first_axis] != self._single_voxel_size[second_axis]:
                raise Exception(
                    "SubpixelSmoothedProjection expects voxel size to be equal in "
                    f"two axes, but got {self._single_voxel_size}"
                )
            voxel_size = self._single_voxel_size[first_axis]
            v_2d = v.squeeze(vertical_axis)

            result_2d = smoothed_projection(
                v_2d,
                beta=beta,
                eta=self.projection_midpoint,
                # expects resolution as pixels / µm
                resolution=1 / (voxel_size / 1e-6),
            )
            result[k] = jnp.expand_dims(result_2d, vertical_axis)

        return result
