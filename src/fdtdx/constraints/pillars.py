from typing import Literal, Self

import jax
import jax.numpy as jnp
from loguru import logger

from fdtdx.constraints.module import ConstraintInterface, ConstraintModule
from fdtdx.constraints.utils import compute_allowed_indices, nearest_index
from fdtdx.core.config import SimulationConfig
from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field, frozen_private_field
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.core.misc import get_air_name


@extended_autoinit
class PillarMapping(ConstraintModule):
    """Constraint module for mapping pillar structures to allowed configurations.

    Maps arbitrary pillar structures to the nearest allowed configurations based on
    material constraints and geometry requirements. Ensures structures meet fabrication
    rules like single polymer columns and no trapped air holes.

    Attributes:
        axis: Axis along which to enforce pillar constraints (0=x, 1=y, 2=z).
        single_polymer_columns: If True, restrict to single polymer columns.
        distance_metric: Method to compute distances between material distributions:
            - "euclidean": Standard Euclidean distance between permittivity values
            - "permittivity_differences_plus_average_permittivity": Weighted combination
              of permittivity differences and average permittivity values, optimized
              for material distribution comparisons
        _allowed_indices: Private array of allowed index combinations.
    """

    axis: int = frozen_field(init=True, kind="KW_ONLY")
    single_polymer_columns: bool = frozen_field(init=True, kind="KW_ONLY")

    distance_metric: Literal["euclidean", "permittivity_differences_plus_average_permittivity"] = frozen_field(
        default="permittivity_differences_plus_average_permittivity",
    )
    _allowed_indices: jax.Array = frozen_private_field()

    def input_interface(
        self,
        output_interface: ConstraintInterface,
    ) -> ConstraintInterface:
        """Define input interface requirements for this constraint module.

        Args:
            output_interface: Interface specification from previous module.

        Returns:
            ConstraintInterface: Required input interface specification.

        Raises:
            Exception: If output interface type is not inverse permittivity.
            Exception: If output interface has multiple shapes.
        """
        if output_interface.type != "inv_permittivity":
            raise Exception("After PillarMapping can only follow a module using" "Inverse permittivities")
        if len(output_interface.shapes) != 1:
            raise Exception(f"Output of PillarMapping needs to be single array, " f"but got {output_interface.shapes=}")
        return output_interface

    def init_module(
        self: Self,
        config: SimulationConfig,
        permittivity_config: dict[str, float],
        output_interface: ConstraintInterface,
    ) -> Self:
        """Initialize the pillar mapping module.

        Sets up allowed index combinations based on material constraints and geometry
        requirements. Computes valid pillar configurations that satisfy fabrication rules.

        Args:
            config: Global simulation configuration.
            permittivity_config: Material permittivity configurations.
            output_interface: Interface specification from previous module.

        Returns:
            Self: Initialized module instance.
        """
        self = super().init_module(
            config=config,
            permittivity_config=permittivity_config,
            output_interface=output_interface,
        )

        air_name = get_air_name(self._permittivity_config)
        air_index = self._permittivity_names.index(air_name)
        allowed_columns = compute_allowed_indices(
            num_layers=list(self._output_interface.shapes.values())[0][self.axis],
            indices=list(range(self._allowed_permittivities.shape[0])),
            fill_holes_with_index=[air_index],
            single_polymer_columns=self.single_polymer_columns,
        )
        self = self.aset("_allowed_indices", allowed_columns)
        logger.info(f"{allowed_columns=}")
        return self

    def transform(
        self,
        input_params: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Transform input parameters to satisfy pillar constraints.

        Maps arbitrary material distributions to nearest allowed pillar configurations
        using straight-through estimation for gradient computation.

        Args:
            input_params: Dictionary of input parameter arrays.

        Returns:
            dict[str, jax.Array]: Transformed parameter arrays satisfying constraints.

        Raises:
            Exception: If invalid axis specified.
        """
        p = list(input_params.values())[0]
        nearest_allowed_index = nearest_index(
            values=p,
            allowed_values=self._allowed_inverse_permittivities,
            axis=self.axis,
            distance_metric=self.distance_metric,
            allowed_indices=self._allowed_indices,
            return_distances=False,
        )
        latents = self._allowed_inverse_permittivities[self._allowed_indices]
        quantized_latents = latents[nearest_allowed_index]
        if self.axis == 2:
            pass  # no transposition needed
        elif self.axis == 1:
            quantized_latents = jnp.transpose(quantized_latents, axes=(0, 2, 1))
        elif self.axis == 0:
            quantized_latents = jnp.transpose(quantized_latents, axes=(2, 0, 1))
        else:
            raise Exception(f"invalid axis: {self.axis}")

        result = {k: straight_through_estimator(input_params[k], quantized_latents) for k in input_params.keys()}
        return result
