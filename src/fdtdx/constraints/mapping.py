from typing import Self, Sequence

import jax

from fdtdx.constraints.module import ConstraintInterface, ConstraintModule, check_interface_compliance
from fdtdx.core.config import SimulationConfig
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field, frozen_private_field


@extended_autoinit
class ConstraintMapping(ExtendedTreeClass):
    """A mapping that chains multiple constraint modules together.

    Manages a sequence of constraint modules that transform parameters between different
    representations while enforcing constraints. Each module's output must match the input
    interface of the next module in the chain.

    Attributes:
        modules: Sequence of ConstraintModule instances to apply in order.
        _input_interface: Interface specification for input parameters.
    """

    modules: Sequence[ConstraintModule] = frozen_field(
        kind="KW_ONLY",
    )
    _input_interface: ConstraintInterface = frozen_private_field()

    def __call__(
        self,
        input_params: dict[str, jax.Array],
    ) -> jax.Array:
        """Transform input parameters through the chain of constraint modules.

        Applies each constraint module in sequence to transform the input parameters,
        validating interface compliance between modules.

        Args:
            input_params: Dictionary mapping parameter names to JAX arrays.

        Returns:
            The final transformed array from the last module.

        Raises:
            Exception: If input parameters don't match module interfaces.
        """
        check_interface_compliance(input_params, self._input_interface)
        # walk through modules
        x = input_params
        for m in self.modules:
            check_interface_compliance(x, m._input_interface)
            x = m.transform(x)
            check_interface_compliance(x, m._output_interface)
        return list(x.values())[0]

    def init_modules(
        self: Self,
        config: SimulationConfig,
        permittivity_config: dict[str, float],
        output_interface: ConstraintInterface,
    ) -> Self:
        """Initialize all constraint modules in the mapping chain.

        Sets up each module with the simulation configuration and ensures proper interface
        matching between modules. The last module must output inverse permittivity values.

        Args:
            config: Global simulation configuration.
            permittivity_config: Material permittivity values.
            output_interface: Interface specification for final output.

        Returns:
            Self with initialized modules.

        Raises:
            Exception: If output interface is invalid or modules can't form valid chain.
        """
        # sanity checks
        if len(output_interface.shapes) != 1:
            raise Exception(f"Output of parameter mapping needs to have length 1, but got {output_interface.shapes=}")
        if output_interface.type != "inv_permittivity":
            raise Exception("Output of last module in constraint mapping needs to be Inverse Permittivity")

        # init list of modules
        cur_output_interface, new_modules = output_interface, []
        for m in self.modules[::-1]:
            m_new = m.init_module(
                config=config,
                permittivity_config=permittivity_config,
                output_interface=cur_output_interface,
            )
            new_modules.append(m_new)
            cur_output_interface = m_new._input_interface

        if cur_output_interface.type != "latent":
            raise Exception(
                "First Module of Constraint Mapping needs to be able to work with latent parameters as Input"
            )

        # set own input shape dtype
        self = self.aset("_input_interface", cur_output_interface)
        self = self.aset("modules", new_modules[::-1])
        return self
