from typing import Self, Sequence

import jax

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field, frozen_private_field
from fdtdx.core.jax.utils import check_shape_dtype
from fdtdx.materials import ContinuousMaterialRange, Material
from fdtdx.objects.parameters.discretization import Discretization
from fdtdx.objects.parameters.latent import LatentParamsTransformation, StandardToInversePermittivityRange


@extended_autoinit
class BaseParameterMapping(ExtendedTreeClass):
    latent_transforms: Sequence[LatentParamsTransformation] = frozen_field(
        kind="KW_ONLY", 
        default=(StandardToInversePermittivityRange(),)
    )
    _input_shape_dtypes: dict[str, jax.ShapeDtypeStruct] = frozen_private_field()

    def __call__(
        self,
        input_params: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        check_shape_dtype(input_params, self._input_shape_dtypes)
        # walk through modules
        x = input_params
        for t in self.latent_transforms:
            check_shape_dtype(x, t._input_shape_dtypes)
            x = t.transform(x)
            check_shape_dtype(x, t._output_shape_dtypes)
        return x

    def init_modules(
        self: Self,
        config: SimulationConfig,
        materials: dict[str, Material] | ContinuousMaterialRange,
        output_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    ) -> Self:
        # init list of modules
        cur_output_shape_dtypes, new_modules = output_shape_dtypes, []
        for m in self.latent_transforms[::-1]:
            m_new = m.init_module(
                config=config,
                materials=materials,
                output_shape_dtypes=cur_output_shape_dtypes,
            )
            new_modules.append(m_new)
            cur_output_shape_dtypes = m_new._input_shape_dtypes

        # set own input shape dtype
        self = self.aset("_input_shape_dtypes", cur_output_shape_dtypes)
        self = self.aset("modules", new_modules[::-1])
        return self


@extended_autoinit
class DiscreteParameterMapping(BaseParameterMapping):
    discretization: Discretization = frozen_field(kind="KW_ONLY")  # type: ignore
    post_transforms: Sequence[DiscreteTransformation] = frozen_field(default=tuple([]), kind="KW_ONLY")