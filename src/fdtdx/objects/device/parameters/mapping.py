from typing import Self, Sequence

import jax

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field, frozen_private_field
from fdtdx.core.jax.utils import check_shape_dtype
from fdtdx.materials import ContinuousMaterialRange, Material
from fdtdx.objects.device.parameters.discrete import DiscreteTransformation
from fdtdx.objects.device.parameters.discretization import ClosestIndex, Discretization
from fdtdx.objects.device.parameters.latent import LatentParamsTransformation, StandardToInversePermittivityRange


@extended_autoinit
class LatentParameterMapping(ExtendedTreeClass):
    latent_transforms: Sequence[LatentParamsTransformation] = frozen_field(
        kind="KW_ONLY", 
        default=(StandardToInversePermittivityRange(),)
    )
    _input_shape_dtypes: dict[str, jax.ShapeDtypeStruct] | jax.ShapeDtypeStruct = frozen_private_field()

    def __call__(
        self,
        input_params: dict[str, jax.Array] | jax.Array,
    ) -> dict[str, jax.Array] | jax.Array:
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
        material: dict[str, Material] | ContinuousMaterialRange,
        output_shape_dtypes: dict[str, jax.ShapeDtypeStruct] | jax.ShapeDtypeStruct,
    ) -> Self:
        # init list of modules
        cur_output_shape_dtypes, new_modules = output_shape_dtypes, []
        for m in self.latent_transforms[::-1]:
            m_new = m.init_module(
                config=config,
                material=material,
                output_shape_dtypes=cur_output_shape_dtypes,
            )
            new_modules.append(m_new)
            cur_output_shape_dtypes = m_new._input_shape_dtypes

        # set own input shape dtype
        self = self.aset("_input_shape_dtypes", cur_output_shape_dtypes)
        self = self.aset("modules", new_modules[::-1])
        return self


@extended_autoinit
class DiscreteParameterMapping(LatentParameterMapping):
    discretization: Discretization = frozen_field(kind="KW_ONLY", default=ClosestIndex())
    post_transforms: Sequence[DiscreteTransformation] = frozen_field(default=tuple([]), kind="KW_ONLY")
    
    def init_modules(
        self: Self,
        config: SimulationConfig,
        material: dict[str, Material],
        output_shape_dtype: jax.ShapeDtypeStruct,
    ) -> Self:
        new_post_transforms = []
        for t in self.post_transforms:
            cur_transform = t.init_module(
                config=config,
                material=material,
            )
            new_post_transforms.append(cur_transform)
        new_discretization = self.discretization.init_module(
            config=config,
            material=material,
            output_shape_dtype=output_shape_dtype,
        )
        
        new_latent_transforms = []
        cur_output_shape_dtype = new_discretization._input_shape_dtypes
        for m in self.latent_transforms:
            m_new = m.init_module(
                config=config,
                material=material,
                output_shape_dtypes=cur_output_shape_dtype,
            )
            new_latent_transforms.append(m_new)
            cur_output_shape_dtype = m_new._input_shape_dtypes
        self = super().init_modules(
            config=config,
            material=material,
            output_shape_dtypes=cur_output_shape_dtype,
        )
        self = self.aset("latent_transforms", new_latent_transforms)
        self = self.aset("discretization", new_discretization)
        self = self.aset("post_transforms", new_post_transforms)
        return self
    
    def __call__(
        self,
        input_params: dict[str, jax.Array] | jax.Array,
    ) -> jax.Array:
        latent = super().__call__(
            input_params=input_params,
        )
        discretized = self.discretization(latent)
        cur_arr = discretized
        for transform in self.post_transforms:
            cur_arr = transform(cur_arr)
        return cur_arr
    