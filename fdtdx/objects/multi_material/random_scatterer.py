import os
from pathlib import Path
from typing import Self
import jax
import jax.numpy as jnp
import pytreeclass as tc
import h5py
from jax.experimental import io_callback

from fdtdx.core.config import SimulationConfig
from fdtdx.core.plotting.colors import PINK
from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field
from fdtdx.objects.multi_material.multi_material import MultiMaterial
from fdtdx.core.jax.typing import (
    SliceTuple3D,
    PartialGridShape3D,
    PartialRealShape3D,
    UNDEFINED_SHAPE_3D
)

@extended_autoinit
class RandomScatterer(MultiMaterial):
    scatter_permittivity: float = frozen_field(
        default=2.25,
        init=False,
        kind="KW_ONLY",
    )
    outer_permittivity: float = 1
    permittivity_config: dict[str, float] = frozen_field(
        default=None, # type: ignore
        init=False,
    )
    partial_voxel_grid_shape: PartialGridShape3D = tc.field(default=UNDEFINED_SHAPE_3D, init=False)  # type: ignore
    partial_voxel_real_shape: PartialRealShape3D = tc.field(default=None, init=False)  # type: ignore
    color: tuple[float, float, float] = PINK
    dataset_path: str | Path = frozen_field(  # type: ignore
        kind="KW_ONLY",
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    )
    _design: jax.Array = tc.field(  # type: ignore
        default=None,
        init=False,
    )
    _fields: jax.Array = tc.field(  # type: ignore
        default=None,
        init=False,
    )
    _inv_permittivity: jax.Array = tc.field(  # type: ignore
        default=None,
        init=False,
    )
    
    def _load_file(self, key: jax.Array):
        # load all dataset file paths
        hdf5_files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".h5"):
                    hdf5_files.append(os.path.join(root, file))
        # randomly select one file path
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(
            subkey, shape=(1,), minval=0, maxval=len(hdf5_files)
        )[0]
        dataset_path = hdf5_files[idx]
        # read in designs and fields in file
        # select a random design (they are of size 128x128x128)
        # resize to shape and set permittivity where design is 1
        key, rng = jax.random.split(key)
        with h5py.File(dataset_path, "r") as f:
            keys = list(f.keys())
            index = jax.random.randint(rng, (1,), 0, len(keys))[0]
            data = f[keys[index]]
            design = jnp.asarray(data["design"])  # type: ignore
            fields = jnp.asarray(data["fields"])  # type: ignore
        return design, fields
    
    
    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        voxel_grid_shape = (
            config.resolution,
            config.resolution,
            config.resolution,
        )
        self = self.aset("partial_voxel_real_shape", voxel_grid_shape)
        key, subkey = jax.random.split(key)
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=subkey,
        )
        permittivity_config = {
            "scatter": self.scatter_permittivity,
            "outside": self.outer_permittivity,
        }
        self = self.aset("permittivity_config", permittivity_config)
        design, fields = io_callback(
            self._load_file,
            (
                jnp.zeros((128, 128, 128), dtype=jnp.bool),
                jnp.zeros((6, 128, 128, 128), dtype=config.dtype),
            ),
            key,
        )
        design = jax.image.resize(design, self.grid_shape, method="nearest")
        fields = jax.image.resize(fields, (6,) + self.grid_shape, method="nearest")
        inv_perm = jnp.where(
            design,
            1 / self.permittivity_config["scatter"],
            1 / self.permittivity_config["outside"],
        ).astype(config.dtype)
        self = self.aset("_design", design)
        self = self.aset("_fields", fields)
        self = self.aset("_inv_permittivity", inv_perm)
        return self
    
    
    def get_inv_permittivity(
        self,
        prev_inv_permittivity: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permittivity and info dict  
        del prev_inv_permittivity, params    
        return self._inv_permittivity, {}
        
        
    def get_inv_permeability(
        self,
        prev_inv_permeability: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permeability and info dict
        del params
        return prev_inv_permeability, {}
    