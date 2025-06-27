from typing import Literal, Sequence

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.detectors.detector import Detector, DetectorState


@autoinit
class FieldDetector(Detector):
    """Detector for measuring field components of electromagnetic fields in the time domain.

    Attributes:
        reduce_volume (bool, optional): If True, reduces the volume of recorded data. Defaults to False.
        components (Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]], optional): Sequence of field components to
            measure. Can include any of: "Ex", "Ey", "Ez", "Hx", "Hy", "Hz".
            Defaults to ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz").
    """

    reduce_volume: bool = frozen_field(default=False)
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    )

    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        num_components = len(self.components)
        component_shape = (num_components,) if self.reduce_volume else (num_components, *self.grid_shape)
        return {"fields": jax.ShapeDtypeStruct(shape=component_shape, dtype=self.dtype)}

    def update(
        self,
        time_step: jax.Array,
        E: jax.Array,
        H: jax.Array,
        state: DetectorState,
        inv_permittivity: jax.Array,
        inv_permeability: jax.Array | float,
    ) -> DetectorState:
        del inv_permeability, inv_permittivity

        E, H = E[:, *self.grid_slice], H[:, *self.grid_slice]
        fields = []
        if "Ex" in self.components:
            fields.append(E[0])
        if "Ey" in self.components:
            fields.append(E[1])
        if "Ez" in self.components:
            fields.append(E[2])
        if "Hx" in self.components:
            fields.append(H[0])
        if "Hy" in self.components:
            fields.append(H[1])
        if "Hz" in self.components:
            fields.append(H[2])

        EH = jnp.stack(fields, axis=0)

        if self.reduce_volume:
            EH = EH.mean(axis=(1, 2, 3))
        arr_idx = self._time_step_to_arr_idx[time_step]
        new_full_arr = state["fields"].at[arr_idx].set(EH)
        new_state = {"fields": new_full_arr}
        return new_state
