from typing import Literal, Self, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.figure import Figure
from rich.progress import Progress

from fdtdx.core.jax.pytrees import extended_autoinit, field, frozen_field

# from fdtdx.core.physics.modes import compute_accurate_mode
from fdtdx.core.physics.modes import compute_mode
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.objects.detectors.plotting.line_plot import plot_line_over_time


@extended_autoinit
class ModeOverlapDetector(Detector):
    """Detector for measuring the overlap of a waveguide mode with the simulation fields.
    This detector computes the overlap of a mode with the phasor fields at a specified
    frequency, enabling frequency-domain analysis of the electromagnetic fields.

    Attributes:
        frequencies: Sequence of frequencies to analyze
        mode_index: Index of the mode to analyze
        filter_pol: Polarization filter to apply to the mode
        direction: Direction of the mode
        components: Sequence of field components to measure. Can include any of:
            "Ex", "Ey", "Ez", "Hx", "Hy", "Hz".
    """

    # Phasor attributes
    frequencies: Sequence[float] = (None,)
    filter_pol: str = "te"
    direction: str = "+"
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )
    dtype: jnp.dtype = frozen_field(default=jnp.complex64, kind="KW_ONLY")

    # Mode overlap attributes
    mode_index: int = field(default=0, kind="KW_ONLY")

    def __post_init__(self):
        if self.dtype not in [jnp.complex64, jnp.complex128]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        self._angular_frequencies = 2 * jnp.pi * jnp.array(self.frequencies)
        self._scale = self._config.time_step_duration / jnp.sqrt(2 * jnp.pi)

    def _shape_dtype_single_time_step(self) -> dict[str, jax.ShapeDtypeStruct]:
        field_dtype = jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64
        # Currently only supporting one frequency
        # num_frequencies = len(self.frequencies)
        num_components = len(self.components)
        phasor_shape = (num_components, *self.grid_shape)
        # phasor_shape = (num_components, *self.grid_shape)
        return {
            "phasor": jax.ShapeDtypeStruct(shape=phasor_shape, dtype=field_dtype),
            "overlap": jax.ShapeDtypeStruct(shape=(1,), dtype=field_dtype),
            "E_mode": jax.ShapeDtypeStruct(shape=(3, *self.grid_shape[1:]), dtype=field_dtype),
            "H_mode": jax.ShapeDtypeStruct(shape=(3, *self.grid_shape[1:]), dtype=field_dtype),
            "mode_init": jax.ShapeDtypeStruct(shape=(), dtype=bool),
        }

    def update(
        self,
        time_step: jax.Array,
        E: jax.Array,
        H: jax.Array,
        state: DetectorState,
        inv_permittivity: jax.Array,
        inv_permeability: jax.Array | float,
    ) -> DetectorState:
        # del inv_permittivity, inv_permeability
        time_passed = time_step * self._config.time_step_duration
        static_scale = 2 / self.num_time_steps_recorded

        # Extract current slice
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

        # Compute phasors
        phase_angles = self._angular_frequencies[:, None] * time_passed
        phasors = jnp.exp(1j * phase_angles)
        new_phasors = EH * phasors * static_scale
        # Take phasor field given at frequency
        # So far, works for only one frequency
        phasor_field = jnp.squeeze(state["phasor"])
        num_e = sum(c.startswith("E") for c in self.components)
        E_phasor = jnp.squeeze(phasor_field[:num_e])
        H_phasor = jnp.squeeze(phasor_field[num_e:])

        # --- Mode Overlap --- #
        # Compute or reuse mode to be compared with
        def compute_mode_helper(_):
            E_mode, H_mode, _ = compute_mode(
                frequency=self.frequencies[0],
                inv_permittivities=inv_permittivity[self.grid_slice],
                inv_permeabilities=inv_permeability,
                resolution=self._config.resolution,
                direction=self.direction,
                mode_index=self.mode_index,
                filter_pol=self.filter_pol,
            )

            return jnp.squeeze(E_mode).astype(self.dtype), jnp.squeeze(H_mode).astype(self.dtype)

        def reuse_mode(_):
            return state["E_mode"], state["H_mode"]

        E_mode, H_mode = jax.lax.cond(
            state["mode_init"],
            reuse_mode,
            compute_mode_helper,
            operand=None,
        )

        def compute_mode_overlap(E_mode, H_mode, E_sim, H_sim, inv_permittivity, axis=0):
            """Compute mode overlap |c_p|^2 between mode and simulation fields."""
            # Cross products for overlap calculation
            E_cross_H_star_sim = jnp.cross(
                jnp.transpose(E_mode, (1, 2, 0)),
                jnp.transpose(jnp.conj(H_sim), (1, 2, 0)),
            )
            E_star_cross_H_sim = jnp.cross(
                jnp.transpose(jnp.conj(E_sim), (1, 2, 0)),
                jnp.transpose(H_mode, (1, 2, 0)),
            )

            numerator = jnp.sum((E_cross_H_star_sim[:, :, axis] + E_star_cross_H_sim[:, :, axis]) / 4.0)

            # Final projection coefficient
            c_p = numerator

            # Modal power
            overlap = jnp.abs(c_p) ** 2

            # return overlap
            return jnp.real(overlap).astype(jnp.float32)

        overlap = compute_mode_overlap(E_mode, H_mode, E_phasor, H_phasor, inv_permittivity[self.grid_slice])

        if self.inverse:
            resultphasor = state["phasor"] - new_phasors
        else:
            resultphasor = state["phasor"] + new_phasors

        arr_idx = self._time_step_to_arr_idx[time_step]
        new_full_arr = state["overlap"].at[arr_idx].set(overlap)
        return {
            "phasor": resultphasor.astype(self.dtype),
            "overlap": new_full_arr.astype(self.dtype),
            "E_mode": E_mode.astype(self.dtype),
            "H_mode": H_mode.astype(self.dtype),
            "mode_init": jnp.asarray(True),
        }

    def draw_plot(
        self,
        state: dict[str, np.ndarray],
        progress: Progress | None = None,
    ) -> dict[str, Figure | str]:
        overlap_arr = state["overlap"].squeeze()

        time_steps = np.where(np.asarray(self._is_on_at_time_step_arr))[0]
        time_steps = time_steps * self._config.time_step_duration

        fig = plot_line_over_time(
            arr=overlap_arr,
            time_steps=time_steps.tolist(),
            metric_name=f"{self.name}: Mode Overlap",
        )

        return {"overlap": fig}

    def init_state(
        self: Self,
    ) -> DetectorState:
        """Initializes detector state arrays for recording data.

        Creates zero-initialized arrays for storing field data based on
        detector configuration.

        Returns:
            DetectorState: Dictionary containing initialized detector arrays.
        """
        # Initialize arrays
        shape_dtype_dict = self._shape_dtype_single_time_step()
        state = {}
        latent_time_size = self._num_latent_time_steps()

        for name, shape_dtype in shape_dtype_dict.items():
            # if name in {"phasor", "overlap"}:
            if name in {"overlap"}:
                cur_arr = jnp.zeros(
                    shape=(latent_time_size, *shape_dtype.shape),
                    dtype=shape_dtype.dtype,
                )
                state[name] = cur_arr
            else:
                # no time dependency
                state[name] = jnp.zeros(shape=shape_dtype.shape, dtype=shape_dtype.dtype)

        state["mode_init"] = False
        return state
