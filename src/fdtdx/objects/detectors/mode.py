from typing import Literal, Self, Sequence

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.physics.modes import compute_mode
from fdtdx.objects.detectors.detector import DetectorState
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.typing import SliceTuple3D


@autoinit
class ModeOverlapDetector(PhasorDetector):
    """
    Detector for measuring the overlap of a waveguide mode with the simulation fields.
    This detector computes the overlap of a mode with the phasor fields at a specified
    frequency, enabling frequency-domain analysis of the electromagnetic fields.

    The mode overlap is calculated by integrating the cross product of the mode fields
    with the simulation fields over a cross-sectional plane. This is useful for
    analyzing waveguide coupling efficiency, transmission coefficients, and modal
    decomposition of electromagnetic fields.

    Attributes:
        direction (Literal["+", "-"]): Direction of mode propagation, either "+" (forward) or "-" (backward).
                  Determines which direction along the waveguide axis the mode is
                  assumed to propagate.
        mode_index (int, optional): Index of the waveguide mode to use for overlap calculation.
                   Defaults to 0 (fundamental mode). Higher indices correspond to higher-order modes.
        filter_pol (Literal["te", "tm"] | None, optional): Optional polarization filter for the mode calculation.
                   Can be "te" (transverse electric), "tm" (transverse magnetic), or None (no filtering).
                   When specified, only modes of the given polarization type are considered. Defaults to None.
        components (Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]], optional): Cannot be specified here since
            the detector needs all components.
        plot (bool, optional): Cannot be specified here since plotting a single scalar is useless.

    """

    direction: Literal["+", "-"] = frozen_field()
    mode_index: int = frozen_field(default=0)
    filter_pol: Literal["te", "tm"] | None = frozen_field(default=None)
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        init=False,  # in this detector, we always want all components. Do not give user a choice
    )  # noqa: DOC603, DOC601
    plot: bool = frozen_field(default=False, init=False)  # noqa: DOC603, DOC601 # single scalar is useless for plotting
    _mode_E: jax.Array = private_field()
    _mode_H: jax.Array = private_field()

    @property
    def propagation_axis(self) -> int:
        if sum([a == 1 for a in self.grid_shape]) != 1:
            raise Exception(f"Invalid ModeOverlapDetector shape: {self.grid_shape}")
        return self.grid_shape.index(1)

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        if len(self.wave_characters) > 1:
            raise NotImplementedError()
        return self

    def apply(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
    ) -> Self:
        del key

        inv_permittivity_slice = inv_permittivities[*self.grid_slice]
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_permeability_slice = inv_permeabilities[*self.grid_slice]
        else:
            inv_permeability_slice = inv_permeabilities

        mode_E, mode_H, _ = compute_mode(
            frequency=self.wave_characters[0].frequency,
            inv_permittivities=inv_permittivity_slice,
            inv_permeabilities=inv_permeability_slice,
            resolution=self._config.resolution,
            direction=self.direction,
            mode_index=self.mode_index,
            filter_pol=self.filter_pol,
        )

        self = self.aset("_mode_E", mode_E, create_new_ok=True)
        self = self.aset("_mode_H", mode_H, create_new_ok=True)
        return self

    def compute_overlap_to_mode(
        self,
        state: DetectorState,
        mode_E: jax.Array,
        mode_H: jax.Array,
    ) -> jax.Array:
        # shape (time step, num_freqs, num_components, *spatial)
        # time steps is always 1 and num_components always 6
        phasors = state["phasor"]
        phasors_E, phasors_H = phasors[0, 0, :3], phasors[0, 0, 3:]

        E_cross_H_star_sim = jnp.cross(
            mode_E,
            jnp.conj(phasors_H),
            axis=0,
        )[self.propagation_axis]

        E_star_cross_H_sim = jnp.cross(
            jnp.conj(phasors_E),
            mode_H,
            axis=0,
        )[self.propagation_axis]

        alpha_coeff = jnp.sum(E_cross_H_star_sim + E_star_cross_H_sim)
        alpha_coeff = alpha_coeff / 4.0

        return alpha_coeff

    def compute_overlap(
        self,
        state: DetectorState,
    ) -> jax.Array:
        if self._mode_E is None or self._mode_H is None:
            raise Exception("Need to call apply on ModeOverlapDetector before calling compute_mode_overlap!")
        return self.compute_overlap_to_mode(
            state=state,
            mode_E=self._mode_E,
            mode_H=self._mode_H,
        )
