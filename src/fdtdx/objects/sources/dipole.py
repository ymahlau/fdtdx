from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.linalg import rotate_vector
from fdtdx.objects.sources.source import Source


@autoinit
class PointDipoleSource(Source):
    """Soft point dipole source (electric or magnetic).

    Injects an impressed current at a single Yee cell. The source is "soft":
    it adds to the field rather than overwriting, so scattered/reflected fields
    pass through without artificial reflections.

    The dipole orientation starts along the ``polarization`` axis and is then
    rotated by ``azimuth_angle`` and ``elevation_angle`` (both in degrees),
    following the same convention as :class:`TFSFPlaneSource`.  When both
    angles are zero the dipole is axis-aligned, recovering the original
    behavior.

    For an electric dipole with unit orientation ``p_hat``, the E-field
    update at each time step is::

        E[i, x, y, z] += -c * inv_eps[i] * p_hat[i] * amplitude * temporal(t)

    for each component *i* in {0, 1, 2}.

    For a magnetic dipole, the dual applies during the H update with
    inv_permeability replacing inv_permittivity.
    """

    #: Polarization axis (0=x, 1=y, 2=z).
    polarization: int = frozen_field()

    #: Azimuth angle in degrees (rotation around vertical axis).
    azimuth_angle: float = frozen_field(default=0.0)

    #: Elevation angle in degrees (rotation around horizontal axis).
    elevation_angle: float = frozen_field(default=0.0)

    #: Source type: "electric" injects into E update, "magnetic" into H update.
    source_type: Literal["electric", "magnetic"] = frozen_field(default="electric")

    #: Source amplitude.
    amplitude: float = frozen_field(default=1.0)

    def __post_init__(self):
        if self.source_type not in ("electric", "magnetic"):
            raise ValueError(f"source_type must be electric or magnetic, got {self.source_type}")
        if self.polarization not in (0, 1, 2):
            raise ValueError(f"polarization must be 0, 1, or 2, got {self.polarization}")

    @property
    def _orientation(self) -> jnp.ndarray:
        """Normalized orientation vector as a (3,) JAX array.

        Starts as the unit vector along ``polarization`` and is rotated by
        ``azimuth_angle`` / ``elevation_angle`` using the same rotation
        convention as :func:`rotate_vector`.
        """
        base = jnp.zeros(3, dtype=jnp.float32).at[self.polarization].set(1.0)
        if self.azimuth_angle == 0.0 and self.elevation_angle == 0.0:
            return base
        horizontal_axis = (self.polarization + 1) % 3
        vertical_axis = (self.polarization + 2) % 3
        axes_tuple = (horizontal_axis, vertical_axis, self.polarization)
        return rotate_vector(
            base,
            azimuth_angle=np.deg2rad(self.azimuth_angle),
            elevation_angle=np.deg2rad(self.elevation_angle),
            axes_tuple=axes_tuple,
        )

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permeabilities
        if self.source_type != "electric":
            return E

        dt = self._config.time_step_duration
        c = self._config.courant_number

        amplitude = self.temporal_profile.get_amplitude(
            time=time_step * dt,
            period=self.wave_character.get_period(),
            phase_shift=self.wave_character.phase_shift,
        )

        orientation = self._orientation
        sign = -1.0 if not inverse else 1.0

        for axis in range(3):
            weight = orientation[axis]
            inv_eps_local = inv_permittivities[axis, *self.grid_slice]
            injection = c * inv_eps_local * weight * self.amplitude * self.static_amplitude_factor * amplitude
            E = E.at[axis, *self.grid_slice].add(sign * injection.astype(E.dtype))

        return E

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities
        if self.source_type != "magnetic":
            return H

        dt = self._config.time_step_duration
        c = self._config.courant_number

        amplitude = self.temporal_profile.get_amplitude(
            time=time_step * dt,
            period=self.wave_character.get_period(),
            phase_shift=self.wave_character.phase_shift,
        )

        orientation = self._orientation
        sign = -1.0 if not inverse else 1.0

        for axis in range(3):
            weight = orientation[axis]
            if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
                inv_mu_local = inv_permeabilities[axis, *self.grid_slice]
            else:
                inv_mu_local = inv_permeabilities
            injection = c * inv_mu_local * weight * self.amplitude * self.static_amplitude_factor * amplitude
            H = H.at[axis, *self.grid_slice].add(sign * injection.astype(H.dtype))

        return H
