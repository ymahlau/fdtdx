from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.linalg import rotate_vector
from fdtdx.core.null import Null
from fdtdx.dispersion import effective_inv_permittivity
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

    The medium permittivity/permeability at the source cell is sampled once
    during :meth:`apply` — at the carrier angular frequency when a dispersive
    coefficient arrays is provided — so dispersive media are handled correctly
    without runtime overhead.
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

    _inv_eps_local: jax.Array = private_field()
    _inv_mu_local: jax.Array | float = private_field()

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

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        *,
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
    ) -> Self:
        del key

        # inv_permittivities shape: (num_components, Nx, Ny, Nz)
        inv_eps_slice = inv_permittivities[:, *self.grid_slice]

        if dispersive_c1 is not None and dispersive_c2 is not None and dispersive_c3 is not None:
            if inv_eps_slice.ndim >= 1 and inv_eps_slice.shape[0] == 9:
                raise NotImplementedError(
                    "Dispersive materials cannot be combined with fully anisotropic "
                    "(off-diagonal) permittivity tensors in v1."
                )
            c1_slice = dispersive_c1[:, :, *self.grid_slice]
            c2_slice = dispersive_c2[:, :, *self.grid_slice]
            c3_slice = dispersive_c3[:, :, *self.grid_slice]
            inv_eps_slice = effective_inv_permittivity(
                inv_eps=inv_eps_slice,
                c1=c1_slice,
                c2=c2_slice,
                c3=c3_slice,
                omega=2.0 * np.pi * self.wave_character.get_frequency(),
                dt=self._config.time_step_duration,
            )

        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_mu_slice: jax.Array | float = inv_permeabilities[:, *self.grid_slice]
        else:
            inv_mu_slice = inv_permeabilities

        self = self.aset("_inv_eps_local", inv_eps_slice, create_new_ok=True)
        self = self.aset("_inv_mu_local", inv_mu_slice, create_new_ok=True)
        return self

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

        if isinstance(self._inv_eps_local, Null):
            inv_eps_source = inv_permittivities[:, *self.grid_slice]
        else:
            inv_eps_source = self._inv_eps_local

        for axis in range(3):
            weight = orientation[axis]
            inv_eps_local = inv_eps_source[axis]
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

        if isinstance(self._inv_mu_local, Null):
            inv_mu_source: jax.Array | float = inv_permeabilities
            if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
                inv_mu_source = inv_permeabilities[:, *self.grid_slice]
        else:
            inv_mu_source = self._inv_mu_local

        for axis in range(3):
            weight = orientation[axis]
            if isinstance(inv_mu_source, jax.Array) and inv_mu_source.ndim > 0:
                inv_mu_local = inv_mu_source[axis]
            else:
                inv_mu_local = inv_mu_source
            injection = c * inv_mu_local * weight * self.amplitude * self.static_amplitude_factor * amplitude
            H = H.at[axis, *self.grid_slice].add(sign * injection.astype(H.dtype))

        return H
