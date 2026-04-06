from typing import Literal

import jax

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.sources.source import Source


@autoinit
class PointDipoleSource(Source):
    """Soft point dipole source (electric or magnetic).

    Injects an impressed current at a single Yee cell. The source is "soft":
    it adds to the field rather than overwriting, so scattered/reflected fields
    pass through without artificial reflections.

    For an electric dipole polarized along axis ``p``, the E-field update at
    each time step is::

        E[p, i, j, k] += -c * inv_eps[p] * amplitude * temporal(t)

    where ``c`` is the Courant number and ``inv_eps`` is the inverse relative
    permittivity.  This uses the same coefficient structure as the curl update
    (``E += c * inv_eps * curl_H``), so ``amplitude=1.0`` produces a source
    contribution comparable to one unit of curl.

    For a magnetic dipole, the dual applies during the H update with
    inv_permeability replacing inv_permittivity.

    Args:
        polarization: Axis along which the dipole is oriented (0=x, 1=y, 2=z).
        source_type: ``"electric"`` injects current into the E update,
            ``"magnetic"`` injects into the H update.
        amplitude: Source strength (default 1.0). Combined with
            ``static_amplitude_factor`` and the temporal profile to compute the
            injected current at each time step.
    """

    #: Polarization axis (0=x, 1=y, 2=z).
    polarization: int = frozen_field()

    #: Source type: "electric" injects into E update, "magnetic" into H update.
    source_type: Literal["electric", "magnetic"] = frozen_field(default="electric")

    #: Source amplitude.
    amplitude: float = frozen_field(default=1.0)

    def __post_init__(self):
        if self.polarization not in (0, 1, 2):
            raise ValueError(f"polarization must be 0, 1, or 2, got {self.polarization}")

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

        inv_eps_local = inv_permittivities[self.polarization, *self.grid_slice]

        # Soft source injection: same coefficient structure as curl update
        # E += c * inv_eps * curl_H  →  source adds  c * inv_eps * amplitude * temporal
        injection = c * inv_eps_local * self.amplitude * self.static_amplitude_factor * amplitude

        sign = -1.0 if not inverse else 1.0
        E = E.at[self.polarization, *self.grid_slice].add(sign * injection.astype(E.dtype))
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

        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_mu_local = inv_permeabilities[self.polarization, *self.grid_slice]
        else:
            inv_mu_local = inv_permeabilities

        # Dual of electric source: H += c * inv_mu * amplitude * temporal
        injection = c * inv_mu_local * self.amplitude * self.static_amplitude_factor * amplitude

        sign = -1.0 if not inverse else 1.0
        H = H.at[self.polarization, *self.grid_slice].add(sign * injection.astype(H.dtype))
        return H
