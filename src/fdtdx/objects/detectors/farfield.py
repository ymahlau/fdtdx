"""Far-field projectors: planar (angular spectrum) and box (surface equivalence).

These placed detectors record near-field phasors during the run and expose post-run
**accessor methods** that project to the far field on spherical, Cartesian, or k-space
observation grids (the recording geometry stays planar/box; spherical is only an output).
The projection math lives in :mod:`fdtdx.core.physics.farfield`.

Class hierarchy (all in this module):
    ``FarFieldProjector(PhasorDetector)`` — shared fields + ``directivity``/``radar_cross_section``
        ``PlanarFarFieldProjector`` — one plane, angular spectrum (+ optional periodic mode)
        ``BoxFarFieldProjector`` — six faces of the simulation volume, surface equivalence

See :mod:`fdtdx.core.physics.farfield` for the eta0-normalization and ``exp(-i w t)`` /
outgoing ``exp(+jkr)`` conventions that let the radiation integral consume the recorded
phasors directly.
"""

import warnings
from typing import TYPE_CHECKING, Literal, Sequence

import jax
import jax.numpy as jnp

from fdtdx import constants
from fdtdx.core.axis import get_transverse_axes
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.physics.farfield import (
    directivity_from_pattern,
    far_field_power_density,
    far_fields_from_NL,
    radiation_vectors,
    radiation_vectors_fft,
    spherical_basis,
    surface_equivalent_currents,
)
from fdtdx.core.physics.metrics import compute_poynting_flux
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.detectors.detector import DetectorState
from fdtdx.objects.detectors.phasor import PhasorDetector

if TYPE_CHECKING:
    from fdtdx.config import SimulationConfig
    from fdtdx.fdtd.container import ArrayContainer
    from fdtdx.objects.boundaries.initialization import BoundaryConfig
    from fdtdx.objects.object import GridCoordinateConstraint
    from fdtdx.objects.static_material.static import SimulationVolume

_DEFAULT_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_DFT_SIZE_WARN = 1e8  # cell * direction product above which the direct DFT is memory-heavy

# (state key, normal axis, outward sign, side) for the six faces of a box.
_BOX_FACES: tuple[tuple[str, int, float, str], ...] = (
    ("face_min_x", 0, -1.0, "min"),
    ("face_max_x", 0, 1.0, "max"),
    ("face_min_y", 1, -1.0, "min"),
    ("face_max_y", 1, 1.0, "max"),
    ("face_min_z", 2, -1.0, "min"),
    ("face_max_z", 2, 1.0, "max"),
)


@autoinit
class FarFieldProjector(PhasorDetector):
    """Base for far-field projectors: shared fields and observation-independent metrics.

    Subclasses implement :meth:`near_field` and :meth:`spherical`; this base derives
    :meth:`directivity` and :meth:`radar_cross_section` from ``spherical``. It also inherits
    ``flux_spectrum``/``measured_power_spectrum``/``transmission`` from ``PhasorDetector``.
    """

    #: Refractive index of the homogeneous medium the recording surface sits in.
    background_index: float = frozen_field(default=1.0)

    #: Far-field (asymptotic 1/r) projection. ``False`` (exact finite distance) not yet supported.
    far_field_approx: bool = frozen_field(default=True)

    def __post_init__(self):
        super().__post_init__()
        if tuple(self.components) != _DEFAULT_COMPONENTS:
            raise ValueError(
                f"{type(self).__name__} must record all six components in default order "
                f"{_DEFAULT_COMPONENTS}, got {tuple(self.components)}."
            )
        if self.reduce_volume:
            raise ValueError(f"{type(self).__name__} requires reduce_volume=False.")

    def _k_values(self) -> jax.Array:
        freqs = jnp.asarray([wc.get_frequency() for wc in self.wave_characters])
        return 2 * jnp.pi * freqs * self.background_index / constants.c

    def _check_far_field(self) -> None:
        if not self.far_field_approx:
            raise NotImplementedError("Exact finite-distance projection (far_field_approx=False) is not implemented.")

    def near_field(self, arrays: "ArrayContainer"):
        """Recorded near-field phasors (shape/layout depends on the subclass)."""
        raise NotImplementedError

    def spherical(
        self,
        arrays: "ArrayContainer",
        theta: jax.Array,
        phi: jax.Array,
        r: jax.Array | float = 1.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Far field ``(E_theta, E_phi)`` at observation angles, per frequency."""
        raise NotImplementedError

    def directivity(self, arrays: "ArrayContainer", theta: jax.Array, phi: jax.Array) -> jax.Array:
        """Directivity ``D(theta, phi)`` per frequency on a 1D ``(theta, phi)`` grid.

        Returns shape ``(num_freqs, len(theta), len(phi))``.
        """
        theta = jnp.asarray(theta)
        phi = jnp.asarray(phi)
        TH, PH = jnp.meshgrid(theta, phi, indexing="ij")
        e_theta, e_phi = self.spherical(arrays, TH, PH)
        u = far_field_power_density(e_theta, e_phi, self.background_index)
        return jax.vmap(lambda u_f: directivity_from_pattern(u_f, theta, phi))(u)

    def radar_cross_section(
        self,
        arrays: "ArrayContainer",
        theta: jax.Array,
        phi: jax.Array,
        incident_power_density: jax.Array | float,
        r: jax.Array | float = 1.0,
    ) -> jax.Array:
        """Bistatic RCS ``sigma = 4 pi r^2 S_scat / S_inc`` per frequency."""
        e_theta, e_phi = self.spherical(arrays, theta, phi, r=r)
        s_scat = far_field_power_density(e_theta, e_phi, self.background_index)
        return 4.0 * jnp.pi * jnp.asarray(r) ** 2 * s_scat / incident_power_density


@autoinit
class PlanarFarFieldProjector(FarFieldProjector):
    """Angular-spectrum projector recorded on a single plane.

    Records the complex near field on one plane (all six components) and projects it into the
    homogeneous half-space it faces — radiation patterns, beam divergence, Fourier-plane / NA
    analysis. Accessors: :meth:`near_field`, :meth:`spherical`, :meth:`cartesian`, :meth:`kspace`,
    plus inherited :meth:`directivity` / :meth:`radar_cross_section`.
    """

    #: Half-space to project into: ``"+"`` (outward normal along +axis) or ``"-"``.
    direction: Literal["+", "-"] = frozen_field(default="+")

    #: Reserved for the periodic / diffraction-order mode.
    periodic: bool = frozen_field(default=False)

    def _outward_sign(self) -> float:
        return 1.0 if self.direction == "+" else -1.0

    def _geometry(self) -> tuple[int, jax.Array, jax.Array, float]:
        """Return ``(normal_axis, positions, area, normal_coord)`` for the plane."""
        axis = self._plane_normal_axis()
        grid = self._config.resolved_grid
        centers = []
        for a in range(3):
            lo, hi = self.grid_slice_tuple[a]
            if grid is not None:
                c = jnp.asarray(grid.centers(a))[lo:hi]
            else:
                spacing = self._config.uniform_spacing()
                c = (jnp.arange(lo, hi) + 0.5) * spacing
            centers.append(c)
        xx, yy, zz = jnp.meshgrid(centers[0], centers[1], centers[2], indexing="ij")
        positions = jnp.stack([xx, yy, zz], axis=-1)
        area = self._face_area(axis)
        normal_coord = float(centers[axis][0])
        return axis, positions, area, normal_coord

    def _plane_fields(self, arrays: "ArrayContainer") -> tuple[jax.Array, jax.Array]:
        phasor = arrays.detector_states[self.name]["phasor"]  # (1, F, 6, *grid)
        static_scale = self._static_scale()
        return phasor[0, :, :3] / static_scale, phasor[0, :, 3:] / static_scale

    def near_field(self, arrays: "ArrayContainer") -> tuple[jax.Array, jax.Array]:
        """Recorded near-field phasors ``(E, H)``, each shape ``(num_freqs, 3, *grid_shape)``."""
        return self._plane_fields(arrays)

    def spherical(
        self,
        arrays: "ArrayContainer",
        theta: jax.Array,
        phi: jax.Array,
        r: jax.Array | float = 1.0,
    ) -> tuple[jax.Array, jax.Array]:
        self._check_far_field()
        axis, positions, area, _ = self._geometry()
        sign = self._outward_sign()
        E, H = self._plane_fields(arrays)
        k_vals = self._k_values()
        theta = jnp.asarray(theta)
        out_shape = theta.shape
        phi = jnp.broadcast_to(jnp.asarray(phi), out_shape)
        r_hat, theta_hat, phi_hat = spherical_basis(theta, phi)
        r_hat = r_hat.reshape(-1, 3)
        theta_hat = theta_hat.reshape(-1, 3)
        phi_hat = phi_hat.reshape(-1, 3)
        r_flat = jnp.broadcast_to(jnp.asarray(r), out_shape).reshape(-1).astype(theta_hat.dtype)
        self._warn_dft_size(int(positions.size // 3), r_hat.shape[0])

        def per_freq(E_f, H_f, k):
            N, L = radiation_vectors(E_f, H_f, positions, area, axis, sign, k, r_hat)
            return far_fields_from_NL(N, L, theta_hat, phi_hat, k, r_flat, self.background_index)

        e_theta, e_phi = jax.vmap(per_freq)(E, H, k_vals)
        return e_theta.reshape((-1, *out_shape)), e_phi.reshape((-1, *out_shape))

    def cartesian(
        self,
        arrays: "ArrayContainer",
        x: jax.Array,
        y: jax.Array,
        z: float,
        approx: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        """Far field on a screen at signed distance ``z`` along the normal axis.

        ``x``/``y`` are 1D in-plane coordinate arrays (m) along the two transverse axes. Returns
        ``(E_theta, E_phi)`` each of shape ``(num_freqs, len(x), len(y))``.
        """
        if not (approx and self.far_field_approx):
            raise NotImplementedError("Exact Cartesian (focal-plane) projection is not implemented.")
        axis, _, _, _ = self._geometry()
        sign = self._outward_sign()
        ta, tb = get_transverse_axes(axis)
        X, Y = jnp.meshgrid(jnp.asarray(x), jnp.asarray(y), indexing="ij")
        P = jnp.zeros((*X.shape, 3))
        P = P.at[..., ta].set(X).at[..., tb].set(Y).at[..., axis].set(sign * z)
        rr = jnp.linalg.norm(P, axis=-1)
        r_hat = P / rr[..., None]
        theta = jnp.arccos(jnp.clip(r_hat[..., 2], -1.0, 1.0))
        phi = jnp.arctan2(r_hat[..., 1], r_hat[..., 0])
        return self.spherical(arrays, theta, phi, r=rr)

    def kspace(self, arrays: "ArrayContainer") -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Far field on the FFT direction-cosine grid (fast path).

        Returns ``(ux, uy, E_theta, E_phi)`` (fftshifted), each ``(num_freqs, Nu, Nv)``. Evanescent
        bins (``ux^2 + uy^2 > 1``) are zeroed.
        """
        self._check_far_field()
        if self._config.has_nonuniform_grid:
            raise NotImplementedError("kspace() requires a uniform grid; use spherical() on non-uniform grids.")
        axis, _, _, normal_coord = self._geometry()
        sign = self._outward_sign()
        ta, tb = get_transverse_axes(axis)
        spacing = self._config.uniform_spacing()
        E, H = self._plane_fields(arrays)
        E2 = jnp.squeeze(E, axis=2 + axis)  # (F, 3, Nu, Nv)
        H2 = jnp.squeeze(H, axis=2 + axis)
        k_vals = self._k_values()

        def per_freq(E_f, H_f, k):
            J, M = surface_equivalent_currents(E_f, H_f, axis, sign)
            ku, kv, n_hat_vec, l_hat_vec = radiation_vectors_fft(J, M, spacing, spacing)
            KU, KV = jnp.meshgrid(ku, kv, indexing="ij")
            kt2 = KU**2 + KV**2
            prop = kt2 <= k**2
            kn = sign * jnp.sqrt(jnp.clip(k**2 - kt2, 0.0))
            phase_n = jnp.exp(-1j * kn * normal_coord)
            N = (n_hat_vec * phase_n).reshape(3, -1)
            L = (l_hat_vec * phase_n).reshape(3, -1)
            r_hat = jnp.zeros((*KU.shape, 3)).at[..., ta].set(KU / k).at[..., tb].set(KV / k).at[..., axis].set(kn / k)
            theta = jnp.arccos(jnp.clip(r_hat[..., 2], -1.0, 1.0))
            phi = jnp.arctan2(r_hat[..., 1], r_hat[..., 0])
            _, theta_hat, phi_hat = spherical_basis(theta, phi)
            e_theta, e_phi = far_fields_from_NL(
                N, L, theta_hat.reshape(-1, 3), phi_hat.reshape(-1, 3), k, 1.0, self.background_index
            )
            e_theta = jnp.where(prop, e_theta.reshape(KU.shape), 0.0)
            e_phi = jnp.where(prop, e_phi.reshape(KU.shape), 0.0)

            def shift(a):
                return jnp.fft.fftshift(a, axes=(0, 1))

            return shift(KU / k), shift(KV / k), shift(e_theta), shift(e_phi)

        return jax.vmap(per_freq)(E2, H2, k_vals)

    def diffraction_orders(
        self,
        arrays: "ArrayContainer",
        orders: Sequence[tuple[int, int]],
    ) -> dict[str, jax.Array]:
        """Diffraction-order decomposition for a periodic (Bloch) plane.

        Plane-wave (Bloch) decomposition of the recorded plane: each order ``(m, n)`` maps to the
        FFT bin at transverse wavevector ``k_t = (2 pi m / Lx, 2 pi n / Ly)`` (``L`` = plane extent).
        This is the enriched successor to the deprecated ``DiffractiveDetector`` — it returns signed
        complex amplitudes, angles, per-order power, and an s/p split.

        Args:
            arrays: Simulation arrays holding the recorded state.
            orders: Sequence of integer ``(m, n)`` diffraction orders.

        Returns:
            Dict with (all per ``(num_freqs, num_orders)`` unless noted):
            ``"orders"`` ``(num_orders, 2)``, ``"theta"``, ``"phi"`` (rad), ``"power"`` (eta0-normalized,
            sums to the plane flux over propagating orders), ``"propagating"`` (bool), and the complex
            ``"amplitude_s"`` (TE, E·phi_hat) / ``"amplitude_p"`` (TM, E·theta_hat).
        """
        self._check_far_field()
        if self._config.has_nonuniform_grid:
            raise NotImplementedError("diffraction_orders() requires a uniform grid (periodic unit cell).")
        axis = self._plane_normal_axis()
        ta, tb = get_transverse_axes(axis)
        sign = self._outward_sign()
        spacing = self._config.uniform_spacing()
        E, H = self._plane_fields(arrays)
        E2 = jnp.squeeze(E, axis=2 + axis)  # (F, 3, Nu, Nv)
        H2 = jnp.squeeze(H, axis=2 + axis)
        _, _, Nu, Nv = E2.shape
        Ek = jnp.fft.fft2(E2, axes=(2, 3)) / (Nu * Nv)
        Hk = jnp.fft.fft2(H2, axes=(2, 3)) / (Nu * Nv)
        ku = 2 * jnp.pi * jnp.fft.fftfreq(Nu, spacing)
        kv = 2 * jnp.pi * jnp.fft.fftfreq(Nv, spacing)
        orders_arr = jnp.asarray(orders)
        m_idx = jnp.mod(orders_arr[:, 0], Nu)
        n_idx = jnp.mod(orders_arr[:, 1], Nv)
        ku_o, kv_o = ku[m_idx], kv[n_idx]
        k_vals = self._k_values()

        def per_freq(Ek_f, Hk_f, k):
            E_ord = Ek_f[:, m_idx, n_idx]  # (3, num_orders)
            H_ord = Hk_f[:, m_idx, n_idx]
            kt2 = ku_o**2 + kv_o**2
            prop = kt2 <= k**2
            kz = sign * jnp.sqrt(jnp.clip(k**2 - kt2, 0.0))
            r_hat = (
                jnp.zeros((ku_o.shape[0], 3)).at[:, ta].set(ku_o / k).at[:, tb].set(kv_o / k).at[:, axis].set(kz / k)
            )
            theta = jnp.arccos(jnp.clip(r_hat[:, 2], -1.0, 1.0))
            phi = jnp.arctan2(r_hat[:, 1], r_hat[:, 0])
            _, theta_hat, phi_hat = spherical_basis(theta, phi)
            poynting = jnp.cross(E_ord, jnp.conj(H_ord), axisa=0, axisb=0, axisc=0)  # (3, num_orders)
            power = jnp.where(prop, 0.5 * spacing**2 * Nu * Nv * jnp.real(poynting[axis]) * sign, 0.0)
            amp_s = jnp.sum(E_ord * phi_hat.T, axis=0)
            amp_p = jnp.sum(E_ord * theta_hat.T, axis=0)
            return theta, phi, power, prop, amp_s, amp_p

        theta, phi, power, prop, amp_s, amp_p = jax.vmap(per_freq)(Ek, Hk, k_vals)
        return {
            "orders": orders_arr,
            "theta": theta,
            "phi": phi,
            "power": power,
            "propagating": prop,
            "amplitude_s": amp_s,
            "amplitude_p": amp_p,
        }

    def _warn_dft_size(self, num_cells: int, num_dir: int) -> None:
        if num_cells * num_dir > _DFT_SIZE_WARN:
            warnings.warn(
                f"Far-field DFT over {num_cells} cells x {num_dir} directions "
                f"(~{num_cells * num_dir:.1e} products) may be memory-heavy; reduce the angular grid "
                "or use kspace() (FFT fast path).",
                UserWarning,
                stacklevel=3,
            )


@autoinit
class BoxFarFieldProjector(FarFieldProjector):
    """Surface-equivalence projector recorded on the six faces of the simulation volume.

    A single placed object spanning the volume interior (just inside the PML, via
    :func:`far_field_box`). Records per-face phasors and sums the radiation integral over all six
    faces — full-4pi radiation patterns, RCS, isolated-emitter power. Accessors: :meth:`near_field`,
    :meth:`spherical`, :meth:`radiated_power`, plus inherited :meth:`directivity` /
    :meth:`radar_cross_section`.
    """

    def _shape_dtype_single_time_step(self) -> dict[str, jax.ShapeDtypeStruct]:
        field_dtype = jnp.complex128 if self.dtype == jnp.complex128 else jnp.complex64
        num_freqs = len(self._angular_frequencies)
        out: dict[str, jax.ShapeDtypeStruct] = {}
        for key, axis, _sign, _side in _BOX_FACES:
            face_shape = list(self.grid_shape)
            face_shape[axis] = 1
            out[key] = jax.ShapeDtypeStruct((num_freqs, 6, *face_shape), field_dtype)
        return out

    def _face_slice(self, axis: int, side: str) -> tuple[slice, slice, slice]:
        bounds = self.grid_slice_tuple
        lo, hi = bounds[axis]
        idx = lo if side == "min" else hi - 1
        sl = [slice(b[0], b[1]) for b in bounds]
        sl[axis] = slice(idx, idx + 1)
        return (sl[0], sl[1], sl[2])

    def update(
        self,
        time_step: jax.Array,
        E: jax.Array,
        H: jax.Array,
        state: DetectorState,
        inv_permittivity: jax.Array,
        inv_permeability: jax.Array | float,
    ) -> DetectorState:
        del inv_permittivity, inv_permeability
        if self.scaling_mode not in ("continuous", "pulse"):
            raise Exception(f"Invalid scaling mode: {self.scaling_mode=}")
        factor = self._phasor_factor(time_step)  # (num_freqs,)
        new_state = dict(state)
        for key, axis, _sign, side in _BOX_FACES:
            sl = self._face_slice(axis, side)
            E_f, H_f = E[:, *sl], H[:, *sl]
            EH = jnp.stack([E_f[0], E_f[1], E_f[2], H_f[0], H_f[1], H_f[2]], axis=0)  # (6, *face)
            phasors = factor.reshape((factor.shape[0],) + (1,) * EH.ndim)
            contrib = (EH * phasors)[None, ...]
            if self.inverse:
                new_state[key] = (state[key] - contrib).astype(self.dtype)
            else:
                new_state[key] = (state[key] + contrib).astype(self.dtype)
        return new_state

    def _face_area_box(self, axis: int) -> jax.Array | float:
        grid = self._config.resolved_grid
        if grid is not None:
            return grid.face_area(axis=axis, slice_tuple=self.grid_slice_tuple)
        spacing = self._config.uniform_spacing()
        return spacing * spacing

    def _face_positions(self, axis: int, side: str) -> jax.Array:
        grid = self._config.resolved_grid
        centers = []
        for a in range(3):
            lo, hi = self.grid_slice_tuple[a]
            if grid is not None:
                c = jnp.asarray(grid.centers(a))[lo:hi]
            else:
                spacing = self._config.uniform_spacing()
                c = (jnp.arange(lo, hi) + 0.5) * spacing
            if a == axis:
                c = c[0:1] if side == "min" else c[-1:]
            centers.append(c)
        xx, yy, zz = jnp.meshgrid(centers[0], centers[1], centers[2], indexing="ij")
        return jnp.stack([xx, yy, zz], axis=-1)

    def _face_fields(self, arrays: "ArrayContainer", key: str) -> tuple[jax.Array, jax.Array]:
        phasor = arrays.detector_states[self.name][key]  # (1, F, 6, *face)
        static_scale = self._static_scale()
        return phasor[0, :, :3] / static_scale, phasor[0, :, 3:] / static_scale

    def near_field(self, arrays: "ArrayContainer") -> dict[str, tuple[jax.Array, jax.Array]]:
        """Recorded per-face near fields: ``{face_key: (E, H)}``, each ``(num_freqs, 3, *face)``."""
        return {key: self._face_fields(arrays, key) for key, _a, _s, _side in _BOX_FACES}

    def spherical(
        self,
        arrays: "ArrayContainer",
        theta: jax.Array,
        phi: jax.Array,
        r: jax.Array | float = 1.0,
    ) -> tuple[jax.Array, jax.Array]:
        self._check_far_field()
        k_vals = self._k_values()
        theta = jnp.asarray(theta)
        out_shape = theta.shape
        phi = jnp.broadcast_to(jnp.asarray(phi), out_shape)
        r_hat, theta_hat, phi_hat = spherical_basis(theta, phi)
        r_hat = r_hat.reshape(-1, 3)
        theta_hat = theta_hat.reshape(-1, 3)
        phi_hat = phi_hat.reshape(-1, 3)
        r_flat = jnp.broadcast_to(jnp.asarray(r), out_shape).reshape(-1).astype(theta_hat.dtype)

        face_data = []
        for key, axis, sign, side in _BOX_FACES:
            E, H = self._face_fields(arrays, key)
            face_data.append((axis, sign, self._face_positions(axis, side), self._face_area_box(axis), E, H))

        e_theta_list, e_phi_list = [], []
        for fi in range(len(self.wave_characters)):
            k = k_vals[fi]
            nl = [
                radiation_vectors(E[fi], H[fi], pos, area, axis, sign, k, r_hat)
                for (axis, sign, pos, area, E, H) in face_data
            ]
            N_tot = jnp.sum(jnp.stack([n for n, _l in nl]), axis=0)
            L_tot = jnp.sum(jnp.stack([_l for _n, _l in nl]), axis=0)
            e_theta, e_phi = far_fields_from_NL(N_tot, L_tot, theta_hat, phi_hat, k, r_flat, self.background_index)
            e_theta_list.append(e_theta.reshape(out_shape))
            e_phi_list.append(e_phi.reshape(out_shape))
        return jnp.stack(e_theta_list), jnp.stack(e_phi_list)

    def radiated_power(self, arrays: "ArrayContainer", frequencies: jax.Array | None = None) -> jax.Array:
        """Net power radiated out of the box, per frequency.

        ``½ Σ_faces Re ∮ (E x H*)·n̂_out dA`` from the recorded face phasors — the measured
        (environment-dependent, Purcell-aware) radiated power. Replaces the old
        ``radiated_power_spectrum`` free function.
        """
        del frequencies
        total = jnp.zeros(len(self.wave_characters))
        for key, axis, sign, _side in _BOX_FACES:
            E, H = self._face_fields(arrays, key)  # (F, 3, *face)
            area = self._face_area_box(axis)

            def flux_fi(E_f, H_f, axis=axis, area=area):
                poynting = compute_poynting_flux(E_f, H_f, axis=0)[axis]
                return 0.5 * jnp.real(jnp.sum(poynting * area))

            total = total + sign * jax.vmap(flux_fi)(E, H)
        return total

    def measured_power_spectrum(
        self,
        arrays: "ArrayContainer",
        frequencies: jax.Array | None = None,
    ) -> jax.Array:
        """Total radiated power per frequency — the measured power for ``transmission``."""
        return self.radiated_power(arrays, frequencies=frequencies)


def far_field_box(
    volume: "SimulationVolume",
    boundary_config: "BoundaryConfig",
    wave_characters: Sequence[WaveCharacter],
    config: "SimulationConfig",
    *,
    margin: int = 2,
    name: str = "far_field_box",
    **kwargs,
) -> tuple[BoxFarFieldProjector, "list[GridCoordinateConstraint]"]:
    """Build a :class:`BoxFarFieldProjector` spanning the volume interior, just inside the PML.

    Mirrors :func:`fdtdx.boundary_objects_from_config`: returns the projector plus the grid-coordinate
    constraints that inset each face by that face's boundary thickness + ``margin`` cells (``margin``
    keeps the equivalent-current surface out of the PML's absorbing region).

    Args:
        volume: The simulation volume the box encloses.
        boundary_config: Boundary config (gives per-face thickness via ``get_dict()``).
        wave_characters: Frequencies to record/project.
        config: Simulation config (used to resolve the grid shape for the ``+`` faces).
        margin: Extra cells between each face and the boundary (>= 2 recommended).
        name: Object name.
        **kwargs: Forwarded to ``BoxFarFieldProjector`` (e.g. ``background_index``, ``scaling_mode``).

    Returns:
        ``(box, constraints)`` — add ``box`` to the object list and extend constraints.
    """
    grid = config.resolved_grid
    if grid is not None:
        shape = grid.shape
    else:
        spacing = config.uniform_spacing()
        shape_list: list[int] = []
        for a in range(3):
            n_cells = volume.partial_grid_shape[a]
            if n_cells is not None:
                shape_list.append(int(n_cells))
                continue
            length = volume.partial_real_shape[a]
            if length is None:
                raise ValueError(f"far_field_box: volume axis {a} has neither grid nor real shape.")
            shape_list.append(round(length / spacing))
        shape = tuple(shape_list)
    thickness = boundary_config.get_dict()
    box = BoxFarFieldProjector(name=name, wave_characters=tuple(wave_characters), **kwargs)
    constraints: list = []
    axis_keys = ((0, "min_x", "max_x"), (1, "min_y", "max_y"), (2, "min_z", "max_z"))
    for axis, min_key, max_key in axis_keys:
        lo = thickness[min_key] + margin
        hi = shape[axis] - thickness[max_key] - margin
        if hi <= lo:
            raise ValueError(
                f"far_field_box: axis {axis} interior collapses (lo={lo} >= hi={hi}); reduce margin or PML."
            )
        constraints.append(box.set_grid_coordinates(axes=(axis,), sides=("-",), coordinates=(lo,)))
        constraints.append(box.set_grid_coordinates(axes=(axis,), sides=("+",), coordinates=(hi,)))
    return box, constraints
