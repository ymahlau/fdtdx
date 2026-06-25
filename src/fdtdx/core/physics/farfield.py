"""Near-to-far-field (NTFF) transformation kernel.

Free functions implementing the surface-equivalence radiation integral, shared by the
planar and box far-field projectors in :mod:`fdtdx.objects.detectors.farfield`.  Like
:mod:`fdtdx.core.physics.metrics`, these operate directly on field arrays and stay
JAX-differentiable so they compose inside an inverse-design pipeline.

Conventions
-----------
* **Time convention**: physics ``exp(-i w t)``; outgoing spherical wave ``exp(+j k r)``.
  ``PhasorDetector`` accumulates against ``exp(+j w t)`` and therefore stores the physics
  complex amplitude ``A`` (with field ``Re[A exp(-i w t)]``), so the projectors consume
  their own recorded phasors with **no conjugation**.
* **eta0-normalized H**: fdtdx stores ``H_fdtdx = eta0 * H_phys``.  Building the magnetic
  radiation vector from ``H_fdtdx`` directly (no ``eta`` factor) yields exactly
  ``eta * N_phys``.  With background index ``n`` (so ``eta = eta0 / n``) the far field is

      E_theta = -P * (L_phi + N_theta / n)
      E_phi   =  P * (L_theta - N_phi / n)

  where ``P = j k exp(+j k r) / (4 pi r)`` and ``N`` is built from ``H_fdtdx``.
* Surface equivalent currents (Balanis): ``J = n_hat x H``, ``M = -n_hat x E`` with
  ``n_hat`` the *outward* surface normal.

The overall sign of the prefactor and the kernel phase is pinned by the plane-wave unit
test (``test_farfield``), which checks the lobe direction and polarisation.
"""

import jax
import jax.numpy as jnp


def spherical_basis(theta: jax.Array, phi: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the spherical unit vectors at observation angles.

    Args:
        theta: Polar angle(s) from the +z axis, radians. Arbitrary shape ``S``.
        phi: Azimuthal angle(s) from the +x axis, radians, broadcastable to ``theta``.

    Returns:
        ``(r_hat, theta_hat, phi_hat)``, each of shape ``(*S, 3)`` with the Cartesian
        ``(x, y, z)`` components last.
    """
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)
    st, ct = jnp.sin(theta), jnp.cos(theta)
    sp, cp = jnp.sin(phi), jnp.cos(phi)
    zero = jnp.zeros_like(st * sp)
    r_hat = jnp.stack([st * cp, st * sp, ct * jnp.ones_like(sp)], axis=-1)
    theta_hat = jnp.stack([ct * cp, ct * sp, -st * jnp.ones_like(sp)], axis=-1)
    phi_hat = jnp.stack([-sp * jnp.ones_like(st), cp * jnp.ones_like(st), zero], axis=-1)
    return r_hat, theta_hat, phi_hat


def surface_equivalent_currents(
    E: jax.Array,
    H: jax.Array,
    normal_axis: int,
    outward_sign: float,
) -> tuple[jax.Array, jax.Array]:
    """Equivalent surface currents ``J = n_hat x H``, ``M = -n_hat x E``.

    Args:
        E: Electric-field phasor, shape ``(3, *spatial)``.
        H: Magnetic-field phasor (eta0-normalized), shape ``(3, *spatial)``.
        normal_axis: Axis index (0/1/2) of the surface normal.
        outward_sign: ``+1`` if the outward normal points along ``+axis``, else ``-1``.

    Returns:
        ``(J, M)`` each of shape ``(3, *spatial)``.
    """
    spatial_ndim = E.ndim - 1
    n_vec = jnp.zeros(3).at[normal_axis].set(outward_sign)
    n_vec = n_vec.reshape((3,) + (1,) * spatial_ndim)
    J = jnp.cross(n_vec, H, axisa=0, axisb=0, axisc=0)
    M = -jnp.cross(n_vec, E, axisa=0, axisb=0, axisc=0)
    return J, M


def radiation_vectors(
    E: jax.Array,
    H: jax.Array,
    positions: jax.Array,
    area: jax.Array | float,
    normal_axis: int,
    outward_sign: float,
    k: jax.Array,
    r_hat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Radiation vectors ``N``, ``L`` for one open surface, by direct DFT.

    Computes ``N = integral J exp(+j k r_hat . r') dA'`` and the analogous ``L`` from ``M``,
    discretised as an area-weighted sum over surface cells.  Callers sum ``N``/``L`` over
    multiple faces (a closed box) before calling :func:`far_fields_from_NL`.

    Args:
        E: Electric-field phasor on the surface, shape ``(3, *spatial)``.
        H: Magnetic-field phasor (eta0-normalized), shape ``(3, *spatial)``.
        positions: Physical cell coordinates, shape ``(*spatial, 3)`` (metres).
        area: Per-cell face area, broadcastable to ``spatial`` (m^2).
        normal_axis: Axis index of the outward surface normal.
        outward_sign: Sign of the outward normal along ``normal_axis``.
        k: Wavenumber ``2 pi n / lambda0`` in the background medium (rad/m).
        r_hat: Observation directions, shape ``(num_dir, 3)``.

    Returns:
        ``(N, L)`` each of shape ``(3, num_dir)``, complex.
    """
    J, M = surface_equivalent_currents(E, H, normal_axis, outward_sign)
    area = jnp.broadcast_to(area, E.shape[1:])
    J = (J * area).reshape(3, -1)
    M = (M * area).reshape(3, -1)
    pos = positions.reshape(-1, 3)
    # Retarded-phase kernel for the exp(-i w t) / outgoing exp(+jkr) convention (Jackson):
    # A(r) ~ exp(+jkr)/r * integral J exp(-j k r_hat . r') dA'.
    phase = jnp.exp(-1j * k * (pos @ r_hat.T))  # (num_cells, num_dir)
    N = J @ phase  # (3, num_dir)
    L = M @ phase
    return N, L


def radiation_vectors_fft(
    J: jax.Array,
    M: jax.Array,
    du: float,
    dv: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Radiation-vector fast path for a uniform plane (2D FFT over the transverse axes).

    For a uniform plane the transverse part of ``integral J exp(-j k r_hat . r') dA'`` is exactly
    an area-weighted 2D DFT of the currents, evaluated at the FFT transverse wavenumbers
    ``ku = 2 pi fftfreq(Nu, du)``.  The caller adds the (direction-dependent) normal-coordinate
    phase, masks evanescent bins, and converts ``(ku, kv)`` to global directions.

    Args:
        J: Electric current ``n_hat x H`` shaped ``(3, Nu, Nv)``.
        M: Magnetic current ``-n_hat x E`` shaped ``(3, Nu, Nv)``.
        du, dv: Uniform transverse cell spacings (m).

    Returns:
        ``(ku, kv, N_hat, L_hat)`` with ``ku`` ``(Nu,)``, ``kv`` ``(Nv,)`` (rad/m) and the
        transverse-only radiation vectors ``N_hat``/``L_hat`` of shape ``(3, Nu, Nv)``.
    """
    Nu, Nv = J.shape[1], J.shape[2]
    dA = du * dv
    n_hat_vec = jnp.fft.fft2(J * dA, axes=(1, 2))
    l_hat_vec = jnp.fft.fft2(M * dA, axes=(1, 2))
    ku = 2 * jnp.pi * jnp.fft.fftfreq(Nu, du)
    kv = 2 * jnp.pi * jnp.fft.fftfreq(Nv, dv)
    return ku, kv, n_hat_vec, l_hat_vec


def far_fields_from_NL(
    N: jax.Array,
    L: jax.Array,
    theta_hat: jax.Array,
    phi_hat: jax.Array,
    k: jax.Array,
    r: jax.Array | float,
    n_background: float,
) -> tuple[jax.Array, jax.Array]:
    """Far-zone ``(E_theta, E_phi)`` from (summed) radiation vectors.

    Args:
        N: Electric radiation vector built from the eta0-normalized H, shape ``(3, num_dir)``.
        L: Magnetic radiation vector built from E, shape ``(3, num_dir)``.
        theta_hat: Polar unit vectors, shape ``(num_dir, 3)``.
        phi_hat: Azimuthal unit vectors, shape ``(num_dir, 3)``.
        k: Wavenumber in the background medium (rad/m).
        r: Observation radius (m). Only sets the global ``1/r`` and ``exp(jkr)`` phase.
        n_background: Background refractive index ``n`` (so ``eta = eta0 / n``).

    Returns:
        ``(E_theta, E_phi)`` each of shape ``(num_dir,)``, complex.
    """
    n_theta = jnp.sum(N * theta_hat.T, axis=0)
    n_phi = jnp.sum(N * phi_hat.T, axis=0)
    l_theta = jnp.sum(L * theta_hat.T, axis=0)
    l_phi = jnp.sum(L * phi_hat.T, axis=0)
    prefactor = 1j * k * jnp.exp(1j * k * r) / (4 * jnp.pi * r)
    e_theta = -prefactor * (l_phi + n_theta / n_background)
    e_phi = prefactor * (l_theta - n_phi / n_background)
    return e_theta, e_phi


def far_field_power_density(
    e_theta: jax.Array,
    e_phi: jax.Array,
    n_background: float,
) -> jax.Array:
    """Radial time-averaged Poynting density ``½ n |E|^2`` at radius ``r``.

    Uses fdtdx's **eta0-normalized** power convention so this is directly comparable to
    ``PhasorDetector.flux_spectrum`` / ``BoxFarFieldProjector.radiated_power`` (which evaluate
    ``½ Re(E x H_norm*)`` without an explicit ``eta0``).  In eta0-normalized fields the wave
    impedance is ``1/n`` (since ``H_norm = n E`` for a plane wave), so the density is
    ``|E|^2 / (2 · 1/n) = ½ n |E|^2`` — a factor ``eta0`` smaller than the SI value, exactly as
    for every other fdtdx power quantity.  The result still carries the far-field ``1/r^2``
    (through ``e_theta``/``e_phi``); multiply by ``r^2`` for radiant intensity ``U(theta, phi)``.
    """
    eta_norm = 1.0 / n_background
    return (jnp.abs(e_theta) ** 2 + jnp.abs(e_phi) ** 2) / (2.0 * eta_norm)


def directivity_from_pattern(
    u: jax.Array,
    theta: jax.Array,
    phi: jax.Array,
) -> jax.Array:
    """Directivity ``D = 4 pi U / P_rad`` from a radiant-intensity pattern on a grid.

    Args:
        u: Radiant intensity ``U(theta, phi)`` (or any quantity proportional to it),
            shape ``(num_theta, num_phi)``.
        theta: 1D polar-angle grid, shape ``(num_theta,)``, radians (typically ``0..pi``).
        phi: 1D azimuth grid, shape ``(num_phi,)``, radians (typically ``0..2pi``).

    Returns:
        Directivity on the same grid, shape ``(num_theta, num_phi)``.
    """
    sin_theta = jnp.sin(theta)[:, None]
    # Average radiant intensity over the sphere: P_rad / (4 pi).
    weight = sin_theta * jnp.ones_like(phi)[None, :]
    avg_u = jnp.sum(u * weight) / jnp.sum(weight)
    return u / avg_u


def radar_cross_section(
    e_theta: jax.Array,
    e_phi: jax.Array,
    r: float,
    n_background: float,
    incident_power_density: jax.Array | float,
) -> jax.Array:
    """Bistatic radar cross section ``sigma = 4 pi r^2 S_scat / S_inc``.

    Args:
        e_theta, e_phi: Scattered far field, shape ``(num_dir,)``.
        r: Observation radius used when computing ``e_theta``/``e_phi`` (m).
        n_background: Background index.
        incident_power_density: Incident plane-wave power density ``|E_inc|^2 / (2 eta)``.

    Returns:
        RCS ``sigma`` (m^2) per direction.
    """
    s_scat = far_field_power_density(e_theta, e_phi, n_background)
    return 4.0 * jnp.pi * r**2 * s_scat / incident_power_density
