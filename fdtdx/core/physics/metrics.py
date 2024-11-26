import jax
import jax.numpy as jnp

def compute_energy(
    E: jax.Array,
    H: jax.Array,
    inv_permittivity: jax.Array,
    inv_permeability: jax.Array,
) -> jax.Array:
    abs_E = jnp.sum(jnp.square(E), axis=0)
    energy_E = 0.5 * (1 / inv_permittivity) * abs_E
    
    abs_H = jnp.sum(jnp.square(H), axis=0)
    energy_H = 0.5 * (1 / inv_permeability) * abs_H

    total_energy = energy_E + energy_H
    return total_energy


def normalize_by_energy(
    E: jax.Array,
    H: jax.Array,
    inv_permittivity: jax.Array,
    inv_permeability: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    total_energy = compute_energy(
        E=E,
        H=H,
        inv_permittivity=inv_permittivity,
        inv_permeability=inv_permeability,
    )
    energy_root = jnp.sqrt(jnp.sum(total_energy))
    norm_E = E / energy_root
    norm_H = H / energy_root
    return norm_E, norm_H


def poynting_flux(E: jax.Array, H: jax.Array):
    return jnp.cross(
        jnp.conj(E), 
        H,
        axisa=0,
        axisb=0,
        axisc=0,
    )

