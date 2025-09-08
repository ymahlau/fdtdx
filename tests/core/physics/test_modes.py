import jax
import jax.numpy as jnp

from fdtdx.core.physics.modes import compute_mode
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.units import m, V, A

def test_modes_basic():
    inv_perm = jnp.ones((200, 200, 1))
    inv_perm = inv_perm.at[75:125, 75:125].set(1 / 12)
    
    mode_E, mode_H, eff_idx = compute_mode(
        wave_character=WaveCharacter(wavelength=1.55e-6*m),
        inv_permittivities=inv_perm,
        inv_permeabilities=1.0,
        resolution=10e-9 * m,
        direction="+",
    )
    assert isinstance(eff_idx, jax.Array)
    assert mode_E.unit.dim == (V/m).unit.dim
    assert mode_H.unit.dim == (A/m).unit.dim
    
    
    
    

