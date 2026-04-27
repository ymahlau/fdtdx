"""Integration tests for custom temporal profiles on placed sources."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fdtdx

TEST_OUTPUT_DIR = Path("tests/generated")
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _gaussian_windowed_carrier(config: fdtdx.SimulationConfig, wave: fdtdx.WaveCharacter) -> jax.Array:
    """Return a compact source-like pulse sampled at the simulation cadence."""
    time = jnp.arange(config.time_steps_total, dtype=config.dtype) * config.time_step_duration
    pulse_center = 0.5 * time[-1]
    pulse_width = 8e-15
    envelope = jnp.exp(-((time - pulse_center) ** 2) / (2 * pulse_width**2))
    carrier = jnp.cos(2 * jnp.pi * wave.get_frequency() * (time - pulse_center))
    return (envelope * carrier).astype(config.dtype)


def test_custom_time_signal_profile_source_workflow():
    """CustomTimeSignalProfile works through placement and source wrappers."""
    config = fdtdx.SimulationConfig(
        resolution=50e-9,
        time=500e-15,
        dtype=jnp.float32,
    )
    wave = fdtdx.WaveCharacter(wavelength=1e-6, phase_shift=jnp.pi / 3)
    signal = _gaussian_windowed_carrier(config, wave)

    volume = fdtdx.SimulationVolume(
        partial_grid_shape=(8, 8, 8),
        name="volume",
    )
    source = fdtdx.UniformPlaneSource(
        name="custom_source",
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
        temporal_profile=fdtdx.CustomTimeSignalProfile(
            signal=signal,
            time_step_duration=config.time_step_duration,
            center_wave=wave,
            fwidth=fdtdx.WaveCharacter(frequency=80e12),
        ),
    )
    constraints = [
        source.same_size(volume, axes=(0, 1)),
        source.place_at_center(volume, axes=(0, 1)),
        source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(4,)),
    ]

    object_container, _, _, placed_config, _ = fdtdx.place_objects(
        object_list=[volume, source],
        config=config,
        constraints=constraints,
        key=jax.random.PRNGKey(0),
    )

    placed_source = object_container.sources[0]
    assert isinstance(placed_source.temporal_profile, fdtdx.CustomTimeSignalProfile)

    time, sampled_signal = placed_source.sample_time_signal(placed_config)
    assert len(time) == placed_config.time_steps_total
    assert jnp.allclose(jnp.asarray(sampled_signal), signal, atol=1e-4)

    filename = TEST_OUTPUT_DIR / "test_plot_time_signal_custom_source.png"
    fig = placed_source.plot_time_signal_and_spectrum(
        config=placed_config,
        filename=filename,
    )

    assert fig is not None
    assert filename.exists()
    plt.close("all")
