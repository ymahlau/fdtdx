"""Physics test: CustomProfilePlaneSource injects the user-specified transverse profile."""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

_WL = 1e-6
_RES = 50e-9
_PML = 8


def test_custom_profile_is_injected():
    """The recorded transverse |Ex| just past the source matches the supplied super-Gaussian."""
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=40e-15, dtype=jnp.float32)
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_real_shape=(3e-6, 3e-6, 2e-6))
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML,
        override_types={"min_x": "periodic", "max_x": "periodic", "min_y": "periodic", "max_y": "periodic"},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WL)
    radius, order = 0.8e-6, 2.0  # gentle, well-resolved flat-top edge

    # a user-supplied profile callable (a flat-top super-Gaussian); any JAX function works
    def super_gaussian(t0, t1):
        return jnp.exp(-(((t0**2 + t1**2) / radius**2) ** order))

    source = fdtdx.CustomProfilePlaneSource(
        name="src",
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
        profile_function=super_gaussian,
        normalize_by_energy=False,  # inject the raw profile so the recorded shape is directly comparable
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_PML + 2,)),
        ]
    )
    objects.append(source)

    det = fdtdx.PhasorDetector(
        name="plane",
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        components=("Ex",),
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 1)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_PML + 3,)),  # 1 cell downstream
        ]
    )
    objects.append(det)

    key = jax.random.PRNGKey(0)
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key)

    rec = np.abs(np.squeeze(np.array(arrays.detector_states["plane"]["phasor"][0, 0, 0])))  # (nx, ny)
    rec = rec / rec.max()

    n = rec.shape[0]
    coords = (np.arange(n) - n / 2 + 0.5) * _RES  # centered transverse cell centers
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    expected = np.exp(-(((X**2 + Y**2) / radius**2) ** order))
    expected = expected / expected.max()

    r_grid = np.sqrt(X**2 + Y**2)

    # The recorded transverse field is the supplied custom super-Gaussian — pinned by three properties
    # that together exclude both a uniform and a Gaussian source:
    assert rec[n // 2, n // 2] > 0.95  # (1) peaked at center
    # (2) localized, i.e. NOT a uniform source: the field has decayed to ~0 well before the domain edge.
    assert rec[r_grid > 1.6 * radius].max() < 0.1, "beam is not localized (looks uniform)"
    # (3) flat-topped, i.e. NOT a Gaussian: at r ~ radius/2 the order-2 super-Gaussian is ~0.94 ideally
    # (recorded ~0.86 after the grid depression), whereas a Gaussian of the same radius would be only
    # exp(-0.25) ~ 0.78 (~0.71 depressed). The 0.82 threshold cleanly separates the two.
    ring = (r_grid > 0.45 * radius) & (r_grid < 0.55 * radius)
    assert rec[ring].mean() > 0.82, f"profile is not flat-topped: ring mean {rec[ring].mean():.3f}"

    # Quantitative shape match over the body (RMS — discretization-limited at 20 cells/lambda; the
    # staggered grid + TFSF injection of a structured profile leave a few-% systematic residual).
    body = expected > 0.5
    rms = float(np.sqrt(np.mean((rec - expected)[body] ** 2)))
    assert rms < 0.08, f"profile RMS deviation {rms:.3f}"
