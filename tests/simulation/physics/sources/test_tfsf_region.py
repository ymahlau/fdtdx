"""Simulation tests for the TFSF box source ``TFSFPlaneSourceRegion``.

The defining property of a TFSF box is that the *total* field lives inside the
box and only the *scattered* field exists outside. In empty vacuum there is no
scatterer, so the field outside the box must cancel to near zero while the clean
incident plane wave fills the interior. These tests exercise that cancellation
for a fully confined 6-face box, for a periodic/wrap transverse configuration
(2-cap slab), for a confined transverse face, and check gradient finiteness.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

_WAVELENGTH = 1.0e-6
_SPACING = 50e-9


def _build(periodic_axes=(), prop=0, box_cells=24, domain=2.5e-6, pml=8, confined_narrow=None):
    config = fdtdx.SimulationConfig(
        time=60e-15,
        grid=fdtdx.UniformGrid(spacing=_SPACING),
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=None,
    )
    volume = fdtdx.SimulationVolume(partial_real_shape=(domain, domain, domain))

    def _shape(a):
        if a in periodic_axes:
            return None
        if confined_narrow is not None and a == confined_narrow:
            return box_cells // 2
        return box_cells

    pol = (0, 1, 0) if prop == 0 else (1, 0, 0)
    region = fdtdx.TFSFPlaneSourceRegion(
        name="src",
        partial_grid_shape=tuple(_shape(a) for a in range(3)),
        propagation_axis=prop,
        direction="+",
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        fixed_E_polarization_vector=pol,
        periodic_axes=periodic_axes,
    )
    constraints = [region.place_at_center(volume)]
    for a in periodic_axes:
        constraints.append(region.same_size(volume, axes=(a,)))

    override = {}
    for a in periodic_axes:
        ax = "xyz"[a]
        override[f"min_{ax}"] = "periodic"
        override[f"max_{ax}"] = "periodic"
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="pml", thickness=pml, override_types=override)
    bd, bc = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints += bc

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, key = fdtdx.place_objects(
        object_list=[volume, region, *bd.values()], config=config, constraints=constraints, key=key
    )
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    return objects, arrays, params, config, key


def _run_E(objects, arrays, config, key):
    _, out = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
    return np.array(out.fields.E), out


def _energy(E, sl):
    return float(np.mean((E[:, sl[0], sl[1], sl[2]] ** 2).sum(0)))


def _region_slices(objects, prop):
    region = next(o for o in objects.objects if isinstance(o, fdtdx.TFSFPlaneSourceRegion))
    box = [region.grid_slice_tuple[i] for i in range(3)]
    inside = tuple(slice(b[0] + 3, b[1] - 3) for b in box)
    outside = list(inside)
    b = box[prop]
    outside[prop] = slice(b[0] - 5, b[0] - 2)  # scattered slab just outside the min cap
    return inside, tuple(outside)


def test_empty_box_cancellation_six_faces():
    """A fully confined vacuum box: strong field inside, near-zero scattered field outside."""
    objects, arrays, _params, config, key = _build(periodic_axes=(), prop=0)
    E, _ = _run_E(objects, arrays, config, key)
    assert np.all(np.isfinite(E))
    inside_sl, outside_sl = _region_slices(objects, 0)
    inside = _energy(E, inside_sl)
    outside = _energy(E, outside_sl)
    assert inside > 1e-3, "incident wave should fill the box interior"
    assert outside / inside < 0.05, f"scattered field outside box too large: {outside / inside:.3e}"


def test_periodic_slab_cancellation():
    """Periodic on both transverse axes (2-cap slab): confinement along propagation axis."""
    objects, arrays, _params, config, key = _build(periodic_axes=(1, 2), prop=0)
    E, _ = _run_E(objects, arrays, config, key)
    assert np.all(np.isfinite(E))
    inside_sl, outside_sl = _region_slices(objects, 0)
    inside = _energy(E, inside_sl)
    outside = _energy(E, outside_sl)
    assert inside > 1e-3
    assert outside / inside < 1e-2, f"scattered field leaked past the caps: {outside / inside:.3e}"


def test_confined_transverse_face():
    """Box narrower than the domain along a transverse axis with PML there.

    The transverse side faces must confine the total field: the region beside
    the box (along the narrow transverse axis) should carry only scattered field.
    """
    objects, arrays, _params, config, key = _build(periodic_axes=(), prop=0, confined_narrow=1)
    E, _ = _run_E(objects, arrays, config, key)
    assert np.all(np.isfinite(E))
    region = next(o for o in objects.objects if isinstance(o, fdtdx.TFSFPlaneSourceRegion))
    box = [region.grid_slice_tuple[i] for i in range(3)]
    inside = tuple(slice(b[0] + 3, b[1] - 3) for b in box)
    beside = list(inside)
    beside[1] = slice(box[1][0] - 5, box[1][0] - 2)  # just outside the min-y face
    beside = tuple(beside)
    e_in = _energy(E, inside)
    e_side = _energy(E, beside)
    assert e_in > 1e-3
    assert e_side / e_in < 0.1, f"total field leaked past the transverse face: {e_side / e_in:.3e}"


def test_dispersive_background_filter_and_run():
    """Box in a broadband Lorentz medium: each face builds an H-side impedance
    filter and the simulation stays finite (dispersive parity with the plane source)."""
    config = fdtdx.SimulationConfig(
        time=40e-15,
        grid=fdtdx.UniformGrid(spacing=_SPACING),
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=None,
    )
    volume = fdtdx.SimulationVolume(partial_real_shape=(1.8e-6, 1.8e-6, 1.8e-6))

    center_freq = fdtdx.constants.c / _WAVELENGTH
    profile = fdtdx.GaussianPulseProfile(
        spectral_width=fdtdx.WaveCharacter(frequency=0.15 * center_freq),
        center_wave=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
    )
    region = fdtdx.TFSFPlaneSourceRegion(
        name="src",
        partial_grid_shape=(16, 16, 16),
        propagation_axis=0,
        direction="+",
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        fixed_E_polarization_vector=(0, 1, 0),
        temporal_profile=profile,
    )
    material = fdtdx.Material(
        permittivity=1.0,
        dispersion=fdtdx.DispersionModel(
            poles=(
                fdtdx.LorentzPole(resonance_frequency=1.4 * 2 * np.pi * center_freq, damping=5e12, delta_epsilon=1.5),
            )
        ),
    )
    background = fdtdx.UniformMaterialObject(material=material)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="pml", thickness=6)
    bd, bc = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints = [background.same_size(volume), region.place_at_center(volume), *bc]

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, key = fdtdx.place_objects(
        object_list=[volume, background, region, *bd.values()], config=config, constraints=constraints, key=key
    )
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)

    placed = next(o for o in objects.objects if isinstance(o, fdtdx.TFSFPlaneSourceRegion))
    assert len(placed._face_H_filter) == 6
    assert all(f is not None for f in placed._face_H_filter), "dispersive background must populate per-face H filters"

    _, out = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
    assert np.all(np.isfinite(np.array(out.fields.E)))


def test_no_corner_energy_accumulation():
    """The box corners (where propagation caps meet confinement faces) must not
    accumulate energy.

    Regression guard for the staggered connecting condition: a naive TFSF box that
    co-locates the tangential-E and tangential-H corrections on the face plane
    leaves an uncancelled, non-radiating magnetic correction at the box edges. In
    lossless vacuum that residual is quasi-static, so it never leaves and — under a
    sustained (CW) drive — the corners run many times hotter than the interior
    incident wave. With the corrections placed on their proper Yee half-cells the
    corners stay at or below the interior level.

    Mirrors the reported failure mode: E polarized along the periodic/wrap axis (z),
    propagation +x, confined along y. The pre-fix code gives corner/interior ~15x;
    the fix keeps it below ~1.
    """
    spacing = 15e-9
    pml = 8
    domain = 480e-9 + 2 * pml * spacing
    box_cells = 16

    config = fdtdx.SimulationConfig(
        time=50e-15,
        grid=fdtdx.UniformGrid(spacing=spacing),
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=None,
    )
    volume = fdtdx.SimulationVolume(partial_real_shape=(domain, domain, 2 * spacing))
    region = fdtdx.TFSFPlaneSourceRegion(
        name="src",
        partial_grid_shape=(box_cells, box_cells, None),
        propagation_axis=0,
        direction="+",
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        fixed_E_polarization_vector=(0, 0, 1),  # E along the periodic/wrap axis
        periodic_axes=(2,),
    )
    override = {"min_z": "periodic", "max_z": "periodic"}
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="pml", thickness=pml, override_types=override)
    bd, bc = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints = [region.place_at_center(volume), region.same_size(volume, axes=(2,)), *bc]

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, key = fdtdx.place_objects(
        object_list=[volume, region, *bd.values()], config=config, constraints=constraints, key=key
    )
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    _, out = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)

    E = np.array(out.fields.E)
    H = np.array(out.fields.H)
    assert np.all(np.isfinite(E)) and np.all(np.isfinite(H))
    energy = (E**2).sum(0) + (H**2).sum(0)  # total EM energy density (eta0-normalized)

    placed = next(o for o in objects.objects if isinstance(o, fdtdx.TFSFPlaneSourceRegion))
    (x0, x1), (y0, y1), _ = [placed.grid_slice_tuple[i] for i in range(3)]
    interior = energy[x0 + 4 : x1 - 4, y0 + 4 : y1 - 4, :].mean()
    # peak over 3x3 blocks straddling each of the four cap/confinement corners
    corner_peak = max(
        energy[sx, sy, :].max()
        for sx in (slice(x0 - 1, x0 + 2), slice(x1 - 2, x1 + 1))
        for sy in (slice(y0 - 1, y0 + 2), slice(y1 - 2, y1 + 1))
    )
    assert interior > 1e-3, "incident wave should fill the box interior"
    ratio = corner_peak / interior
    assert ratio < 3.0, f"box corners accumulate energy: corner/interior = {ratio:.2f} (pre-fix ~15x)"


def test_region_gradient_finite():
    """Differentiate a field-energy loss through the box source (checkpointed path)."""
    objects, arrays, _params, config, key = _build(periodic_axes=(), prop=0, box_cells=12, domain=1.4e-6, pml=5)
    config = config.aset("gradient_config", fdtdx.GradientConfig(method="checkpointed", num_checkpoints=3))

    def loss_fn(inv_perm):
        a = arrays.aset("inv_permittivities", inv_perm)
        _, out = fdtdx.run_fdtd(arrays=a, objects=objects, config=config, key=key, show_progress=False)
        return jnp.sum(out.fields.E**2)

    val, grad = jax.value_and_grad(loss_fn)(arrays.inv_permittivities)
    assert jnp.isfinite(val)
    assert jnp.all(jnp.isfinite(grad))
