"""Mode sources and mode-overlap detectors under ``config.symmetry``.

A symmetry-reduced run must launch the same guided mode as the full-domain run, restricted to the
kept half. Solving the mode on the reduced cross-section with the mode solver's own symmetric solve
does *not* achieve that: FDTDX rasterizes materials per cell and hands the same array to every
component, while the solver samples them on its staggered Yee grid, which breaks the discrete
mirror symmetry and shifts ``neff`` at first order in the cell size. These tests pin the
mirror-then-solve route that reproduces the full-domain mode instead.

Only mode solves run here, no FDTD time stepping.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx

_SPACING = 25e-9
_WAVELENGTH = 1.55e-6
_TRANSVERSE_CELLS = 40  # 1 um cross-section, even so every symmetric axis splits exactly
_PROPAGATION_CELLS = 8
_CORE_CELLS = (16, 8)  # 400 x 200 nm silicon core, centered => symmetric about both center planes


def _build(symmetry, mode_index=0, mode_symmetry=None, detector=False):
    """Centered Si waveguide with a +x mode source (and optionally a mode-overlap detector)."""
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_SPACING),
        time=5e-15,
        dtype=jnp.float32,
        symmetry=symmetry,
    )
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_PROPAGATION_CELLS, _TRANSVERSE_CELLS, _TRANSVERSE_CELLS))
    objects, constraints = [volume], []
    bound_dict, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=2), volume
    )
    constraints.extend(boundary_constraints)
    objects.extend(bound_dict.values())

    core = fdtdx.UniformMaterialObject(
        name="core",
        partial_grid_shape=(None, _CORE_CELLS[0], _CORE_CELLS[1]),
        material=fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_silicon),
    )
    constraints.extend([core.same_size(volume, axes=(0,)), core.place_at_center(volume, axes=(0, 1, 2))])
    objects.append(core)

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    extra = {} if mode_symmetry is None else {"symmetry": mode_symmetry}
    source = fdtdx.ModePlaneSource(
        name="src",
        partial_grid_shape=(1, None, None),
        wave_character=wave,
        direction="+",
        mode_index=mode_index,
        **extra,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(1, 2)),
            source.place_at_center(volume, axes=(1, 2)),
            source.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(2,)),
        ]
    )
    objects.append(source)

    if detector:
        overlap = fdtdx.ModeOverlapDetector(
            name="det",
            partial_grid_shape=(1, None, None),
            wave_characters=(wave,),
            direction="+",
            mode_index=mode_index,
        )
        constraints.extend(
            [
                overlap.same_size(volume, axes=(1, 2)),
                overlap.place_at_center(volume, axes=(1, 2)),
                overlap.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(5,)),
            ]
        )
        objects.append(overlap)

    return objects, constraints, config


def _apply(symmetry, **kwargs):
    objects, constraints, config = _build(symmetry, **kwargs)
    key = jax.random.PRNGKey(0)
    container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, container, _ = fdtdx.apply_params(arrays, container, params, key=key)
    return container, config


def _by_name(container, name):
    return next(obj for obj in container.objects if obj.name == name)


def _kept_half(array: np.ndarray, symmetry) -> np.ndarray:
    """Upper half along every symmetric axis of a ``(3, Nx, Ny, Nz)`` array."""
    index: list[slice] = [slice(None)] * 4
    for axis in range(3):
        if symmetry[axis] != 0:
            index[1 + axis] = slice(array.shape[1 + axis] // 2, None)
    return array[tuple(index)]


# PEC on y (Ey is normal there -> even) and PMC on z (Ey is tangential -> even): the walls that
# match the fundamental quasi-TE mode of this waveguide.
_HALF_Y = (0, -1, 0)
_HALF_Z = (0, 0, 1)
_QUARTER = (0, -1, 1)


class TestModeSourceUnderSymmetry:
    @pytest.mark.parametrize(
        "symmetry, label",
        [(_HALF_Y, "half-y-PEC"), (_HALF_Z, "half-z-PMC"), (_QUARTER, "quarter")],
    )
    def test_effective_index_matches_full_domain(self, symmetry, label):
        # The reduced source must launch the *same* mode, so its effective index has to agree with
        # the full-domain solve to numerical precision - not merely to within discretization error.
        full = _by_name(_apply((0, 0, 0))[0], "src")
        reduced = _by_name(_apply(symmetry)[0], "src")
        n_full = complex(full._neff).real
        n_reduced = complex(reduced._neff).real
        assert abs(n_reduced - n_full) < 1e-5, f"{label}: neff {n_reduced:.6f} vs full {n_full:.6f}"

    @pytest.mark.parametrize("symmetry", [_HALF_Y, _HALF_Z, _QUARTER])
    def test_profile_matches_full_domain_kept_half(self, symmetry):
        # Up to the documented normalization (unit flux through the plane the source occupies, so
        # sqrt(2**k) larger than the restriction of the full-domain mode) and a global sign, the
        # injected profile is the full-domain mode restricted to the kept half. It is not bit-equal:
        # the parity projection drops the part of the discrete mode that the wall cannot support (a
        # few percent of its norm at this resolution), so compare with a modal-fidelity measure.
        #
        # The amplitude tolerance is loose because the *discrete* mode is only mirror-symmetric to
        # first order in the cell size: the solver samples materials on its staggered grid while
        # FDTDX hands it one cell-centred permittivity array, so the solved mode's flux does not split
        # exactly evenly between the halves. Measured here for the y axis: 0.446/0.554 at 25 nm and
        # 0.473/0.527 at 12.5 nm, so renormalizing to unit flux over the reduced plane inflates the
        # amplitude by 5.9% (25 nm) resp. 2.8% (12.5 nm) for that axis. The z axis splits to 0.2%. See
        # test_magnetic_plane_projection_uses_the_plain_flip for the sharp check, which uses z only.
        full = _by_name(_apply((0, 0, 0))[0], "src")
        reduced = _by_name(_apply(symmetry)[0], "src")
        multiplicity = 2 ** sum(1 for s in symmetry if s != 0)

        for name in ("_E", "_H"):
            expected = _kept_half(np.asarray(getattr(full, name)), symmetry).ravel()
            actual = np.asarray(getattr(reduced, name)).ravel()
            assert actual.shape == expected.shape
            scale = float(np.vdot(expected, actual) / np.vdot(expected, expected))
            assert abs(abs(scale) - np.sqrt(multiplicity)) < 0.15, f"{name}: unexpected amplitude scale {scale:.4f}"
            fidelity = abs(np.vdot(expected, actual)) / (np.linalg.norm(expected) * np.linalg.norm(actual))
            assert fidelity > 0.97, f"{name}: only {fidelity:.4f} overlap with the full-domain mode"

    @pytest.mark.parametrize("symmetry", [_HALF_Y, _QUARTER])
    def test_flux_normalization_is_preserved(self, symmetry):
        # The convention is unchanged by the fix: a mode source carries the same Poynting flux
        # through its own plane as it would in the full domain (unit power through the plane it
        # occupies), which is what keeps mode-overlap amplitudes normalized to 1 for a perfectly
        # transmitted mode. The reduced fields are therefore sqrt(2**k) larger than the restriction
        # of the full-domain mode, not equal to it.
        def plane_flux(source):
            E = np.asarray(source._E)
            H = np.asarray(source._H)
            return float(np.sum(E[1] * H[2] - E[2] * H[1]))  # S_x over the plane

        full_flux = plane_flux(_by_name(_apply((0, 0, 0))[0], "src"))
        reduced_flux = plane_flux(_by_name(_apply(symmetry)[0], "src"))
        assert abs(full_flux) > 0
        ratio = reduced_flux / full_flux
        assert abs(ratio - 1.0) < 2e-3, f"flux through the reduced plane is {ratio:.6f} of the full-domain one"

    def test_magnetic_plane_projection_uses_the_plain_flip(self):
        # The parity projection mirrors the solved mode with the index map of the wall type. A
        # magnetic plane sits half a cell below the reduced domain, so that map is the plain flip;
        # using the electric m±j map here instead (the same map for both wall types) costs an order
        # of magnitude in both metrics below - measured 0.9978 fidelity and a 1.0% amplitude error
        # versus 0.9990 and 0.2% - because it projects onto the parity of a plane half a cell away.
        full = _by_name(_apply((0, 0, 0))[0], "src")
        reduced = _by_name(_apply(_HALF_Z)[0], "src")
        for name in ("_E", "_H"):
            expected = _kept_half(np.asarray(getattr(full, name)), _HALF_Z).ravel()
            actual = np.asarray(getattr(reduced, name)).ravel()
            # sqrt(2) larger than the full-domain restriction (unit power through the reduced plane).
            scale = abs(np.vdot(expected, actual) / np.vdot(expected, expected)) / np.sqrt(2.0)
            assert abs(scale - 1.0) < 5e-3, f"{name}: amplitude is {scale:.4f} of the expected sqrt(2)"
            assert _fidelity(expected, actual) > 0.9985, f"{name}: {_fidelity(expected, actual):.6f}"

    def test_explicit_mode_solver_symmetry_is_ignored_but_kept(self):
        # An explicit mode-solver symmetry tuple no longer does anything under config.symmetry (a
        # warning says so). It must not change the solved mode.
        with_field = _by_name(_apply(_QUARTER, mode_symmetry=(0, 1))[0], "src")
        without_field = _by_name(_apply(_QUARTER)[0], "src")
        assert with_field.symmetry == (0, 1)  # kept verbatim, not auto-derived
        assert np.allclose(np.asarray(with_field._E), np.asarray(without_field._E), atol=1e-6)

    def test_incompatible_wall_type_raises(self):
        # PMC on y makes Ey (normal there) odd, which the fundamental quasi-TE mode is not: the
        # parity projection would annihilate it, so this must fail loudly instead of injecting noise.
        objects, constraints, config = _build((0, 1, 0))
        with pytest.raises(ValueError, match="symmetry imposed by the walls"):
            key = jax.random.PRNGKey(0)
            container, arrays, params, config, _ = fdtdx.place_objects(
                object_list=objects, config=config, constraints=constraints, key=key
            )
            fdtdx.apply_params(arrays, container, params, key=key)


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Phase- and scale-invariant modal overlap of two field arrays (1.0 = same mode)."""
    a, b = a.ravel(), b.ravel()
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestModeOverlapDetectorUnderSymmetry:
    def test_reference_mode_matches_source_mode(self):
        # Source and detector must solve the same reduced mode, otherwise every overlap (and every
        # S-parameter built from it) is off by their mismatch. The detector keeps the complex mode
        # (it is paired with recorded phasors) while the source injects its real part, so compare the
        # real parts and the effective index.
        container, _ = _apply(_QUARTER, detector=True)
        source = _by_name(container, "src")
        detector = _by_name(container, "det")
        source_E = np.asarray(source._E)
        reference_E = np.real(np.asarray(detector._mode_E[0]))
        assert reference_E.shape == source_E.shape
        assert _fidelity(source_E, reference_E) > 0.9999, "reference mode is not the mode being launched"
        ratio = np.linalg.norm(reference_E) / np.linalg.norm(source_E)
        assert abs(ratio - 1.0) < 1e-3, f"reference mode amplitude differs from the source by {ratio:.6f}"
        assert abs(complex(detector._mode_neff[0]).real - complex(source._neff).real) < 1e-6

    def test_reference_mode_matches_full_domain_kept_half(self):
        full = _by_name(_apply((0, 0, 0), detector=True)[0], "det")
        reduced = _by_name(_apply(_QUARTER, detector=True)[0], "det")
        expected = _kept_half(np.asarray(full._mode_E[0]), _QUARTER)
        actual = np.asarray(reduced._mode_E[0])
        assert actual.shape == expected.shape
        assert _fidelity(expected, actual) > 0.97, "reduced reference mode is not the full-domain mode"
        ratio = np.linalg.norm(actual) / np.linalg.norm(expected)
        assert abs(ratio - 2.0) < 0.1, f"unexpected amplitude scale {ratio:.4f}"
