"""Tests for objects/detectors/mode.py - arbitrary / analytic reference modes.

Covers the abstract :class:`BaseModeOverlapDetector` seam and the two new subclasses
that accept a user-provided mode instead of the waveguide mode solver:
:class:`CustomModeOverlapDetector` and :class:`GaussianModeOverlapDetector`, plus the
:func:`gaussian_mode_fields` / :func:`gaussian_mode_function` helpers.
"""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.physics.metrics import compute_poynting_flux
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.dispersion import LorentzPole, compute_pole_coefficients, effective_inv_permittivity
from fdtdx.objects.detectors.mode import (
    BaseModeOverlapDetector,
    CustomModeOverlapDetector,
    GaussianModeOverlapDetector,
    gaussian_mode_fields,
    gaussian_mode_function,
)


@pytest.fixture
def single_frequency():
    """Single optical frequency wave character."""
    return [WaveCharacter(wavelength=1e-6)]


@pytest.fixture
def two_frequencies():
    """Two frequency wave characters."""
    return [WaveCharacter(wavelength=1e-6), WaveCharacter(wavelength=1.5e-6)]


def _constant_mode_function(*, coordinates, frequency, propagation_axis, inv_permittivity):
    """A trivial mode_function: uniform E along axis 0, H along axis 1."""
    del frequency, propagation_axis, inv_permittivity
    amp = jnp.ones(coordinates[0].shape, dtype=jnp.float32)
    mode_E = jnp.zeros((3, *amp.shape), dtype=jnp.float32).at[0].set(amp)
    mode_H = jnp.zeros((3, *amp.shape), dtype=jnp.float32).at[1].set(amp)
    return mode_E, mode_H


def _weighted_flux(det, mode_E, mode_H):
    """Integrated Poynting flux of a stored reference mode over the detector plane."""
    flux = compute_poynting_flux(mode_E.astype(jnp.complex64), mode_H.astype(jnp.complex64))
    s_axis = 0.5 * jnp.real(flux[det.propagation_axis])
    return float(jnp.abs(jnp.sum(s_axis * det._face_area_weights())))


class TestBaseIsAbstract:
    """BaseModeOverlapDetector must not be directly instantiable."""

    def test_isabstract(self):
        """The ABC is not concrete."""
        assert inspect.isabstract(BaseModeOverlapDetector)

    def test_instantiation_raises(self, single_frequency):
        """Constructing the ABC directly fails."""
        with pytest.raises(TypeError):
            BaseModeOverlapDetector(wave_characters=single_frequency)


class TestCustomModeOverlapDetector:
    """CustomModeOverlapDetector evaluates a user-provided mode_function."""

    def test_roundtrip_shapes(self, simulation_config, plane_grid_slice, random_key, single_frequency):
        """apply() stores per-frequency mode fields of the plane's shape."""
        det = CustomModeOverlapDetector(
            wave_characters=single_frequency,
            mode_function=_constant_mode_function,
            normalize=False,
        )
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((3, 8, 8, 8), dtype=jnp.float32), 1.0)

        # one frequency, 3 components, plane shape (8, 8, 1)
        assert det._mode_E.shape == (1, 3, 8, 8, 1)
        assert det._mode_H.shape == (1, 3, 8, 8, 1)

        state = det.init_state()
        state = det.update(
            jnp.array(0),
            jnp.ones((3, 8, 8, 8), dtype=jnp.float32),
            jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 0.5,
            state,
            jnp.ones((3, 8, 8, 8), dtype=jnp.float32),
            1.0,
        )
        overlap = det.compute_overlap(state)
        assert overlap.shape == (1,)
        assert jnp.iscomplexobj(overlap)

    def test_callable_survives_placement(self, simulation_config, plane_grid_slice, random_key, single_frequency):
        """A Python callable stored as a frozen field survives pytree placement/apply."""
        det = CustomModeOverlapDetector(
            wave_characters=single_frequency,
            mode_function=lambda *, coordinates, frequency, propagation_axis, inv_permittivity: (
                jnp.zeros((3, *coordinates[0].shape), dtype=jnp.float32).at[0].set(1.0),
                jnp.zeros((3, *coordinates[0].shape), dtype=jnp.float32).at[1].set(1.0),
            ),
        )
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((3, 8, 8, 8), dtype=jnp.float32), 1.0)
        assert det._mode_E.shape == (1, 3, 8, 8, 1)

    def test_compute_overlap_requires_apply(self, simulation_config, plane_grid_slice, random_key, single_frequency):
        """compute_overlap() without apply() is an error, not a silent zero."""
        det = CustomModeOverlapDetector(
            wave_characters=single_frequency,
            mode_function=_constant_mode_function,
        )
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()
        with pytest.raises(Exception, match="before calling compute_overlap"):
            det.compute_overlap(state)

    def test_multi_frequency_stacks(self, simulation_config, plane_grid_slice, random_key, two_frequencies):
        """Multiple wave characters stack along the leading axis."""
        det = CustomModeOverlapDetector(
            wave_characters=two_frequencies,
            mode_function=_constant_mode_function,
        )
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((3, 8, 8, 8), dtype=jnp.float32), 1.0)
        assert det._mode_E.shape == (2, 3, 8, 8, 1)

        state = det.init_state()
        state = det.update(
            jnp.array(0),
            jnp.ones((3, 8, 8, 8), dtype=jnp.float32),
            jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 0.5,
            state,
            jnp.ones((3, 8, 8, 8), dtype=jnp.float32),
            1.0,
        )
        assert det.compute_overlap(state).shape == (2,)

    def test_normalize_gives_unit_flux(self, simulation_config, plane_grid_slice, random_key, single_frequency):
        """normalize=True renormalizes the mode to unit through-plane flux."""
        det = CustomModeOverlapDetector(
            wave_characters=single_frequency,
            mode_function=_constant_mode_function,
            normalize=True,
        )
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((3, 8, 8, 8), dtype=jnp.float32), 1.0)
        assert _weighted_flux(det, det._mode_E[0], det._mode_H[0]) == pytest.approx(1.0, rel=1e-4)

    def test_no_normalize_keeps_raw_amplitude(self, simulation_config, plane_grid_slice, random_key, single_frequency):
        """normalize=False leaves the supplied amplitude alone."""
        det = CustomModeOverlapDetector(
            wave_characters=single_frequency,
            mode_function=_constant_mode_function,
            normalize=False,
        )
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((3, 8, 8, 8), dtype=jnp.float32), 1.0)
        # raw E·H amplitude over the plane is far from unity (uniform mode, tiny cell areas)
        assert _weighted_flux(det, det._mode_E[0], det._mode_H[0]) != pytest.approx(1.0, rel=1e-2)


class TestGaussianModeOverlapDetector:
    """GaussianModeOverlapDetector builds an analytic Gaussian reference mode."""

    def test_roundtrip(self, simulation_config, plane_grid_slice, random_key, single_frequency):
        """The Gaussian detector places, applies and overlaps end to end."""
        det = GaussianModeOverlapDetector(
            wave_characters=single_frequency,
            mode_radius=3e-7,
            direction="+",
        )
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((3, 8, 8, 8), dtype=jnp.float32), 1.0)
        assert det._mode_E.shape == (1, 3, 8, 8, 1)

        state = det.init_state()
        state = det.update(
            jnp.array(0),
            jnp.ones((3, 8, 8, 8), dtype=jnp.float32),
            jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 0.5,
            state,
            jnp.ones((3, 8, 8, 8), dtype=jnp.float32),
            1.0,
        )
        overlap = det.compute_overlap(state)
        assert overlap.shape == (1,)
        assert jnp.all(jnp.isfinite(jnp.abs(overlap)))

    def test_only_expected_components_nonzero(self, simulation_config, plane_grid_slice, random_key, single_frequency):
        """Default polarization: E along the first transverse axis, H along the second."""
        det = GaussianModeOverlapDetector(
            wave_characters=single_frequency,
            mode_radius=3e-7,
            direction="+",
        )
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((3, 8, 8, 8), dtype=jnp.float32), 1.0)
        mode_E, mode_H = det._mode_E[0], det._mode_H[0]
        # propagation axis is 2 (singleton); transverse axes are (0, 1)
        assert det.propagation_axis == 2
        assert jnp.any(mode_E[0] != 0) and not jnp.any(mode_E[1] != 0) and not jnp.any(mode_E[2] != 0)
        assert jnp.any(mode_H[1] != 0) and not jnp.any(mode_H[0] != 0) and not jnp.any(mode_H[2] != 0)

    def test_matches_gaussian_mode_function(self, simulation_config, plane_grid_slice, random_key, single_frequency):
        """The detector class and the standalone factory produce the same mode."""
        det_cls = GaussianModeOverlapDetector(
            wave_characters=single_frequency,
            mode_radius=3e-7,
            direction="+",
            normalize=False,
        )
        det_cls = det_cls.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det_cls = det_cls.apply(random_key, jnp.ones((3, 8, 8, 8), dtype=jnp.float32), 1.0)

        det_fn = CustomModeOverlapDetector(
            wave_characters=single_frequency,
            mode_function=gaussian_mode_function(radius=3e-7, direction="+"),
            normalize=False,
        )
        det_fn = det_fn.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det_fn = det_fn.apply(random_key, jnp.ones((3, 8, 8, 8), dtype=jnp.float32), 1.0)

        np.testing.assert_allclose(np.array(det_cls._mode_E), np.array(det_fn._mode_E), rtol=1e-6)
        np.testing.assert_allclose(np.array(det_cls._mode_H), np.array(det_fn._mode_H), rtol=1e-6)


class TestDispersionHandling:
    """Custom/Gaussian detectors pick up ε(ω_c) via the shared apply() correction."""

    @staticmethod
    def _dispersive_coeffs(det, pole):
        """Single-pole Lorentz ADE coefficients for a test medium."""
        dt = det._config.time_step_duration
        c1, c2, c3 = compute_pole_coefficients((pole,), dt)

        def mk(a):  # broadcast a per-pole scalar to (num_poles, 1, Nx, Ny, Nz)
            """Build a placed detector for the given mode function."""
            return jnp.full((1, 1, 8, 8, 8), float(a[0]), dtype=jnp.float32)

        return mk(c1), mk(c2), mk(c3), dt

    def test_gaussian_and_custom_use_effective_permittivity(self, simulation_config, plane_grid_slice, random_key):
        """Both analytic detectors see eps(omega_c), not eps_infinity."""
        wc = WaveCharacter(wavelength=1e-6)
        omega = 2.0 * np.pi * wc.get_frequency()
        eps_inf = 2.0
        inv_eps = jnp.full((3, 8, 8, 8), 1.0 / eps_inf, dtype=jnp.float32)
        pole = LorentzPole(resonance_frequency=3.0 * wc.get_frequency(), damping=1e12, delta_epsilon=1.5)

        det = GaussianModeOverlapDetector(wave_characters=[wc], mode_radius=3e-7, direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        c1, c2, c3, dt = self._dispersive_coeffs(det, pole)

        # Independently compute the expected effective index on the detector plane slice,
        # using the very same helper apply() calls internally.
        sl = (slice(None), slice(0, 8), slice(0, 8), slice(0, 1))
        csl = (slice(None), slice(None), slice(0, 8), slice(0, 8), slice(0, 1))
        inv_eps_eff = effective_inv_permittivity(inv_eps[sl], c1[csl], c2[csl], c3[csl], omega, dt)
        n_eff_expected = float(jnp.sqrt(jnp.mean(1.0 / inv_eps_eff)))
        assert abs(n_eff_expected - eps_inf**0.5) > 1e-3  # the pole genuinely shifts the index

        # Gaussian: without dispersion args -> ε∞; with them -> ε(ω_c).
        n_inf = det.apply(random_key, inv_eps, 1.0)._mode_neff[0]
        n_disp = det.apply(random_key, inv_eps, 1.0, dispersive_c1=c1, dispersive_c2=c2, dispersive_c3=c3)._mode_neff[0]
        assert float(jnp.real(n_inf)) == pytest.approx(eps_inf**0.5, rel=1e-5)
        assert float(jnp.real(n_disp)) == pytest.approx(n_eff_expected, rel=1e-4)

        # Custom detector reports the same effective index (its callable saw ε(ω_c)).
        cdet = CustomModeOverlapDetector(wave_characters=[wc], mode_function=_constant_mode_function)
        cdet = cdet.place_on_grid(plane_grid_slice, simulation_config, random_key)
        cdet = cdet.apply(random_key, inv_eps, 1.0, dispersive_c1=c1, dispersive_c2=c2, dispersive_c3=c3)
        assert float(jnp.real(cdet._mode_neff[0])) == pytest.approx(n_eff_expected, rel=1e-4)


class TestGaussianModeFields:
    """gaussian_mode_fields orientation and Poynting-direction physics."""

    @pytest.mark.parametrize("propagation_axis", [0, 1, 2])
    @pytest.mark.parametrize("direction", ["+", "-"])
    def test_orientation_and_poynting_sign(self, propagation_axis, direction):
        # Build a small transverse plane (singleton on the propagation axis).
        """E/H sit on the right axes and the flux follows `direction`."""
        line = jnp.linspace(-1e-7, 1e-7, 6)
        singleton = jnp.array([0.0])
        axis_coords = [line, line, line]
        axis_coords[propagation_axis] = singleton
        coordinates = jnp.meshgrid(*axis_coords, indexing="ij")

        mode_E, mode_H = gaussian_mode_fields(
            coordinates,
            propagation_axis,
            radius=5e-7,
            direction=direction,
            wavelength=1e-6,
            refractive_index=2.0,
        )

        transverse = [a for a in range(3) if a != propagation_axis]
        pol_axis, h_axis = transverse[0], transverse[1]

        # E only along the polarization axis, H only along the other transverse axis.
        for comp in range(3):
            if comp == pol_axis:
                assert jnp.any(mode_E[comp] != 0)
            else:
                assert not jnp.any(mode_E[comp] != 0)
            if comp == h_axis:
                assert jnp.any(mode_H[comp] != 0)
            else:
                assert not jnp.any(mode_H[comp] != 0)

        # |H| = n |E| ratio for the analytic plane wave (n = 2.0 here).
        np.testing.assert_allclose(
            np.abs(np.array(mode_H[h_axis])), 2.0 * np.abs(np.array(mode_E[pol_axis])), rtol=1e-5
        )

        # Poynting vector along the propagation axis has the sign of the direction.
        flux = compute_poynting_flux(mode_E.astype(jnp.complex64), mode_H.astype(jnp.complex64))
        s_total = float(jnp.real(jnp.sum(flux[propagation_axis])))
        if direction == "+":
            assert s_total > 0
        else:
            assert s_total < 0

    def test_polarization_axis_override(self):
        # propagation axis 2 -> transverse (0, 1); force E along axis 1 instead of default 0.
        """polarization_axis picks which transverse axis E points along."""
        line = jnp.linspace(-1e-7, 1e-7, 6)
        coordinates = jnp.meshgrid(line, line, jnp.array([0.0]), indexing="ij")
        mode_E, mode_H = gaussian_mode_fields(
            coordinates, 2, radius=5e-7, direction="+", polarization_axis=1, wavelength=1e-6
        )
        assert jnp.any(mode_E[1] != 0) and not jnp.any(mode_E[0] != 0)
        assert jnp.any(mode_H[0] != 0) and not jnp.any(mode_H[1] != 0)

    def test_invalid_polarization_axis_raises(self):
        """A longitudinal polarization_axis is rejected."""
        line = jnp.linspace(-1e-7, 1e-7, 6)
        coordinates = jnp.meshgrid(line, line, jnp.array([0.0]), indexing="ij")
        with pytest.raises(ValueError, match="transverse"):
            gaussian_mode_fields(coordinates, 2, radius=5e-7, direction="+", polarization_axis=2, wavelength=1e-6)


class TestGaussianModeGeneralization:
    """Arbitrary polarization vector and off-normal propagation angle (like the source)."""

    @staticmethod
    def _plane_coords():
        """A small square plane normal to axis 2."""
        line = jnp.linspace(-1e-7, 1e-7, 6)
        return jnp.meshgrid(line, line, jnp.array([0.0]), indexing="ij")

    def test_fixed_E_polarization_vector(self):
        """E follows the (normalized) fixed polarization vector; here diagonal in-plane."""
        mode_E, mode_H = gaussian_mode_fields(
            self._plane_coords(),
            2,
            radius=5e-7,
            direction="+",
            fixed_E_polarization_vector=(1.0, 1.0, 0.0),
            wavelength=1e-6,
        )
        # E equally along axes 0 and 1, nothing along the propagation axis.
        np.testing.assert_allclose(np.array(mode_E[0]), np.array(mode_E[1]), rtol=1e-6)
        assert not jnp.any(mode_E[2] != 0)
        # H is orthogonal (cross(k, E)) -> also diagonal in plane, still |H| = n|E| (n=1).
        assert jnp.any(mode_H[0] != 0) and jnp.any(mode_H[1] != 0)

    def test_polarization_axis_and_vector_conflict_raises(self):
        """polarization_axis and an explicit vector are mutually exclusive."""
        with pytest.raises(ValueError, match="not both"):
            gaussian_mode_fields(
                self._plane_coords(),
                2,
                radius=5e-7,
                direction="+",
                polarization_axis=0,
                fixed_E_polarization_vector=(1.0, 0.0, 0.0),
                wavelength=1e-6,
            )

    def test_normal_incidence_is_real(self):
        """No tilt -> the reference mode stays real-valued (regression)."""
        mode_E, _ = gaussian_mode_fields(self._plane_coords(), 2, radius=5e-7, direction="+", wavelength=1e-6)
        assert not jnp.iscomplexobj(mode_E)

    def test_tilt_adds_phase_ramp_and_longitudinal_component(self):
        """A tilted beam becomes complex, gains a propagation-axis E component, and ramps."""
        mode_E, _ = gaussian_mode_fields(
            self._plane_coords(),
            2,
            radius=5e-7,
            direction="+",
            azimuth_angle=25.0,
            elevation_angle=15.0,
            refractive_index=1.5,
            wavelength=1e-6,
        )
        assert jnp.iscomplexobj(mode_E)
        # elevation tilt gives the E field a component along the (former) plane normal
        assert jnp.any(jnp.abs(mode_E[2]) > 1e-6)
        # transverse phase ramp: the phase is not constant across the plane
        nonzero = mode_E[0][jnp.abs(mode_E[0]) > 0]
        assert float(jnp.ptp(jnp.angle(nonzero))) > 0.1

    @pytest.mark.parametrize("direction", ["+", "-"])
    def test_tilt_preserves_propagation_sign(self, direction):
        """A tilted mode still carries flux in its nominal direction."""
        mode_E, mode_H = gaussian_mode_fields(
            self._plane_coords(),
            2,
            radius=5e-7,
            direction=direction,
            azimuth_angle=20.0,
            refractive_index=1.0,
            wavelength=1e-6,
        )
        s_total = float(jnp.real(jnp.sum(compute_poynting_flux(mode_E, mode_H)[2])))
        assert (s_total > 0) if direction == "+" else (s_total < 0)

    def test_detector_with_tilt_and_polarization(
        self, simulation_config, plane_grid_slice, random_key, single_frequency
    ):
        """GaussianModeOverlapDetector exposes the same polarization/angle knobs."""
        det = GaussianModeOverlapDetector(
            wave_characters=single_frequency,
            mode_radius=3e-7,
            direction="+",
            fixed_E_polarization_vector=(0.0, 1.0, 0.0),
            azimuth_angle=15.0,
            elevation_angle=5.0,
        )
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((3, 8, 8, 8), dtype=jnp.float32), 1.0)
        assert jnp.iscomplexobj(det._mode_E)

        state = det.init_state()
        state = det.update(
            jnp.array(0),
            jnp.ones((3, 8, 8, 8), dtype=jnp.float32),
            jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 0.5,
            state,
            jnp.ones((3, 8, 8, 8), dtype=jnp.float32),
            1.0,
        )
        overlap = det.compute_overlap(state)
        assert overlap.shape == (1,)
        assert jnp.all(jnp.isfinite(jnp.abs(overlap)))


class TestGaussianRadiusValidation:
    """A non-positive radius is rejected before it can produce NaNs."""

    @staticmethod
    def _plane_coords():
        """A small square plane normal to axis 2."""
        line = jnp.linspace(-1e-7, 1e-7, 6)
        return jnp.meshgrid(line, line, jnp.array([0.0]), indexing="ij")

    @pytest.mark.parametrize("radius", [0.0, -1e-7])
    def test_non_positive_radius_raises(self, radius):
        """A zero or negative radius is rejected."""
        with pytest.raises(ValueError, match="radius must be positive"):
            gaussian_mode_fields(self._plane_coords(), 2, radius=radius, direction="+", wavelength=1e-6)

    @pytest.mark.parametrize("radius", [0.0, -1e-7])
    def test_detector_rejects_non_positive_mode_radius(
        self, radius, simulation_config, plane_grid_slice, random_key, single_frequency
    ):
        """The detector validates mode_radius at placement."""
        det = GaussianModeOverlapDetector(
            wave_characters=single_frequency,
            mode_radius=radius,
            direction="+",
        )
        with pytest.raises(ValueError, match="mode_radius must be positive"):
            det.place_on_grid(plane_grid_slice, simulation_config, random_key)


class TestTiltedEnvelopeProjection:
    """Under tilt the envelope must use the beam's own cross-section, not the plane's.

    A collimated beam of circular cross-section cut by a tilted plane leaves an elliptical
    footprint, stretched by ``1/cos(theta)`` along the tilt direction — the same projection
    ``GaussianPlaneSource`` applies. Measuring that footprint is therefore the check that
    source and detector describe the same beam.
    """

    RADIUS = 5e-7

    def _amplitude_along(self, axis, azimuth):
        """|mode_E| sampled along one transverse axis of a plane normal to axis 2."""
        line = jnp.linspace(-2e-6, 2e-6, 201)
        zero = jnp.array([0.0])
        coords = [zero, zero, zero]
        coords[axis] = line
        mode_E, _ = gaussian_mode_fields(
            jnp.meshgrid(*coords, indexing="ij"),
            2,
            radius=self.RADIUS,
            direction="+",
            azimuth_angle=azimuth,
            refractive_index=1.0,
            wavelength=1e-6,
        )
        return np.asarray(line), np.abs(np.asarray(mode_E)).max(axis=0).ravel()

    @staticmethod
    def _fitted_radius(line, amplitude):
        """Recover the 1/e amplitude radius from the sampled profile.

        Normalized by the peak: under tilt the polarization vector spreads over components,
        so the largest component is below 1 at the beam center.
        """
        positive = line >= 0
        normalized = amplitude[positive] / amplitude.max()
        return float(np.interp(np.exp(-1.0), normalized[::-1], line[positive][::-1]))

    @pytest.mark.parametrize("azimuth", [20.0, 35.0])
    def test_footprint_stretches_by_one_over_cos(self, azimuth):
        """Along the tilt direction the footprint widens by exactly 1/cos(theta)."""
        line, amplitude = self._amplitude_along(0, azimuth)
        expected = self.RADIUS / np.cos(np.deg2rad(azimuth))
        assert self._fitted_radius(line, amplitude) == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("azimuth", [20.0, 35.0])
    def test_footprint_unchanged_across_the_tilt(self, azimuth):
        """The axis orthogonal to the tilt keeps the untilted radius."""
        line, amplitude = self._amplitude_along(1, azimuth)
        assert self._fitted_radius(line, amplitude) == pytest.approx(self.RADIUS, rel=1e-3)

    def test_untilted_footprint_is_circular(self):
        """Regression: with no tilt the projection is the identity."""
        for axis in (0, 1):
            line, amplitude = self._amplitude_along(axis, 0.0)
            assert self._fitted_radius(line, amplitude) == pytest.approx(self.RADIUS, rel=1e-3)


class TestDivergenceAngle:
    """Wavefront curvature at the detector plane."""

    RADIUS = 1e-6
    WAVELENGTH = 1e-6
    WAVENUMBER = 2 * np.pi / WAVELENGTH
    LINE = np.linspace(-1e-6, 1e-6, 41)

    def _fields(self, divergence_angle):
        """Gaussian fields on a fixed square plane at the given divergence angle."""
        line = jnp.asarray(self.LINE)
        coords = jnp.meshgrid(line, line, jnp.array([0.0]), indexing="ij")
        return gaussian_mode_fields(
            coords,
            2,
            radius=self.RADIUS,
            direction="+",
            divergence_angle=divergence_angle,
            refractive_index=1.0,
            wavelength=self.WAVELENGTH,
        )

    def test_zero_angle_is_the_collimated_mode(self):
        """Regression: the default reproduces the real, flat-phase profile."""
        flat, _ = self._fields(0.0)
        assert not jnp.iscomplexobj(flat)

    def test_curvature_is_quadratic_with_the_right_sign(self):
        """Phase follows k r^2 / 2R, increasing outward (diverging) for a positive angle."""
        angle = 10.0
        mode_E, _ = self._fields(angle)
        assert jnp.iscomplexobj(mode_E)

        centre = len(self.LINE) // 2
        phase = np.unwrap(np.angle(np.asarray(mode_E[0])[:, centre, 0]))
        inv_curvature_radius = np.tan(np.deg2rad(angle)) / self.RADIUS
        expected = 0.5 * self.WAVENUMBER * inv_curvature_radius * self.LINE**2
        np.testing.assert_allclose(phase - phase[centre], expected, atol=1e-6)

    def test_converging_is_the_conjugate_of_diverging(self):
        """A negative angle mirrors the phase, leaving the envelope untouched."""
        diverging, _ = self._fields(12.0)
        converging, _ = self._fields(-12.0)
        np.testing.assert_allclose(np.asarray(converging), np.conj(np.asarray(diverging)), rtol=1e-6)

    def test_envelope_is_independent_of_curvature(self):
        """Curvature is pure phase — |E| must match the collimated profile."""
        flat, _ = self._fields(0.0)
        curved, _ = self._fields(15.0)
        np.testing.assert_allclose(np.abs(np.asarray(curved)), np.asarray(flat), rtol=1e-6)

    def test_detector_exposes_divergence_angle(self, simulation_config, plane_grid_slice, random_key, single_frequency):
        """A diverging reference mode reaches the stored fields through the detector."""
        det = GaussianModeOverlapDetector(
            wave_characters=single_frequency,
            mode_radius=1e-7,
            direction="+",
            divergence_angle=8.0,
        )
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        det = det.apply(random_key, jnp.ones((3, 8, 8, 8), dtype=jnp.float32), 1.0)
        assert jnp.iscomplexobj(det._mode_E)
        assert jnp.all(jnp.isfinite(jnp.abs(det._mode_E)))
