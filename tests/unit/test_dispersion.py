"""Unit tests for the dispersion module (Lorentz / Drude / ADE coefficients)."""

import numpy as np
import pytest

from fdtdx.dispersion import (
    DispersionModel,
    Pole,
    compute_pole_coefficients,
    drude_pole,
    lorentz_pole,
)
from fdtdx.materials import (
    Material,
    compute_allowed_dispersive_coefficients,
    compute_max_dispersive_poles,
)


class TestPoleFactories:
    def test_lorentz_pole_factory(self):
        p = lorentz_pole(omega_0=1e15, gamma=1e13, delta_epsilon=2.0)
        assert isinstance(p, Pole)
        assert p.omega_0 == 1e15
        assert p.gamma == 1e13
        assert p.coupling_sq == pytest.approx(2.0 * 1e15**2)

    def test_drude_pole_factory(self):
        p = drude_pole(omega_p=1e16, gamma=1e14)
        assert p.omega_0 == 0.0
        assert p.gamma == 1e14
        assert p.coupling_sq == pytest.approx(1e16**2)

    def test_lorentz_rejects_nonpositive_omega_0(self):
        with pytest.raises(ValueError):
            lorentz_pole(omega_0=0.0, gamma=1e13, delta_epsilon=1.0)
        with pytest.raises(ValueError):
            lorentz_pole(omega_0=-1e15, gamma=1e13, delta_epsilon=1.0)

    def test_drude_rejects_nonpositive_omega_p(self):
        with pytest.raises(ValueError):
            drude_pole(omega_p=0.0, gamma=1e14)

    def test_lorentz_rejects_negative_gamma(self):
        with pytest.raises(ValueError):
            lorentz_pole(omega_0=1e15, gamma=-1.0, delta_epsilon=1.0)


class TestDispersionModel:
    def test_empty_model_num_poles(self):
        m = DispersionModel(poles=())
        assert m.num_poles == 0
        # susceptibility of an empty model is exactly zero
        assert m.susceptibility(1e15) == 0.0 + 0.0j

    def test_lorentz_susceptibility_at_zero_frequency(self):
        # At omega=0, chi = delta_epsilon
        delta_eps = 3.5
        omega_0 = 2e15
        gamma = 1e13
        m = DispersionModel(poles=(lorentz_pole(omega_0, gamma, delta_eps),))
        chi = m.susceptibility(0.0)
        assert chi.imag == pytest.approx(0.0, abs=1e-18)
        assert chi.real == pytest.approx(delta_eps)

    def test_lorentz_susceptibility_closed_form(self):
        # Compare model.susceptibility against the hand-written Lorentz formula
        # at several frequencies away from the resonance.
        delta_eps = 1.7
        omega_0 = 1.5e15
        gamma = 5e13
        m = DispersionModel(poles=(lorentz_pole(omega_0, gamma, delta_eps),))
        for omega in (0.3e15, 0.9e15, 1.4e15, 2.5e15):
            expected = (delta_eps * omega_0**2) / (omega_0**2 - omega**2 - 1j * gamma * omega)
            assert m.susceptibility(omega) == pytest.approx(expected, rel=1e-12)

    def test_drude_susceptibility_closed_form(self):
        omega_p = 9e15
        gamma = 1.5e13
        m = DispersionModel(poles=(drude_pole(omega_p, gamma),))
        for omega in (0.5e15, 2e15, 5e15):
            expected = -(omega_p**2) / (omega**2 + 1j * gamma * omega)
            assert m.susceptibility(omega) == pytest.approx(expected, rel=1e-12)

    def test_permittivity_includes_eps_inf(self):
        m = DispersionModel(poles=(lorentz_pole(1e15, 1e13, 2.0),))
        eps_inf = 2.25
        omega = 0.0
        eps = m.permittivity(omega, eps_inf=eps_inf)
        # At omega=0, chi = delta_epsilon = 2.0 -> eps = 2.25 + 2.0 = 4.25
        assert eps.real == pytest.approx(eps_inf + 2.0)
        assert eps.imag == pytest.approx(0.0, abs=1e-18)


class TestComputePoleCoefficients:
    def test_empty_poles_returns_empty_arrays(self):
        c1, c2, c3 = compute_pole_coefficients((), dt=1e-17)
        assert c1.shape == (0,)
        assert c2.shape == (0,)
        assert c3.shape == (0,)

    def test_lorentz_coefficients_closed_form(self):
        p = lorentz_pole(omega_0=2e15, gamma=3e13, delta_epsilon=1.5)
        dt = 5e-18
        c1, c2, c3 = compute_pole_coefficients((p,), dt=dt)
        denom = 1.0 + 0.5 * p.gamma * dt
        exp_c1 = (2.0 - p.omega_0**2 * dt**2) / denom
        exp_c2 = -(1.0 - 0.5 * p.gamma * dt) / denom
        exp_c3 = (p.coupling_sq * dt**2) / denom
        assert c1[0] == pytest.approx(exp_c1, rel=1e-12)
        assert c2[0] == pytest.approx(exp_c2, rel=1e-12)
        assert c3[0] == pytest.approx(exp_c3, rel=1e-12)

    def test_drude_coefficients_closed_form(self):
        p = drude_pole(omega_p=1e16, gamma=1e14)
        dt = 2e-18
        c1, c2, c3 = compute_pole_coefficients((p,), dt=dt)
        denom = 1.0 + 0.5 * p.gamma * dt
        # omega_0 = 0 for Drude -> c1 = 2/denom
        assert c1[0] == pytest.approx(2.0 / denom, rel=1e-12)
        assert c2[0] == pytest.approx(-(1.0 - 0.5 * p.gamma * dt) / denom, rel=1e-12)
        assert c3[0] == pytest.approx(p.coupling_sq * dt**2 / denom, rel=1e-12)

    def test_coefficients_physical_regime_c2_near_minus_one(self):
        # For gamma*dt << 1, c2 should be very close to -1 (makes reverse
        # recurrence numerically well-conditioned).
        p = lorentz_pole(omega_0=1e15, gamma=1e13, delta_epsilon=2.0)
        c1, c2, c3 = compute_pole_coefficients((p,), dt=1e-17)
        assert abs(c2[0] + 1.0) < 2e-4

    def test_multiple_poles(self):
        poles = (
            lorentz_pole(omega_0=1e15, gamma=1e13, delta_epsilon=2.0),
            drude_pole(omega_p=5e15, gamma=1e14),
        )
        c1, c2, c3 = compute_pole_coefficients(poles, dt=5e-18)
        assert c1.shape == (2,)
        # c3 for Drude should be non-zero because coupling_sq = omega_p**2
        assert c3[1] > 0


class TestMaterialIsDispersive:
    def test_material_is_not_dispersive_by_default(self):
        m = Material(permittivity=2.25)
        assert m.is_dispersive is False
        assert m.dispersion is None

    def test_material_with_empty_dispersion_is_not_dispersive(self):
        m = Material(permittivity=2.25, dispersion=DispersionModel(poles=()))
        assert m.is_dispersive is False

    def test_material_with_one_pole_is_dispersive(self):
        disp = DispersionModel(poles=(lorentz_pole(1e15, 1e13, 2.0),))
        m = Material(permittivity=2.25, dispersion=disp)
        assert m.is_dispersive is True
        assert m.dispersion.num_poles == 1


class TestAllowedDispersiveCoefficients:
    def test_all_nondispersive_returns_zeros(self):
        mats = {
            "air": Material(permittivity=1.0),
            "si": Material(permittivity=11.7),
        }
        c1, c2, c3 = compute_allowed_dispersive_coefficients(mats, dt=1e-17, max_num_poles=0)
        assert c1.shape == (2, 0)
        assert c2.shape == (2, 0)
        assert c3.shape == (2, 0)

    def test_max_num_dispersive_poles_helper(self):
        mats = {
            "air": Material(permittivity=1.0),
            "gold": Material(
                permittivity=1.0,
                dispersion=DispersionModel(
                    poles=(
                        drude_pole(omega_p=1.37e16, gamma=1e14),
                        lorentz_pole(omega_0=4.1e15, gamma=7e14, delta_epsilon=1.0),
                    )
                ),
            ),
        }
        assert compute_max_dispersive_poles(mats) == 2

    def test_pole_padding_mixed_counts(self):
        mats = {
            "air": Material(permittivity=1.0),  # non-dispersive, 0 poles
            "one_pole": Material(
                permittivity=2.0,
                dispersion=DispersionModel(poles=(lorentz_pole(1e15, 1e13, 1.0),)),
            ),
            "two_pole": Material(
                permittivity=2.0,
                dispersion=DispersionModel(
                    poles=(
                        lorentz_pole(1e15, 1e13, 1.0),
                        drude_pole(2e15, 1e14),
                    )
                ),
            ),
        }
        c1, c2, c3 = compute_allowed_dispersive_coefficients(mats, dt=1e-17, max_num_poles=2)
        assert c1.shape == (3, 2)
        # Non-dispersive material (air) must have all-zero coefficients.
        # Find air in the ordered list — it has the smallest permittivity.
        # Ordered: air (eps=1), one_pole (eps=2), two_pole (eps=2).
        assert np.all(c1[0] == 0.0)
        assert np.all(c2[0] == 0.0)
        assert np.all(c3[0] == 0.0)
        # one_pole has a Lorentz pole in slot 0 and zero padding in slot 1
        assert c3[1, 0] > 0
        assert c3[1, 1] == 0.0
        # two_pole has non-zero coefficients in both slots
        assert c3[2, 0] > 0
        assert c3[2, 1] > 0
