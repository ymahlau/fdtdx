"""Unit tests for the dispersion module (Lorentz / Drude / ADE coefficients)."""

import numpy as np
import pytest

from fdtdx.dispersion import (
    CCPRPole,
    DispersionModel,
    DrudePole,
    LorentzPole,
    Pole,
    compute_eps_spectrum_from_coefficients,
    compute_pole_coefficients,
    susceptibility_from_coefficients,
)
from fdtdx.materials import (
    Material,
    compute_allowed_dispersive_coefficients,
    compute_max_dispersive_poles,
)


class TestPoleSubclasses:
    def test_lorentz_pole_parameters(self):
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0)
        assert isinstance(p, Pole)
        assert p.omega_0 == 1e15
        assert p.gamma == 1e13
        assert p.coupling_sq == pytest.approx(2.0 * 1e15**2)

    def test_drude_pole_parameters(self):
        p = DrudePole(plasma_frequency=1e16, damping=1e14)
        assert isinstance(p, Pole)
        assert p.omega_0 == 0.0
        assert p.gamma == 1e14
        assert p.coupling_sq == pytest.approx(1e16**2)


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
        m = DispersionModel(poles=(LorentzPole(resonance_frequency=omega_0, damping=gamma, delta_epsilon=delta_eps),))
        chi = m.susceptibility(0.0)
        assert chi.imag == pytest.approx(0.0, abs=1e-18)
        assert chi.real == pytest.approx(delta_eps)

    def test_lorentz_susceptibility_closed_form(self):
        # Compare model.susceptibility against the hand-written Lorentz formula
        # at several frequencies away from the resonance.
        delta_eps = 1.7
        omega_0 = 1.5e15
        gamma = 5e13
        m = DispersionModel(poles=(LorentzPole(resonance_frequency=omega_0, damping=gamma, delta_epsilon=delta_eps),))
        for omega in (0.3e15, 0.9e15, 1.4e15, 2.5e15):
            expected = (delta_eps * omega_0**2) / (omega_0**2 - omega**2 - 1j * gamma * omega)
            assert m.susceptibility(omega) == pytest.approx(expected, rel=1e-12)

    def test_drude_susceptibility_closed_form(self):
        omega_p = 9e15
        gamma = 1.5e13
        m = DispersionModel(poles=(DrudePole(plasma_frequency=omega_p, damping=gamma),))
        for omega in (0.5e15, 2e15, 5e15):
            expected = -(omega_p**2) / (omega**2 + 1j * gamma * omega)
            assert m.susceptibility(omega) == pytest.approx(expected, rel=1e-12)

    def test_permittivity_includes_eps_inf(self):
        m = DispersionModel(poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0),))
        eps_inf = 2.25
        omega = 0.0
        eps = m.permittivity(omega, eps_inf=eps_inf)
        # At omega=0, chi = delta_epsilon = 2.0 -> eps = 2.25 + 2.0 = 4.25
        assert eps.real == pytest.approx(eps_inf + 2.0)
        assert eps.imag == pytest.approx(0.0, abs=1e-18)


class TestComputePoleCoefficients:
    def test_empty_poles_returns_empty_arrays(self):
        c1, c2, c3, c4 = compute_pole_coefficients((), dt=1e-17)
        assert c1.shape == (0,)
        assert c2.shape == (0,)
        assert c3.shape == (0,)
        assert c4.shape == (0,)

    def test_lorentz_coefficients_closed_form(self):
        p = LorentzPole(resonance_frequency=2e15, damping=3e13, delta_epsilon=1.5)
        dt = 5e-18
        c1, c2, c3, c4 = compute_pole_coefficients((p,), dt=dt)
        denom = 1.0 + 0.5 * p.gamma * dt
        exp_c1 = (2.0 - p.omega_0**2 * dt**2) / denom
        exp_c2 = -(1.0 - 0.5 * p.gamma * dt) / denom
        exp_c3 = (p.coupling_sq * dt**2) / denom
        assert c1[0] == pytest.approx(exp_c1, rel=1e-12)
        assert c2[0] == pytest.approx(exp_c2, rel=1e-12)
        assert c3[0] == pytest.approx(exp_c3, rel=1e-12)
        # Lorentz has no dE/dt coupling, so c4 must be exactly zero.
        assert c4[0] == 0.0

    def test_drude_coefficients_closed_form(self):
        p = DrudePole(plasma_frequency=1e16, damping=1e14)
        dt = 2e-18
        c1, c2, c3, c4 = compute_pole_coefficients((p,), dt=dt)
        denom = 1.0 + 0.5 * p.gamma * dt
        # omega_0 = 0 for Drude -> c1 = 2/denom
        assert c1[0] == pytest.approx(2.0 / denom, rel=1e-12)
        assert c2[0] == pytest.approx(-(1.0 - 0.5 * p.gamma * dt) / denom, rel=1e-12)
        assert c3[0] == pytest.approx(p.coupling_sq * dt**2 / denom, rel=1e-12)
        assert c4[0] == 0.0

    def test_coefficients_physical_regime_c2_near_minus_one(self):
        # For gamma*dt << 1, c2 should be very close to -1 (makes reverse
        # recurrence numerically well-conditioned).
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0)
        _, c2, _, _ = compute_pole_coefficients((p,), dt=1e-17)
        assert abs(c2[0] + 1.0) < 2e-4

    def test_multiple_poles(self):
        poles = (
            LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0),
            DrudePole(plasma_frequency=5e15, damping=1e14),
        )
        c1, _, c3, _ = compute_pole_coefficients(poles, dt=5e-18)
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
        disp = DispersionModel(poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0),))
        m = Material(permittivity=2.25, dispersion=disp)
        assert m.is_dispersive is True
        assert m.dispersion.num_poles == 1


class TestAllowedDispersiveCoefficients:
    def test_all_nondispersive_returns_zeros(self):
        mats = {
            "air": Material(permittivity=1.0),
            "si": Material(permittivity=11.7),
        }
        c1, c2, c3, c4 = compute_allowed_dispersive_coefficients(mats, dt=1e-17, max_num_poles=0)
        assert c1.shape == (2, 0)
        assert c2.shape == (2, 0)
        assert c3.shape == (2, 0)
        assert c4.shape == (2, 0)

    def test_max_num_dispersive_poles_helper(self):
        mats = {
            "air": Material(permittivity=1.0),
            "gold": Material(
                permittivity=1.0,
                dispersion=DispersionModel(
                    poles=(
                        DrudePole(plasma_frequency=1.37e16, damping=1e14),
                        LorentzPole(resonance_frequency=4.1e15, damping=7e14, delta_epsilon=1.0),
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
                dispersion=DispersionModel(
                    poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=1.0),)
                ),
            ),
            "two_pole": Material(
                permittivity=2.0,
                dispersion=DispersionModel(
                    poles=(
                        LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=1.0),
                        DrudePole(plasma_frequency=2e15, damping=1e14),
                    )
                ),
            ),
        }
        c1, c2, c3, c4 = compute_allowed_dispersive_coefficients(mats, dt=1e-17, max_num_poles=2)
        assert c1.shape == (3, 2)
        # Lorentz/Drude materials have no dE/dt coupling -> c4 all zero.
        assert np.all(c4 == 0.0)
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


def _ccpr_chi_direct(q: complex, r: complex, omega: float) -> complex:
    """Reference CCPR pair susceptibility from the raw pole/residue definition
    (exp(-i omega t) convention, Laplace variable s = -i omega)."""
    s = -1j * omega
    return r / (s - q) + np.conjugate(r) / (s - np.conjugate(q))


class TestCCPRPole:
    def test_parameter_conversion_closed_form(self):
        q = complex(-1.0e13, -2.0e15)
        r = complex(1.0e14, 5.0e14)
        p = CCPRPole(pole=q, residue=r)
        assert isinstance(p, Pole)
        assert p.omega_0 == pytest.approx(abs(q))
        assert p.gamma == pytest.approx(-2.0 * q.real)
        assert p.coupling_sq == pytest.approx(-2.0 * (r * q.conjugate()).real)
        assert p.coupling_edot == pytest.approx(2.0 * r.real)

    def test_susceptibility_matches_direct_pole_residue(self):
        # The model susceptibility (unified numerator a - i*omega*b) must equal the
        # raw complex-conjugate pole/residue sum at several frequencies.
        q = complex(-3.0e13, -2.0e15)
        r = complex(-4.0e14, 1.2e15)
        m = DispersionModel(poles=(CCPRPole(pole=q, residue=r),))
        for omega in (0.4e15, 1.1e15, 2.0e15, 3.3e15):
            expected = _ccpr_chi_direct(q, r, omega)
            assert m.susceptibility(omega) == pytest.approx(expected, rel=1e-9)

    def test_real_residue_gives_nonzero_edot_coupling(self):
        # A residue with a non-zero real part is exactly what distinguishes CCPR
        # from Lorentz/Drude — it produces b = coupling_edot != 0.
        p = CCPRPole(pole=complex(-1e13, -2e15), residue=complex(7e14, 3e14))
        assert p.coupling_edot != 0.0
        _, _, _, c4 = compute_pole_coefficients((p,), dt=1e-17)
        assert c4[0] != 0.0

    def test_lorentz_is_ccpr_special_case(self):
        # A Lorentz pole equals a CCPR pole with a purely imaginary residue
        # (b = 0). Construct the matching (q, r) and compare susceptibilities.
        omega_0, gamma, delta_eps = 2.0e15, 5.0e13, 1.7
        omega_d = np.sqrt(omega_0**2 - (gamma / 2) ** 2)
        q = complex(-gamma / 2.0, -omega_d)
        r = complex(0.0, delta_eps * omega_0**2 / (2.0 * omega_d))
        ccpr = DispersionModel(poles=(CCPRPole(pole=q, residue=r),))
        lorentz = DispersionModel(
            poles=(LorentzPole(resonance_frequency=omega_0, damping=gamma, delta_epsilon=delta_eps),)
        )
        # The CCPR residue is purely imaginary => no dE/dt coupling.
        assert ccpr.poles[0].coupling_edot == pytest.approx(0.0, abs=1e-3)
        for omega in (0.5e15, 1.5e15, 2.5e15):
            assert ccpr.susceptibility(omega) == pytest.approx(lorentz.susceptibility(omega), rel=1e-9)

    def test_coefficients_closed_form(self):
        p = CCPRPole(pole=complex(-2.0e13, -1.8e15), residue=complex(3.0e14, -6.0e14))
        dt = 4e-18
        c1, c2, c3, c4 = compute_pole_coefficients((p,), dt=dt)
        denom = 1.0 + 0.5 * p.gamma * dt
        assert c1[0] == pytest.approx((2.0 - p.omega_0**2 * dt**2) / denom, rel=1e-12)
        assert c2[0] == pytest.approx(-(1.0 - 0.5 * p.gamma * dt) / denom, rel=1e-12)
        assert c3[0] == pytest.approx((p.coupling_sq * dt**2 - p.coupling_edot * dt) / denom, rel=1e-12)
        assert c4[0] == pytest.approx((p.coupling_edot * dt) / denom, rel=1e-12)

    def test_susceptibility_from_coefficients_roundtrip(self):
        # susceptibility_from_coefficients inverts (c1..c4) back to the exact
        # continuous (omega_0, gamma, a, b) and re-evaluates chi — so it must
        # reproduce the analytic model susceptibility.
        p = CCPRPole(pole=complex(-2.0e13, -1.8e15), residue=complex(3.0e14, -6.0e14))
        dt = 4e-18
        c1, c2, c3, c4 = compute_pole_coefficients((p,), dt=dt)
        m = DispersionModel(poles=(p,))
        for omega in (0.5e15, 1.3e15, 2.4e15):
            chi = susceptibility_from_coefficients(
                c1=c1[:, None], c2=c2[:, None], c3=c3[:, None], omega=omega, dt=dt, c4=c4[:, None]
            )
            assert complex(chi[0]) == pytest.approx(m.susceptibility(omega), rel=1e-3)

    def test_eps_spectrum_roundtrip_numpy(self):
        # The numpy setup-time spectrum helper (float64) reproduces eps(omega)
        # for a CCPR pole to high precision.
        p = CCPRPole(pole=complex(-2.0e13, -1.8e15), residue=complex(3.0e14, -6.0e14))
        dt = 4e-18
        c1, c2, c3, c4 = compute_pole_coefficients((p,), dt=dt)
        eps_inf = 2.25
        # shape coefficients to (num_poles, 1, 1) and inv_eps to (1, 1)
        c1a, c2a, c3a, c4a = (x[:, None, None] for x in (c1, c2, c3, c4))
        inv_eps = np.full((1, 1), 1.0 / eps_inf)
        omegas = np.array([0.7e15, 1.9e15, 3.0e15])
        eps = compute_eps_spectrum_from_coefficients(c1a, c2a, c3a, inv_eps, omegas, dt, c4=c4a)
        m = DispersionModel(poles=(p,))
        for i, omega in enumerate(omegas):
            expected = eps_inf + m.susceptibility(float(omega))
            assert eps[i] == pytest.approx(expected, rel=1e-9)

    def test_from_critical_point_matches_closed_form(self):
        # from_critical_point maps (A, phi, Omega, Gamma) to (q, r) that reproduce
        # the documented critical-point susceptibility.
        A, phi, Omega, Gamma = 1.3, 0.6, 2.0e15, 8.0e13
        p = CCPRPole.from_critical_point(amplitude=A, phase=phi, resonance_frequency=Omega, damping=Gamma)
        m = DispersionModel(poles=(p,))

        def cp_chi(omega):
            return (
                A
                * Omega
                * (np.exp(1j * phi) / (Omega - omega - 1j * Gamma) + np.exp(-1j * phi) / (Omega + omega + 1j * Gamma))
            )

        for omega in (0.5e15, 1.8e15, 2.6e15):
            assert m.susceptibility(omega) == pytest.approx(cp_chi(omega), rel=1e-9)
