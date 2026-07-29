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
    compute_pole_coefficients_per_axis,
    compute_pole_coefficients_tensor,
    compute_pole_delta_coefficients_per_axis,
    compute_pole_delta_coefficients_tensor,
    susceptibility_from_coefficients,
    to_delta_form,
    to_observer_form,
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
        c1, c2, c3, c4, _ = compute_pole_coefficients((), dt=1e-17)
        assert c1.shape == (0,)
        assert c2.shape == (0,)
        assert c3.shape == (0,)
        assert c4.shape == (0,)

    def test_lorentz_coefficients_closed_form(self):
        p = LorentzPole(resonance_frequency=2e15, damping=3e13, delta_epsilon=1.5)
        dt = 5e-18
        c1, c2, c3, c4, _ = compute_pole_coefficients((p,), dt=dt)
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
        c1, c2, c3, c4, _ = compute_pole_coefficients((p,), dt=dt)
        denom = 1.0 + 0.5 * p.gamma * dt
        # omega_0 = 0 for Drude -> c1 = 2/denom
        assert c1[0] == pytest.approx(2.0 / denom, rel=1e-12)
        assert c2[0] == pytest.approx(-(1.0 - 0.5 * p.gamma * dt) / denom, rel=1e-12)
        assert c3[0] == pytest.approx(p.coupling_sq * dt**2 / denom, rel=1e-12)
        assert c4[0] == 0.0

    def test_coefficients_physical_regime_c2_near_minus_one(self):
        # For gamma*dt << 1, c2 should be very close to -1 (the near-lossless
        # oscillator limit).
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0)
        _, c2, _, _, _ = compute_pole_coefficients((p,), dt=1e-17)
        assert abs(c2[0] + 1.0) < 2e-4

    def test_multiple_poles(self):
        poles = (
            LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0),
            DrudePole(plasma_frequency=5e15, damping=1e14),
        )
        c1, _, c3, _, _ = compute_pole_coefficients(poles, dt=5e-18)
        assert c1.shape == (2,)
        # c3 for Drude should be non-zero because coupling_sq = omega_p**2
        assert c3[1] > 0

    def test_large_gamma_dt_is_accepted(self):
        # gamma * dt >= 2 drives c2 to 0 (and positive beyond) but keeps the
        # forward recurrence inside the unit circle: |c2| < 1 holds for every
        # gamma * dt > 0, and the binding Jury bound is omega_0 * dt < 2. It is
        # therefore not rejected (it used to be, as a conditioning bound for the
        # reverse-time ADE update, which no longer exists).
        # Heavily overdamped (gamma >> omega_0) so that gamma * dt == 2 while
        # omega_0 * dt stays far below the Jury bound.
        p = LorentzPole(resonance_frequency=1e12, damping=1e16, delta_epsilon=2.0)
        dt = 2.0 / p.gamma
        assert p.omega_0 * dt < 2.0
        c1, c2, c3, _, _ = compute_pole_coefficients((p,), dt=dt)
        assert np.all(np.isfinite(c1)) and np.all(np.isfinite(c3))
        assert np.allclose(c2, 0.0), "gamma * dt == 2 must put c2 exactly at zero"

    def test_omega0_dt_at_least_two_raises(self):
        # omega_0 * dt >= 2 violates the forward Jury bound |c1| < 1 - c2 (the
        # recurrence roots leave the unit circle) even when gamma * dt is tiny.
        p = LorentzPole(resonance_frequency=1e15, damping=1e10, delta_epsilon=2.0)
        with pytest.raises(ValueError, match=r"omega_0 \* dt"):
            compute_pole_coefficients((p,), dt=2.0 / p.omega_0)

    def test_omega0_dt_just_below_two_is_ok(self):
        # Just under the bound must not raise.
        p = LorentzPole(resonance_frequency=1e15, damping=1e10, delta_epsilon=2.0)
        c1, _, _, _, _ = compute_pole_coefficients((p,), dt=1.9 / p.omega_0)
        assert np.isfinite(c1[0])

    def test_zero_coupling_pole_skips_bounds(self):
        # A Lorentz pole with delta_epsilon = 0 contributes nothing (coupling_sq =
        # 0, so c3 = 0 and the polarization stays zero); its unused omega_0 / gamma
        # must not trip the stability bounds even when both products exceed 2.
        p = LorentzPole(resonance_frequency=3e17, damping=3e17, delta_epsilon=0.0)
        c1, c2, c3, c4, _ = compute_pole_coefficients((p,), dt=1e-17)
        assert c3[0] == 0.0 and c4[0] == 0.0
        assert np.isfinite(c1[0]) and np.isfinite(c2[0])


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
        a1, a0, b1, c4, _ = compute_allowed_dispersive_coefficients(mats, dt=1e-17, max_num_poles=0, num_components=1)
        assert a1.shape == (2, 0, 1)
        assert a0.shape == (2, 0, 1)
        assert b1.shape == (2, 0, 1)
        assert c4.shape == (2, 0, 1)

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
        c1, c2, c3, c4, _ = compute_allowed_dispersive_coefficients(mats, dt=1e-17, max_num_poles=2, num_components=1)
        assert c1.shape == (3, 2, 1)
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
        _, _, _, c4, _ = compute_pole_coefficients((p,), dt=1e-17)
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
        """Closed form of the default ``"centered_edot"`` scheme: the b*dE/dt term is
        centred, so it splits symmetrically between c4 and c5 and leaves c3 = a*dt^2/D.
        The legacy ``"central"`` closed form is pinned separately in
        :class:`TestIntegratorSchemes`."""
        p = CCPRPole(pole=complex(-2.0e13, -1.8e15), residue=complex(3.0e14, -6.0e14))
        assert p.integrator == "centered_edot"
        dt = 4e-18
        c1, c2, c3, c4, c5 = compute_pole_coefficients((p,), dt=dt)
        denom = 1.0 + 0.5 * p.gamma * dt
        assert c1[0] == pytest.approx((2.0 - p.omega_0**2 * dt**2) / denom, rel=1e-12)
        assert c2[0] == pytest.approx(-(1.0 - 0.5 * p.gamma * dt) / denom, rel=1e-12)
        assert c3[0] == pytest.approx((p.coupling_sq * dt**2) / denom, rel=1e-12)
        assert c4[0] == pytest.approx((p.coupling_edot * dt) / (2.0 * denom), rel=1e-12)
        assert c5[0] == pytest.approx(-(p.coupling_edot * dt) / (2.0 * denom), rel=1e-12)

    def test_susceptibility_from_coefficients_roundtrip(self):
        # susceptibility_from_coefficients evaluates the discrete transfer function
        # of the stored observer coefficients, which approximates the analytic model
        # susceptibility to the scheme's O((omega*dt)^2) truncation.
        p = CCPRPole(pole=complex(-2.0e13, -1.8e15), residue=complex(3.0e14, -6.0e14))
        dt = 4e-18
        a1, a0, b1, c4, b0 = to_delta_form(*compute_pole_coefficients((p,), dt=dt))
        m = DispersionModel(poles=(p,))
        for omega in (0.5e15, 1.3e15, 2.4e15):
            chi = susceptibility_from_coefficients(
                a1=a1[:, None],
                a0=a0[:, None],
                b1=b1[:, None],
                omega=omega,
                dt=dt,
                c4=c4[:, None],
                b0=b0[:, None],
            )
            assert complex(chi[0]) == pytest.approx(m.susceptibility(omega), rel=1e-3)

    def test_eps_spectrum_roundtrip_numpy(self):
        # The numpy setup-time spectrum helper (float64) reproduces eps(omega)
        # for a CCPR pole to high precision.
        p = CCPRPole(pole=complex(-2.0e13, -1.8e15), residue=complex(3.0e14, -6.0e14))
        dt = 4e-18
        a1, a0, b1, c4, b0 = to_delta_form(*compute_pole_coefficients((p,), dt=dt))
        eps_inf = 2.25
        # shape coefficients to (num_poles, 1, 1) and inv_eps to (1, 1)
        a1a, a0a, b1a, c4a, b0a = (x[:, None, None] for x in (a1, a0, b1, c4, b0))
        inv_eps = np.full((1, 1), 1.0 / eps_inf)
        omegas = np.array([0.7e15, 1.9e15, 3.0e15])
        eps = compute_eps_spectrum_from_coefficients(a1a, a0a, b1a, inv_eps, omegas, dt, c4=c4a, b0=b0a)
        m = DispersionModel(poles=(p,))
        for i, omega in enumerate(omegas):
            # The helper returns the DISCRETE eps the grid realizes, so it differs
            # from the continuum model by the scheme's O((omega*dt)^2) truncation
            # (~1e-5 relative here at omega*dt ~ 1.2e-2).
            expected = eps_inf + m.susceptibility(float(omega))
            assert eps[i] == pytest.approx(expected, rel=1e-4)

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


class TestPerAxisPoles:
    """Per-axis (diagonally anisotropic) pole parameters."""

    def test_scalar_pole_is_isotropic_and_axes_uniform(self):
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0)
        assert p.is_isotropic
        assert p.omega_0_axes == (1e15, 1e15, 1e15)
        assert p.gamma_axes == (1e13, 1e13, 1e13)
        assert p.coupling_sq_axes == pytest.approx((2e30, 2e30, 2e30))
        assert p.coupling_edot_axes == (0.0, 0.0, 0.0)

    def test_per_axis_lorentz_axes_values(self):
        p = LorentzPole(
            resonance_frequency=(1e15, 2e15, 3e15),
            damping=(1e13, 2e13, 3e13),
            delta_epsilon=(1.0, 2.0, 0.0),
        )
        assert not p.is_isotropic
        assert p.omega_0_axes == (1e15, 2e15, 3e15)
        assert p.gamma_axes == (1e13, 2e13, 3e13)
        assert p.coupling_sq_axes == pytest.approx((1e30, 2.0 * 4e30, 0.0))

    def test_per_axis_drude_axes_values(self):
        p = DrudePole(plasma_frequency=(2e15, 0.0, 0.0), damping=1e13)
        assert not p.is_isotropic
        assert p.omega_0_axes == (0.0, 0.0, 0.0)
        assert p.coupling_sq_axes == pytest.approx((4e30, 0.0, 0.0))

    def test_per_axis_ccpr_axes_values(self):
        qx = complex(-1.0e13, -2.0e15)
        qy = complex(-2.0e13, -1.5e15)
        rx = complex(1.0e14, 5.0e14)
        ry = complex(-3.0e14, 2.0e14)
        p = CCPRPole(pole=(qx, qy, qy), residue=(rx, ry, ry))
        assert not p.is_isotropic
        assert p.omega_0_axes[0] == pytest.approx(abs(qx))
        assert p.omega_0_axes[1] == pytest.approx(abs(qy))
        assert p.gamma_axes[0] == pytest.approx(-2.0 * qx.real)
        assert p.coupling_sq_axes[1] == pytest.approx(-2.0 * (ry * qy.conjugate()).real)
        assert p.coupling_edot_axes[0] == pytest.approx(2.0 * rx.real)
        assert p.coupling_edot_axes[1] == pytest.approx(2.0 * ry.real)

    def test_scalar_accessors_raise_for_per_axis_pole(self):
        p = LorentzPole(resonance_frequency=(1e15, 2e15, 3e15), damping=1e13, delta_epsilon=2.0)
        with pytest.raises(ValueError, match="omega_0_axes"):
            _ = p.omega_0
        with pytest.raises(ValueError, match="coupling_sq_axes"):
            _ = p.coupling_sq
        # damping is uniform, so its scalar accessor still works
        assert p.gamma == 1e13

    def test_invalid_tuple_length_raises(self):
        p = LorentzPole(resonance_frequency=(1e15, 2e15), damping=1e13, delta_epsilon=2.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="3-tuple"):
            _ = p.omega_0_axes

    def test_model_is_isotropic(self):
        iso = DispersionModel(poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0),))
        aniso = DispersionModel(poles=(DrudePole(plasma_frequency=(2e15, 0.0, 0.0), damping=1e13),))
        assert iso.is_isotropic
        assert not aniso.is_isotropic

    def test_susceptibility_axes_matches_scalar_for_isotropic(self):
        m = DispersionModel(poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0),))
        for omega in (0.5e15, 1.5e15):
            chi_axes = m.susceptibility_axes(omega)
            chi = m.susceptibility(omega)
            assert chi_axes[0] == chi
            assert chi_axes[1] == chi
            assert chi_axes[2] == chi

    def test_susceptibility_raises_for_anisotropic_model(self):
        m = DispersionModel(poles=(DrudePole(plasma_frequency=(2e15, 0.0, 0.0), damping=1e13),))
        with pytest.raises(ValueError, match="susceptibility_axes"):
            m.susceptibility(1e15)
        with pytest.raises(ValueError, match="susceptibility_axes"):
            m.permittivity(1e15)

    def test_per_axis_susceptibility_matches_independent_models(self):
        # A per-axis model must equal three independent isotropic models per axis.
        w0 = (1e15, 2e15, 1.5e15)
        g = (1e13, 3e13, 2e13)
        de = (2.0, 0.5, 1.0)
        m = DispersionModel(poles=(LorentzPole(resonance_frequency=w0, damping=g, delta_epsilon=de),))
        for omega in (0.7e15, 1.8e15):
            chi_axes = m.susceptibility_axes(omega)
            for ax in range(3):
                ref = DispersionModel(
                    poles=(LorentzPole(resonance_frequency=w0[ax], damping=g[ax], delta_epsilon=de[ax]),)
                )
                assert chi_axes[ax] == pytest.approx(ref.susceptibility(omega), rel=1e-12)

    def test_zero_coupling_axis_contributes_zero(self):
        m = DispersionModel(poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=(2.0, 0.0, 0.0)),))
        chi_axes = m.susceptibility_axes(1.2e15)
        assert chi_axes[0] != 0.0
        assert chi_axes[1] == 0.0
        assert chi_axes[2] == 0.0

    def test_permittivity_axes_with_tuple_eps_inf(self):
        m = DispersionModel(poles=(DrudePole(plasma_frequency=(2e15, 0.0, 0.0), damping=1e13),))
        eps = m.permittivity_axes(1e15, eps_inf=(2.0, 3.0, 4.0))
        chi = m.susceptibility_axes(1e15)
        assert eps[0] == pytest.approx(2.0 + chi[0])
        assert eps[1] == pytest.approx(3.0)
        assert eps[2] == pytest.approx(4.0)


class TestPerAxisCoefficients:
    def test_per_axis_coefficient_shapes_and_closed_form(self):
        w0 = (1e15, 2e15, 3e15)
        g = (1e13, 2e13, 3e13)
        de = (1.0, 2.0, 3.0)
        p = LorentzPole(resonance_frequency=w0, damping=g, delta_epsilon=de)
        dt = 4e-18
        c1, c2, c3, c4, _ = compute_pole_coefficients_per_axis((p,), dt=dt)
        assert c1.shape == (1, 3)
        for ax in range(3):
            denom = 1.0 + 0.5 * g[ax] * dt
            assert c1[0, ax] == pytest.approx((2.0 - w0[ax] ** 2 * dt**2) / denom, rel=1e-12)
            assert c2[0, ax] == pytest.approx(-(1.0 - 0.5 * g[ax] * dt) / denom, rel=1e-12)
            assert c3[0, ax] == pytest.approx((de[ax] * w0[ax] ** 2 * dt**2) / denom, rel=1e-12)
            assert c4[0, ax] == 0.0

    def test_empty_poles_return_empty_arrays(self):
        c1, _c2, _c3, c4, _ = compute_pole_coefficients_per_axis((), dt=1e-17)
        assert c1.shape == (0, 3)
        assert c4.shape == (0, 3)

    def test_wrapper_matches_per_axis_for_isotropic(self):
        poles = (
            LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0),
            DrudePole(plasma_frequency=2e15, damping=5e13),
        )
        dt = 4e-18
        c1s, c2s, c3s, c4s, _ = compute_pole_coefficients(poles, dt=dt)
        c1a, c2a, c3a, c4a, _ = compute_pole_coefficients_per_axis(poles, dt=dt)
        assert c1s.shape == (2,)
        for scalar, axes in ((c1s, c1a), (c2s, c2a), (c3s, c3a), (c4s, c4a)):
            assert np.array_equal(scalar, axes[:, 0])
            assert np.array_equal(axes[:, 0], axes[:, 1])
            assert np.array_equal(axes[:, 0], axes[:, 2])

    def test_wrapper_raises_for_per_axis_pole(self):
        p = DrudePole(plasma_frequency=(2e15, 0.0, 0.0), damping=1e13)
        with pytest.raises(ValueError, match="per-axis"):
            compute_pole_coefficients((p,), dt=1e-17)

    def test_per_axis_large_gamma_dt_is_accepted(self):
        # A large gamma on the y axis alone is not a stability violation (only
        # omega_0 * dt >= 2 is); the coefficients must stay finite.
        p = LorentzPole(resonance_frequency=1e15, damping=(1e13, 3e17, 1e13), delta_epsilon=1.0)
        c1, c2, c3, _, _ = compute_pole_coefficients_per_axis((p,), dt=1e-17)
        assert np.all(np.isfinite(c1)) and np.all(np.isfinite(c2)) and np.all(np.isfinite(c3))

    def test_per_axis_omega0_stability_check_names_axis(self):
        # omega_0 * dt >= 2 only on the z axis must raise and identify the axis.
        p = LorentzPole(resonance_frequency=(1e15, 1e15, 3e17), damping=1e13, delta_epsilon=1.0)
        with pytest.raises(ValueError, match=r"omega_0 \* dt.*axis z"):
            compute_pole_coefficients_per_axis((p,), dt=1e-17)

    def test_per_axis_inert_axes_skip_bounds(self):
        # resonance only on x (delta_epsilon = 0 on y, z).
        # A large unused omega_0 AND gamma on the inert y/z axes must not raise.
        p = LorentzPole(
            resonance_frequency=(1e15, 3e17, 3e17),
            damping=(1e13, 3e17, 3e17),
            delta_epsilon=(2.0, 0.0, 0.0),
        )
        c1, c2, c3, _, _ = compute_pole_coefficients_per_axis((p,), dt=1e-17)
        # Inert axes carry zero coupling; the active x axis is fine (omega_0*dt<2).
        assert c3[0, 1] == 0.0 and c3[0, 2] == 0.0
        assert np.all(np.isfinite(c1)) and np.all(np.isfinite(c2))

    def test_eps_spectrum_averages_component_axis(self):
        # With a 3-component coefficient axis the spectrum helper must average
        # the per-axis susceptibilities (mirroring its eps_inf reduction).
        p = LorentzPole(resonance_frequency=(1e15, 2e15, 1.5e15), damping=1e13, delta_epsilon=(2.0, 1.0, 0.5))
        dt = 4e-18
        a1, a0, b1, c4, b0 = compute_pole_delta_coefficients_per_axis((p,), dt=dt)
        # shape (num_poles, 3, 1) with a single spatial cell
        a1a, a0a, b1a, c4a, b0a = (x[:, :, None] for x in (a1, a0, b1, c4, b0))
        eps_inf = 2.25
        inv_eps = np.full((1, 1), 1.0 / eps_inf)
        omegas = np.array([0.7e15, 1.9e15])
        eps = compute_eps_spectrum_from_coefficients(a1a, a0a, b1a, inv_eps, omegas, dt, c4=c4a, b0=b0a)
        m = DispersionModel(poles=(p,))
        for i, omega in enumerate(omegas):
            chi_axes = m.susceptibility_axes(float(omega))
            expected = eps_inf + sum(chi_axes) / 3.0
            # Discrete vs continuum: O((omega*dt)^2) truncation, see above.
            assert eps[i] == pytest.approx(expected, rel=1e-4)

    def test_eps_spectrum_tensor_eps_inf_inverted_properly(self):
        # inv_eps_inf stores the inverse tensor: with off-diagonal entries,
        # diag(eps) != 1/diag(eps^-1), so the helper must invert per cell.
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0, orientation=(1.0, 1.0, 0.0))
        dt = 4e-18
        a1, a0, b1, c4, b0 = compute_pole_delta_coefficients_tensor((p,), dt)
        a1a, a0a, b1a, c4a, b0a = (x[:, :, None] for x in (a1, a0, b1, c4, b0))
        eps_inf_mat = np.array([[2.5, 0.4, 0.0], [0.4, 2.0, 0.1], [0.0, 0.1, 3.0]])
        inv_eps = np.linalg.inv(eps_inf_mat).reshape(9, 1)
        omegas = np.array([0.7e15, 1.3e15])
        eps = compute_eps_spectrum_from_coefficients(a1a, a0a, b1a, inv_eps, omegas, dt, c4=c4a, b0=b0a)
        m = DispersionModel(poles=(p,))
        for i, omega in enumerate(omegas):
            chi = m.susceptibility_tensor(float(omega))
            expected = np.trace(eps_inf_mat) / 3.0 + np.trace(chi) / 3.0
            # Discrete vs continuum: O((omega*dt)^2) truncation, see above.
            assert eps[i] == pytest.approx(expected, rel=1e-4)


class TestAllowedDispersiveCoefficientsPerAxis:
    def test_three_component_output(self):
        mats = {
            "air": Material(permittivity=1.0),
            "hbn_like": Material(
                permittivity=(4.9, 4.9, 2.9),
                dispersion=DispersionModel(
                    poles=(
                        LorentzPole(
                            resonance_frequency=(2.6e14, 2.6e14, 1.5e14),
                            damping=(9e11, 9e11, 7e11),
                            delta_epsilon=(2.0, 2.0, 0.5),
                        ),
                    )
                ),
            ),
        }
        a1, _a0, b1, _c4, _ = compute_allowed_dispersive_coefficients(mats, dt=1e-17, max_num_poles=1, num_components=3)
        assert a1.shape == (2, 1, 3)
        # air row is all zero
        assert np.all(a1[0] == 0.0)
        # per-axis columns differ for the anisotropic material
        assert b1[1, 0, 0] != b1[1, 0, 2]

    def test_num_components_one_with_anisotropic_material_raises(self):
        mats = {
            "aniso": Material(
                permittivity=1.0,
                dispersion=DispersionModel(poles=(DrudePole(plasma_frequency=(2e15, 0.0, 0.0), damping=1e13),)),
            ),
        }
        with pytest.raises(ValueError, match="isotropic dispersion"):
            compute_allowed_dispersive_coefficients(mats, dt=1e-17, max_num_poles=1, num_components=1)

    def test_invalid_num_components_raises(self):
        with pytest.raises(ValueError, match="num_components"):
            compute_allowed_dispersive_coefficients({}, dt=1e-17, max_num_poles=0, num_components=2)


class TestOrientedPoles:
    """Oriented poles: 1D oscillators along a unit vector (off-diagonal coupling)."""

    def test_orientation_normalized_and_flags(self):
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0, orientation=(1.0, 1.0, 0.0))
        assert p.is_oriented and not p.is_isotropic
        assert p.orientation == pytest.approx((2**-0.5, 2**-0.5, 0.0))

    def test_orientation_with_per_axis_params_raises(self):
        with pytest.raises(ValueError, match="scalar"):
            LorentzPole(
                resonance_frequency=(1e15, 2e15, 3e15), damping=1e13, delta_epsilon=2.0, orientation=(1.0, 0.0, 0.0)
            )

    def test_zero_orientation_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            DrudePole(plasma_frequency=1e15, damping=1e13, orientation=(0.0, 0.0, 0.0))

    def test_non_finite_orientation_raises(self):
        # NaN would slip through a plain norm check (nan < eps is False)
        with pytest.raises(ValueError, match="finite"):
            DrudePole(plasma_frequency=1e15, damping=1e13, orientation=(float("nan"), 0.0, 0.0))
        with pytest.raises(ValueError, match="finite"):
            DrudePole(plasma_frequency=1e15, damping=1e13, orientation=(float("inf"), 1.0, 0.0))

    def test_huge_orientation_normalizes(self):
        # finite but overflow-prone components: the scale-first norm must
        # normalize these instead of rejecting them
        p = DrudePole(plasma_frequency=1e15, damping=1e13, orientation=(1e200, 1e200, 0.0))
        assert p.orientation == pytest.approx((2**-0.5, 2**-0.5, 0.0))
        p = DrudePole(plasma_frequency=1e15, damping=1e13, orientation=(1e-200, 0.0, 0.0))
        assert p.orientation == pytest.approx((1.0, 0.0, 0.0))

    def test_oriented_ccpr_with_edot_raises(self):
        with pytest.raises(NotImplementedError, match="dE/dt"):
            CCPRPole(pole=complex(-1e13, -2e15), residue=complex(1e14, 5e14), orientation=(1.0, 0.0, 0.0))
        # a purely imaginary residue has no dE/dt coupling and is allowed
        p = CCPRPole(pole=complex(-1e13, -2e15), residue=complex(0.0, 5e14), orientation=(1.0, 0.0, 0.0))
        assert p.is_oriented

    def test_tensor_coefficients_closed_form(self):
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0, orientation=(1.0, 1.0, 0.0))
        dt = 4e-18
        c1, _c2, c3, c4, _ = compute_pole_coefficients_tensor((p,), dt)
        assert c1.shape == (1, 3) and c3.shape == (1, 9)
        u = np.asarray(p.orientation)
        denom = 1.0 + 0.5 * p.gamma * dt
        expected = (p.coupling_sq * dt**2 / denom) * np.outer(u, u)
        mat = c3[0].reshape(3, 3)
        assert np.allclose(mat, expected)
        assert np.allclose(mat, mat.T)
        assert np.all(np.linalg.eigvalsh(mat) >= -1e-30)
        assert np.allclose(c4[0], 0.0)

    def test_tensor_coefficients_per_axis_pole_is_diagonal(self):
        pa = LorentzPole(resonance_frequency=(1e15, 2e15, 3e15), damping=1e13, delta_epsilon=(1.0, 2.0, 3.0))
        dt = 4e-18
        _, _, c3t, _, _ = compute_pole_coefficients_tensor((pa,), dt)
        _, _, c3a, _, _ = compute_pole_coefficients_per_axis((pa,), dt)
        assert np.allclose(c3t[0].reshape(3, 3), np.diag(c3a[0]))

    def test_tensor_coefficients_negative_coupling_raises(self):
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=-2.0, orientation=(1.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="passivity"):
            compute_pole_coefficients_tensor((p,), 4e-18)

    def test_per_axis_function_rejects_oriented(self):
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0, orientation=(0.0, 1.0, 0.0))
        with pytest.raises(ValueError, match="tensor"):
            compute_pole_coefficients_per_axis((p,), 4e-18)


class TestRotatedModel:
    def _model(self):
        return DispersionModel(
            poles=(LorentzPole(resonance_frequency=(1e15, 2e15, 3e15), damping=1e13, delta_epsilon=(1.0, 2.0, 0.5)),)
        )

    def test_rotation_identity(self):
        import math

        m = self._model()
        ang = math.pi / 6
        mr = m.rotated((0.0, 0.0, ang))
        assert mr.has_off_diagonal_coupling and len(mr.poles) == 3
        r_mat = np.array([[math.cos(ang), -math.sin(ang), 0.0], [math.sin(ang), math.cos(ang), 0.0], [0.0, 0.0, 1.0]])
        for omega in (0.7e15, 1.8e15):
            assert np.allclose(mr.susceptibility_tensor(omega), r_mat @ m.susceptibility_tensor(omega) @ r_mat.T)

    def test_euler_and_matrix_input_agree(self):
        import math

        m = self._model()
        ang = math.pi / 5
        by_euler = m.rotated((0.0, 0.0, ang))
        by_matrix = m.rotated(
            (
                (math.cos(ang), -math.sin(ang), 0.0),
                (math.sin(ang), math.cos(ang), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        assert np.allclose(by_euler.susceptibility_tensor(1.3e15), by_matrix.susceptibility_tensor(1.3e15))

    def test_signed_permutation_stays_per_axis(self):
        import math

        m = self._model()
        m90 = m.rotated((0.0, 0.0, math.pi / 2))
        assert not m90.has_off_diagonal_coupling
        chi90 = m90.susceptibility_axes(1.3e15)
        chi = m.susceptibility_axes(1.3e15)
        assert chi90 == (chi[1], chi[0], chi[2])

    def test_improper_rotation_raises(self):
        with pytest.raises(ValueError, match="proper rotation"):
            self._model().rotated(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)))

    def test_zero_strength_axes_dropped(self):
        m = DispersionModel(
            poles=(DrudePole(plasma_frequency=(2e15, 0.0, 0.0), damping=1e13),),
        )
        mr = m.rotated((0.0, 0.0, 0.3))
        assert len(mr.poles) == 1
        assert mr.poles[0].is_oriented

    def test_susceptibility_axes_raises_for_oriented(self):
        mr = self._model().rotated((0.0, 0.0, 0.3))
        with pytest.raises(ValueError, match="susceptibility_tensor"):
            mr.susceptibility_axes(1e15)

    def test_monoclinic_two_oscillators_vs_analytic(self):
        import math

        ang = math.pi / 6
        u1 = (1.0, 0.0, 0.0)
        u2 = (math.cos(ang), math.sin(ang), 0.0)
        p1 = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0, orientation=u1)
        p2 = LorentzPole(resonance_frequency=2e15, damping=2e13, delta_epsilon=0.5, orientation=u2)
        m = DispersionModel(poles=(p1, p2))
        omega = 1.4e15
        expected = np.zeros((3, 3), dtype=complex)
        for p, u in ((p1, u1), (p2, u2)):
            chi = p.coupling_sq / (p.omega_0**2 - omega**2 - 1j * p.gamma * omega)
            expected += chi * np.outer(np.asarray(u), np.asarray(u))
        assert np.allclose(m.susceptibility_tensor(omega), expected)
        # genuinely off-diagonal and symmetric
        assert abs(m.susceptibility_tensor(omega)[0, 1]) > 0


class TestMixedTierSampling:
    """Setup-time sampling helpers with 9-component couplings / permittivities."""

    def test_susceptibility_from_coefficients_9_coupling(self):
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0, orientation=(1.0, 1.0, 0.0))
        dt = 4e-18
        a1, a0, b1, c4, b0 = compute_pole_delta_coefficients_tensor((p,), dt)
        omega = 1.3e15
        chi = susceptibility_from_coefficients(
            a1=a1[:, :, None],
            a0=a0[:, :, None],
            b1=b1[:, :, None],
            omega=omega,
            dt=dt,
            c4=c4[:, :, None],
            b0=b0[:, :, None],
        )
        assert chi.shape == (9, 1)
        m = DispersionModel(poles=(p,))
        expected = m.susceptibility_tensor(omega).reshape(-1)
        # float32 roundtrip through the coefficient inversion (2 - c1*D cancels)
        assert np.allclose(np.asarray(chi[:, 0]), expected, rtol=1e-2, atol=1e-6)

    def test_effective_inv_permittivity_tensor_path_no_inf(self):
        # 9-component inv_eps with vacuum cells (zero off-diagonals) — the old
        # elementwise 1/inv_eps would produce inf there.
        import jax.numpy as jnp

        from fdtdx.dispersion import effective_inv_permittivity

        inv_eps = jnp.zeros((9, 2, 1, 1)).at[(0, 4, 8), :].set(1.0)
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0, orientation=(1.0, 1.0, 0.0))
        dt = 4e-18
        a1, a0, b1, _c4, b0 = compute_pole_delta_coefficients_tensor((p,), dt)
        # pole only in cell 0
        a1a = jnp.zeros((1, 3, 2, 1, 1)).at[:, :, 0].set(a1[0][None, :, None, None])
        a0a = jnp.zeros((1, 3, 2, 1, 1)).at[:, :, 0].set(a0[0][None, :, None, None])
        b1a = jnp.zeros((1, 9, 2, 1, 1)).at[:, :, 0].set(b1[0][None, :, None, None])
        b0a = jnp.zeros((1, 9, 2, 1, 1)).at[:, :, 0].set(b0[0][None, :, None, None])
        omega = 1.3e15
        result = effective_inv_permittivity(inv_eps, a1a, a0a, b1a, omega, dt, b0=b0a)
        assert result.shape == (9, 2, 1, 1)
        assert bool(jnp.all(jnp.isfinite(result)))
        # vacuum cell stays identity
        assert np.allclose(np.asarray(result[(0, 4, 8), 1]), 1.0, atol=1e-6)
        # dispersive cell matches inverse of Re(I + chi); float32 coefficient
        # roundtrip limits the agreement to ~1e-3 relative
        m = DispersionModel(poles=(p,))
        eps_mat = np.eye(3) + np.real(m.susceptibility_tensor(omega))
        expected = np.linalg.inv(eps_mat).reshape(-1)
        assert np.allclose(np.asarray(result[:, 0, 0, 0]), expected, rtol=1e-2, atol=1e-4)


class TestAllowedCoefficientsCoupling:
    def test_oriented_material_requires_9_coupling(self):
        mats = {
            "oriented": Material(
                permittivity=1.0,
                dispersion=DispersionModel(
                    poles=(
                        LorentzPole(
                            resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0, orientation=(1.0, 1.0, 0.0)
                        ),
                    )
                ),
            ),
        }
        with pytest.raises(ValueError, match="axis-aligned"):
            compute_allowed_dispersive_coefficients(mats, dt=1e-17, max_num_poles=1, num_components=3)
        a1, _, b1, _, _ = compute_allowed_dispersive_coefficients(
            mats, dt=1e-17, max_num_poles=1, num_components=3, coupling_components=9
        )
        assert a1.shape == (1, 1, 3)
        assert b1.shape == (1, 1, 9)
        assert b1[0, 0, 1] != 0.0  # off-diagonal weight present

    def test_diagonal_material_reduces_exactly(self):
        mats = {
            "per_axis": Material(
                permittivity=1.0,
                dispersion=DispersionModel(
                    poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=(1.0, 2.0, 3.0)),)
                ),
            ),
        }
        _, _, b1_full, _, _ = compute_allowed_dispersive_coefficients(
            mats, dt=1e-17, max_num_poles=1, num_components=3, coupling_components=9
        )
        _, _, b1_diag, _, _ = compute_allowed_dispersive_coefficients(
            mats, dt=1e-17, max_num_poles=1, num_components=3, coupling_components=3
        )
        assert np.allclose(b1_full[0, 0, (0, 4, 8)], b1_diag[0, 0])
        off_diag = [i for i in range(9) if i not in (0, 4, 8)]
        assert np.allclose(b1_full[0, 0, off_diag], 0.0)

    def test_per_axis_material_with_scalar_coupling_raises(self):
        # coupling_components=1 keeps only the xx entry, which would silently
        # drop the yy/zz couplings of a per-axis pole.
        mats = {
            "per_axis": Material(
                permittivity=1.0,
                dispersion=DispersionModel(
                    poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=(1.0, 2.0, 3.0)),)
                ),
            ),
        }
        with pytest.raises(ValueError, match="coupling_components=1"):
            compute_allowed_dispersive_coefficients(
                mats, dt=1e-17, max_num_poles=1, num_components=3, coupling_components=1
            )


class TestIntegratorSchemes:
    """Per-pole time-integrator selection (central / centered_edot / bilinear)."""

    # A gold-like critical point: complex pole and residue, so b = 2*Re(r) != 0.
    Q = complex(-2.0e13, -1.8e15)
    R = complex(3.0e14, -6.0e14)
    DT = 1e-17

    @staticmethod
    def _chi_exact(omega, w0, gamma, a, b):
        """Continuum susceptibility in the exp(-i omega t) convention."""
        return (a - 1j * omega * b) / (w0**2 - omega**2 - 1j * gamma * omega)

    @staticmethod
    def _chi_discrete(omega, dt, c1, c2, c3, c4, c5):
        """P-form transfer function chi_d(z) = (c4 z^2 + c3 z + c5)/(z^2 - c1 z - c2)."""
        z = np.exp(-1j * omega * dt)
        return (c4 * z * z + c3 * z + c5) / (z * z - c1 * z - c2)

    def _scalars(self, scheme, dt=None):
        p = CCPRPole(pole=self.Q, residue=self.R, integrator=scheme)
        vals = compute_pole_coefficients((p,), dt=dt or self.DT)
        return p, [float(v[0]) for v in vals]

    def test_default_integrator_is_centered_edot(self):
        assert LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=1.0).integrator == "centered_edot"
        assert DrudePole(plasma_frequency=1e16, damping=1e13).integrator == "centered_edot"
        assert CCPRPole(pole=self.Q, residue=self.R).integrator == "centered_edot"
        assert (
            CCPRPole.from_critical_point(amplitude=1.0, phase=0.3, resonance_frequency=4e15, damping=1e14).integrator
            == "centered_edot"
        )

    def test_unknown_integrator_raises(self):
        with pytest.raises(ValueError, match="Unknown dispersion integrator"):
            LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=1.0, integrator="trapezoid")

    def test_central_reproduces_legacy_closed_form(self):
        """Regression lock: 'central' must stay bit-identical to the pre-change scheme."""
        p, (c1, c2, c3, c4, c5) = self._scalars("central")
        w0, g, a, b, dt = p.omega_0, p.gamma, p.coupling_sq, p.coupling_edot, self.DT
        d = 1.0 + g * dt / 2.0
        assert c1 == (2.0 - w0**2 * dt**2) / d
        assert c2 == -(1.0 - g * dt / 2.0) / d
        assert c3 == (a * dt**2 - b * dt) / d
        assert c4 == b * dt / d
        assert c5 == 0.0

    def test_centered_edot_halves_c4_and_mirrors_it_into_c5(self):
        _, central = self._scalars("central")
        _, centered = self._scalars("centered_edot")
        assert centered[0] == central[0]  # c1 unchanged
        assert centered[1] == central[1]  # c2 unchanged
        assert centered[3] == pytest.approx(central[3] / 2.0, rel=1e-14)
        assert centered[4] == pytest.approx(-central[3] / 2.0, rel=1e-14)

    @pytest.mark.parametrize(
        "pole",
        [
            LorentzPole(resonance_frequency=3e15, damping=1e14, delta_epsilon=2.0),
            DrudePole(plasma_frequency=1.2e16, damping=8e13),
            CCPRPole(pole=complex(-1e13, -2e15), residue=complex(0.0, 5e14)),  # purely imaginary residue -> b = 0
        ],
    )
    def test_centered_edot_identical_to_central_when_b_is_zero(self, pole):
        """The new default must not perturb any Lorentz/Drude material, bit for bit."""
        assert pole.coupling_edot == 0.0
        cen = compute_pole_coefficients((pole.aset("integrator", "centered_edot"),), dt=self.DT)
        cnt = compute_pole_coefficients((pole.aset("integrator", "central"),), dt=self.DT)
        for a, b in zip(cen, cnt):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("scheme", ["central", "centered_edot", "bilinear"])
    def test_observer_form_reproduces_p_form_transfer_function(self, scheme):
        """to_observer_form must be an exact realization, not an approximation."""
        _, (c1, c2, c3, c4, c5) = self._scalars(scheme)
        _, _, beta1, c4o, beta0 = to_observer_form(*(np.array(v) for v in (c1, c2, c3, c4, c5)))
        omegas = np.linspace(0.05, 2.0, 11) / self.DT
        z = np.exp(-1j * omegas * self.DT)
        chi_obs = c4o + (beta1 * z + beta0) / (z * z - c1 * z - c2)
        chi_pform = self._chi_discrete(omegas, self.DT, c1, c2, c3, c4, c5)
        np.testing.assert_allclose(chi_obs, chi_pform, rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize(
        "scheme, expected_order",
        [("central", 1.0), ("centered_edot", 2.0), ("bilinear", 2.0)],
    )
    def test_convergence_order_in_chi(self, scheme, expected_order):
        """'central' is only first order once b != 0; the other two are second order."""
        omega = 2.2e15
        p = CCPRPole(pole=self.Q, residue=self.R)
        exact = self._chi_exact(omega, p.omega_0, p.gamma, p.coupling_sq, p.coupling_edot)
        errs = []
        for pts_per_period in (10, 40, 160):
            dt = 2.0 * np.pi / (omega * pts_per_period)
            _, coeffs = self._scalars(scheme, dt=dt)
            errs.append(abs(self._chi_discrete(omega, dt, *coeffs) - exact) / abs(exact))
        order = np.log(errs[0] / errs[-1]) / np.log(16.0)
        assert order == pytest.approx(expected_order, abs=0.25), f"{scheme}: order {order:.2f}, errors {errs}"

    def test_central_is_first_order_only_because_of_b(self):
        """With b = 0 the 'central' scheme recovers second order, pinning the cause."""
        omega = 2.2e15
        pole_no_b = CCPRPole(pole=self.Q, residue=complex(0.0, self.R.imag), integrator="central")
        exact = self._chi_exact(
            omega, pole_no_b.omega_0, pole_no_b.gamma, pole_no_b.coupling_sq, pole_no_b.coupling_edot
        )
        errs = []
        for pts in (10, 160):
            dt = 2.0 * np.pi / (omega * pts)
            coeffs = [float(v[0]) for v in compute_pole_coefficients((pole_no_b,), dt=dt)]
            errs.append(abs(self._chi_discrete(omega, dt, *coeffs) - exact) / abs(exact))
        order = np.log(errs[0] / errs[-1]) / np.log(16.0)
        assert order == pytest.approx(2.0, abs=0.25)

    @pytest.mark.parametrize("omega0_dt", [0.1, 1.99, 2.0, 10.0, 100.0])
    def test_bilinear_is_unconditionally_stable(self, omega0_dt):
        """Roots stay inside the unit circle for bilinear at every omega_0 * dt."""
        dt = self.DT
        pole = LorentzPole(
            resonance_frequency=omega0_dt / dt,
            damping=0.2 / dt,  # gamma * dt = 0.2, a clear damping margin
            delta_epsilon=1.0,
            integrator="bilinear",
        )
        c1, c2, _, _, _ = (float(v[0]) for v in compute_pole_coefficients((pole,), dt=dt))
        assert max(abs(np.roots([1.0, -c1, -c2]))) < 1.0

    @pytest.mark.parametrize("scheme", ["central", "centered_edot"])
    def test_central_schemes_keep_the_omega0_dt_bound(self, scheme):
        dt = self.DT
        pole = LorentzPole(resonance_frequency=2.0 / dt, damping=0.2 / dt, delta_epsilon=1.0, integrator=scheme)
        with pytest.raises(ValueError, match="omega_0 \\* dt"):
            compute_pole_coefficients((pole,), dt=dt)

    def test_bilinear_is_exact_at_dc(self):
        """chi_d(z=1) == chi(0) = a / omega_0^2 for the bilinear map."""
        p, (c1, c2, c3, c4, c5) = self._scalars("bilinear")
        chi_dc = (c4 + c3 + c5) / (1.0 - c1 - c2)
        assert chi_dc == pytest.approx(p.coupling_sq / p.omega_0**2, rel=1e-12)

    def test_bilinear_rejected_for_oriented_poles(self):
        with pytest.raises(NotImplementedError, match="bilinear"):
            LorentzPole(
                resonance_frequency=1e15,
                damping=1e13,
                delta_epsilon=1.0,
                orientation=(1.0, 1.0, 0.0),
                integrator="bilinear",
            )

    def test_with_integrator_switches_every_pole(self):
        model = DispersionModel(
            poles=(
                LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=1.0),
                CCPRPole(pole=self.Q, residue=self.R),
            )
        )
        switched = model.with_integrator("bilinear")
        assert [p.integrator for p in switched.poles] == ["bilinear", "bilinear"]
        # original untouched (pytrees are immutable)
        assert [p.integrator for p in model.poles] == ["centered_edot", "centered_edot"]

    def test_with_integrator_rejects_bilinear_for_oriented_model(self):
        model = DispersionModel(
            poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=1.0, orientation=(0.0, 1.0, 0.0)),)
        )
        with pytest.raises(NotImplementedError, match="bilinear"):
            model.with_integrator("bilinear")

    def test_with_integrator_rejects_unknown_scheme(self):
        model = DispersionModel(poles=(DrudePole(plasma_frequency=1e16, damping=1e13),))
        with pytest.raises(ValueError, match="Unknown dispersion integrator"):
            model.with_integrator("newmark")

    def test_rotated_preserves_integrator(self):
        model = DispersionModel(
            poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=(2.0, 1.0, 0.0)),)
        )
        rotated = model.rotated((0.0, 0.0, 0.3))
        assert rotated.num_poles > 0
        assert all(p.integrator == "centered_edot" for p in rotated.poles)


class TestDivisorScheme:
    """The implicit E-update divisor of a realistic metal CCPR fit.

    Silver fit of Prokopidis & Zografopoulos-style CCPR data (as used by
    ``scripts/nanoparticle_sim_forward.py``). For a physical fit the total
    1/omega tail must vanish, i.e. sum_p b_p = -sigma/eps0; the conductivity term
    then cancels the 'centered_edot' c4 and the divisor sits near 1 at every
    resolution, while 'central' puts twice that weight on E^{n+1} and marches
    negative.
    """

    EPS_INF = 3.07
    SIGMA = 1.49e7
    POLES = (
        (complex(-1.89e14, 0.0), complex(-1.00e18, 0.0)),
        (complex(-5.46e14, -6.37e15), complex(1.30e15, 1.54e15)),
        (complex(-5.68e14, -3.43e14), complex(1.61e17, 1.00e12)),
    )

    def _material(self, scheme):
        model = DispersionModel(poles=tuple(CCPRPole(pole=q, residue=r, integrator=scheme) for q, r in self.POLES))
        return Material(permittivity=self.EPS_INF, electric_conductivity=self.SIGMA, dispersion=model)

    @staticmethod
    def _dt(resolution_nm, courant_factor=0.8):
        import math

        from fdtdx import constants

        return (courant_factor / math.sqrt(3.0)) * resolution_nm * 1e-9 / constants.c

    def test_fit_is_physical_high_frequency_tail_cancels_conductivity(self):
        """sum_p b_p == -sigma/eps0 to within the fit residual — the reason the
        centered c4 and the trapezoidal conductivity term cancel."""
        from fdtdx.constants import eps0

        total_b = sum(2.0 * r.real for _, r in self.POLES)
        assert total_b == pytest.approx(-self.SIGMA / eps0, rel=0.01)

    @pytest.mark.parametrize(
        "resolution_nm, central_divisor",
        [(0.5, 0.7908), (1.0, 0.5816), (2.0, 0.1635), (4.0, -0.6717), (8.0, -2.3387)],
    )
    def test_central_divisor_collapses_with_resolution(self, resolution_nm, central_divisor):
        from fdtdx.materials import _min_dispersive_divisor

        div, _ = _min_dispersive_divisor(self._material("central"), self._dt(resolution_nm))
        assert div == pytest.approx(central_divisor, abs=5e-4)

    @pytest.mark.parametrize("resolution_nm", [0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    @pytest.mark.parametrize("scheme", ["centered_edot", "bilinear"])
    def test_second_order_schemes_keep_the_divisor_near_one(self, resolution_nm, scheme):
        from fdtdx.materials import _min_dispersive_divisor

        div, _ = _min_dispersive_divisor(self._material(scheme), self._dt(resolution_nm))
        assert 0.9 < div < 1.1

    def test_validator_rejects_central_and_accepts_centered(self):
        from fdtdx.materials import validate_dispersive_divisor_stability

        dt = self._dt(4.0)
        with pytest.raises(ValueError, match="non-positive"):
            validate_dispersive_divisor_stability({"Ag": self._material("central")}, dt=dt, courant_factor=0.8)
        # no raise, no warning
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_dispersive_divisor_stability({"Ag": self._material("centered_edot")}, dt=dt, courant_factor=0.8)

    def test_validator_message_suggests_switching_integrator(self):
        from fdtdx.materials import validate_dispersive_divisor_stability

        with pytest.raises(ValueError, match="centered_edot"):
            validate_dispersive_divisor_stability(
                {"Ag": self._material("central")}, dt=self._dt(4.0), courant_factor=0.8
            )

    def test_validator_covers_bilinear_lorentz_despite_zero_edot(self):
        """Under bilinear, c4 != 0 even at b = 0, so the 'skip Lorentz' shortcut
        would wrongly bypass the check. A safe fit must still pass cleanly."""
        import warnings

        from fdtdx.materials import validate_dispersive_divisor_stability

        model = DispersionModel(
            poles=(LorentzPole(resonance_frequency=3e15, damping=1e14, delta_epsilon=2.0),)
        ).with_integrator("bilinear")
        mat = Material(permittivity=2.25, dispersion=model)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_dispersive_divisor_stability({"lor": mat}, dt=self._dt(2.0), courant_factor=0.8)


class TestDeltaBasis:
    """The delta (zeta = z - 1) basis the FDTD loop stores and marches.

    See the :mod:`fdtdx.dispersion` module docstring: storing ``a1 = 2 - c1`` and
    ``a0 = 1 - c1 - c2`` instead of ``c1``/``c2`` is what makes float32
    dispersion usable, because for a real pole ``a0`` is ``O((omega_0 dt)^2)`` —
    orders of magnitude below the float32 resolution of ``c1 ~ 2`` itself.
    """

    # Gold's near-DC pole from the CCPR fit of Gedeon, Hassan & Cala Lesina,
    # ACS Photonics 2023, 10, 3875 (Table 1): purely real pole and residue, so
    # omega_0 = gamma / 2 and the ADE has a double root. Paired with the static
    # conductivity sigma = 1.21e7 S/m whose 1/omega tail its `b` cancels.
    AU_REAL_POLE = CCPRPole(pole=complex(-1.28e14, 0.0), residue=complex(-6.85e17, 0.0))
    AU_COMPLEX_POLES = (
        CCPRPole(pole=complex(-6.36e14, -3.89e15), residue=complex(2.06e15, 8.70e14)),
        CCPRPole(pole=complex(-2.96e15, -6.12e15), residue=complex(1.60e13, 1.47e16)),
    )
    # dt at 1 nm / courant_factor 0.8 on a uniform 3D grid.
    DT_1NM = 0.8 / np.sqrt(3.0) * 1e-9 / 299_792_458.0

    ALL_POLES = (
        LorentzPole(resonance_frequency=3e15, damping=1e14, delta_epsilon=2.0),
        DrudePole(plasma_frequency=1.2e16, damping=8e13),
        CCPRPole(pole=complex(-2.0e13, -1.8e15), residue=complex(3.0e14, -6.0e14)),
        AU_REAL_POLE,
    )

    @pytest.mark.parametrize("scheme", ["central", "centered_edot", "bilinear"])
    @pytest.mark.parametrize("pole", ALL_POLES)
    def test_closed_forms_match_generic_conversion(self, pole, scheme):
        """The cancellation-free closed forms agree with ``to_delta_form``.

        Only to the accuracy the generic conversion can deliver: it forms
        ``1 - c1 - c2`` from float64 O(1) numbers, so its ``a0`` carries an
        absolute error of ~1e-16 regardless of how small ``a0`` really is.
        """
        p = pole.aset("integrator", scheme)
        direct = compute_pole_delta_coefficients_per_axis((p,), dt=self.DT_1NM)
        via_pform = to_delta_form(*compute_pole_coefficients_per_axis((p,), dt=self.DT_1NM))
        for got, want in zip(direct, via_pform):
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-15)

    @pytest.mark.parametrize("scheme", ["central", "centered_edot", "bilinear"])
    def test_closed_form_a0_is_exact_where_the_conversion_is_not(self, scheme):
        """A Drude pole has omega_0 = 0, hence a0 == 0 *exactly* — the whole point.

        ``1 - c1 - c2`` cannot deliver that: it lands on float64 round-off.
        """
        p = DrudePole(plasma_frequency=1.2e16, damping=8e13, integrator=scheme)
        _, a0_direct, _, _, _ = compute_pole_delta_coefficients_per_axis((p,), dt=self.DT_1NM)
        _, a0_conv, _, _, _ = to_delta_form(*compute_pole_coefficients_per_axis((p,), dt=self.DT_1NM))
        np.testing.assert_array_equal(a0_direct, 0.0)
        assert np.any(a0_conv != 0.0)

    @pytest.mark.parametrize("scheme", ["central", "centered_edot", "bilinear"])
    @pytest.mark.parametrize("pole", ALL_POLES)
    def test_delta_form_reproduces_p_form_transfer_function(self, pole, scheme):
        """The basis change must be exact, not an approximation."""
        p = pole.aset("integrator", scheme)
        c1, c2, c3, c4, c5 = (v[0, 0] for v in compute_pole_coefficients_per_axis((p,), dt=self.DT_1NM))
        a1, a0, b1, c4d, b0 = (v[0, 0] for v in compute_pole_delta_coefficients_per_axis((p,), dt=self.DT_1NM))
        omegas = np.linspace(0.02, 1.5, 17) / self.DT_1NM
        z = np.exp(-1j * omegas * self.DT_1NM)
        chi_pform = (c4 * z * z + c3 * z + c5) / (z * z - c1 * z - c2)
        zeta = z - 1.0
        chi_delta = c4d + (b1 * zeta + b0) / (zeta * zeta + a1 * zeta + a0)
        np.testing.assert_allclose(chi_delta, chi_pform, rtol=1e-9, atol=0.0)

    def test_zero_padded_slots_stay_all_zero(self):
        """Padded pole slots must be inert, i.e. literally zero — not (a1, a0) = (2, 1).

        ``to_delta_form`` applied to all-zero P-form coefficients would give
        ``a1 = 2, a0 = 1``; the assembly path must zero-pad the delta
        coefficients instead so ``b1 = b0 = 0`` pins the state at zero.
        """
        mats = {
            "vacuum": Material(),
            "gold": Material(permittivity=2.31, dispersion=DispersionModel(poles=(self.AU_REAL_POLE,))),
        }
        arrays = compute_allowed_dispersive_coefficients(mats, dt=self.DT_1NM, max_num_poles=3, num_components=1)
        a1, a0, b1, _c4, _b0 = arrays
        # gold carries one pole, so slots 1 and 2 are padding in every material
        for arr in arrays:
            assert arr.shape == (len(mats), 3, 1)
            np.testing.assert_array_equal(arr[:, 1:], 0.0)
        # the non-dispersive material is all-zero even in slot 0
        non_disp_rows = [i for i in range(a1.shape[0]) if not np.any(b1[i])]
        assert len(non_disp_rows) == 1
        for arr in arrays:
            np.testing.assert_array_equal(arr[non_disp_rows[0]], 0.0)
        # ... and the dispersive one is not, so the test is not vacuously true
        disp_row = 1 - non_disp_rows[0]
        assert np.all(a1[disp_row, 0] > 0.0) and np.all(a0[disp_row, 0] > 0.0)

    def test_det_equals_minus_c2(self):
        """``1 - a1 + a0 == -c2`` pins the delta-basis denominator against the P-form."""
        for scheme in ("central", "centered_edot", "bilinear"):
            for pole in self.ALL_POLES:
                p = pole.aset("integrator", scheme)
                _, c2, _, _, _ = compute_pole_coefficients_per_axis((p,), dt=self.DT_1NM)
                a1, a0, _, _, _ = compute_pole_delta_coefficients_per_axis((p,), dt=self.DT_1NM)
                np.testing.assert_allclose(1.0 - a1 + a0, -c2, rtol=1e-12, atol=0.0)

    def test_a0_is_quadratically_small_for_a_real_pole(self):
        """The precision hazard itself: a0 sits far below the float32 ulp of c1."""
        a1, a0, _, _, _ = compute_pole_delta_coefficients_per_axis((self.AU_REAL_POLE,), dt=self.DT_1NM)
        # real pole: omega_0 = |q|, gamma = 2|q|, so with u = |q| dt and D = 1 + u
        #   a1 = (2u + u^2) / D,   a0 = u^2 / D
        u = abs(complex(self.AU_REAL_POLE.pole)) * self.DT_1NM
        assert a0[0, 0] == pytest.approx(u**2 / (1.0 + u), rel=1e-9)
        assert a1[0, 0] == pytest.approx((2.0 * u + u**2) / (1.0 + u), rel=1e-9)
        # this is the whole problem: a0 is ~3.9e-8 while float32 resolves c1 ~ 2
        # only to ~1.2e-7, so 1 - c1 - c2 in float32 would be pure noise
        assert a0[0, 0] < np.spacing(np.float32(2.0))

    def test_float32_storage_preserves_realized_eps_for_gold(self):
        """Regression: the realized eps'' of the Au fit must survive a float32 cast.

        Stored as ``(c1, c2, beta1, c4, beta0)`` this is wrong by tens of percent
        at 1 nm and gets *worse* under refinement. In the delta basis the float32
        cast must be a non-event.
        """
        eps0 = 8.8541878128e-12
        poles = (self.AU_REAL_POLE, *self.AU_COMPLEX_POLES)
        eps_inf, sigma = 2.31, 1.21e7
        model = DispersionModel(poles=poles)
        omega = 2.0 * np.pi * 299_792_458.0 / 560e-9

        exact = eps_inf + model.susceptibility(omega) + 1j * sigma / (omega * eps0)
        assert exact.imag == pytest.approx(1.2838, rel=1e-3)  # sanity: eps'' ~ 1.28

        for dt in (self.DT_1NM, self.DT_1NM / 2.0, self.DT_1NM / 5.0):
            coeffs = compute_pole_delta_coefficients_per_axis(poles, dt=dt)
            a1, a0, b1, c4, b0 = (np.asarray(v[:, :1], dtype=np.float32) for v in coeffs)
            chi = susceptibility_from_coefficients(a1, a0, b1, omega, dt, c4=c4, b0=b0)
            realized = eps_inf + complex(np.asarray(chi)[0]) + 1j * sigma / (omega * eps0)
            # 1% covers the scheme's own O((omega dt)^2) truncation at these dt;
            # the float32 storage error is ~1e-3 of eps'' or better.
            assert realized.imag == pytest.approx(exact.imag, rel=1e-2), f"dt={dt:g}"
            assert realized.real == pytest.approx(exact.real, rel=1e-3), f"dt={dt:g}"

    @pytest.mark.parametrize(
        "pole",
        [
            LorentzPole(resonance_frequency=3e15, damping=1e14, delta_epsilon=2.0),
            DrudePole(plasma_frequency=1.2e16, damping=8e13),
            CCPRPole(pole=complex(-1e13, -2e15), residue=complex(0.0, 5e14)),  # b = 0
        ],
    )
    @pytest.mark.parametrize("scheme", ["central", "centered_edot"])
    def test_b0_equals_b1_exactly_when_no_pole_is_implicit(self, pole, scheme):
        """``update_E`` substitutes ``b1`` for ``b0`` when the ``b0`` array is not
        allocated. That is only sound if the two coincide *bit for bit* whenever
        every pole has ``c4 = c5 = 0``, which is exactly the allocation condition.
        """
        p = pole.aset("integrator", scheme)
        _, _, b1, c4, b0 = compute_pole_delta_coefficients_per_axis((p,), dt=self.DT_1NM)
        np.testing.assert_array_equal(c4, 0.0)
        np.testing.assert_array_equal(b0, b1)

    def test_bilinear_always_allocates_b0(self):
        """Under 'bilinear' even a b = 0 pole has c4 != 0, so b0 is always stored
        and the b0-is-b1 substitution above is never reached."""
        p = LorentzPole(resonance_frequency=3e15, damping=1e14, delta_epsilon=2.0, integrator="bilinear")
        _, _, b1, c4, b0 = compute_pole_delta_coefficients_per_axis((p,), dt=self.DT_1NM)
        assert np.all(c4 != 0.0)
        assert np.any(b0 != b1)
