"""Unit tests for the dispersion module (Lorentz / Drude / ADE coefficients)."""

import numpy as np
import pytest

from fdtdx.dispersion import (
    DispersionModel,
    DrudePole,
    LorentzPole,
    Pole,
    compute_eps_spectrum_from_coefficients,
    compute_pole_coefficients,
    compute_pole_coefficients_per_axis,
    compute_pole_coefficients_tensor,
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
        c1, c2, c3 = compute_pole_coefficients((), dt=1e-17)
        assert c1.shape == (0,)
        assert c2.shape == (0,)
        assert c3.shape == (0,)

    def test_lorentz_coefficients_closed_form(self):
        p = LorentzPole(resonance_frequency=2e15, damping=3e13, delta_epsilon=1.5)
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
        p = DrudePole(plasma_frequency=1e16, damping=1e14)
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
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0)
        _, c2, _ = compute_pole_coefficients((p,), dt=1e-17)
        assert abs(c2[0] + 1.0) < 2e-4

    def test_multiple_poles(self):
        poles = (
            LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0),
            DrudePole(plasma_frequency=5e15, damping=1e14),
        )
        c1, _, c3 = compute_pole_coefficients(poles, dt=5e-18)
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
        c1, c2, c3 = compute_pole_coefficients((p,), dt=dt)
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
        c1, _, _ = compute_pole_coefficients((p,), dt=1.9 / p.omega_0)
        assert np.isfinite(c1[0])

    def test_zero_coupling_pole_skips_bounds(self):
        # A Lorentz pole with delta_epsilon = 0 contributes nothing (coupling_sq =
        # 0, so c3 = 0 and the polarization stays zero); its unused omega_0 / gamma
        # must not trip the stability bounds even when both products exceed 2.
        p = LorentzPole(resonance_frequency=3e17, damping=3e17, delta_epsilon=0.0)
        c1, c2, c3 = compute_pole_coefficients((p,), dt=1e-17)
        assert c3[0] == 0.0
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
        c1, c2, c3 = compute_allowed_dispersive_coefficients(mats, dt=1e-17, max_num_poles=0, num_components=1)
        assert c1.shape == (2, 0, 1)
        assert c2.shape == (2, 0, 1)
        assert c3.shape == (2, 0, 1)

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
        c1, c2, c3 = compute_allowed_dispersive_coefficients(mats, dt=1e-17, max_num_poles=2, num_components=1)
        assert c1.shape == (3, 2, 1)
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


class TestPerAxisPoles:
    """Per-axis (diagonally anisotropic) pole parameters."""

    def test_scalar_pole_is_isotropic_and_axes_uniform(self):
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0)
        assert p.is_isotropic
        assert p.omega_0_axes == (1e15, 1e15, 1e15)
        assert p.gamma_axes == (1e13, 1e13, 1e13)
        assert p.coupling_sq_axes == pytest.approx((2e30, 2e30, 2e30))

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
        c1, c2, c3 = compute_pole_coefficients_per_axis((p,), dt=dt)
        assert c1.shape == (1, 3)
        for ax in range(3):
            denom = 1.0 + 0.5 * g[ax] * dt
            assert c1[0, ax] == pytest.approx((2.0 - w0[ax] ** 2 * dt**2) / denom, rel=1e-12)
            assert c2[0, ax] == pytest.approx(-(1.0 - 0.5 * g[ax] * dt) / denom, rel=1e-12)
            assert c3[0, ax] == pytest.approx((de[ax] * w0[ax] ** 2 * dt**2) / denom, rel=1e-12)

    def test_empty_poles_return_empty_arrays(self):
        c1, _c2, c3 = compute_pole_coefficients_per_axis((), dt=1e-17)
        assert c1.shape == (0, 3)
        assert c3.shape == (0, 3)

    def test_wrapper_matches_per_axis_for_isotropic(self):
        poles = (
            LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0),
            DrudePole(plasma_frequency=2e15, damping=5e13),
        )
        dt = 4e-18
        c1s, c2s, c3s = compute_pole_coefficients(poles, dt=dt)
        c1a, c2a, c3a = compute_pole_coefficients_per_axis(poles, dt=dt)
        assert c1s.shape == (2,)
        for scalar, axes in ((c1s, c1a), (c2s, c2a), (c3s, c3a)):
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
        c1, c2, c3 = compute_pole_coefficients_per_axis((p,), dt=1e-17)
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
        c1, c2, c3 = compute_pole_coefficients_per_axis((p,), dt=1e-17)
        # Inert axes carry zero coupling; the active x axis is fine (omega_0*dt<2).
        assert c3[0, 1] == 0.0 and c3[0, 2] == 0.0
        assert np.all(np.isfinite(c1)) and np.all(np.isfinite(c2))

    def test_eps_spectrum_averages_component_axis(self):
        # With a 3-component coefficient axis the spectrum helper must average
        # the per-axis susceptibilities (mirroring its eps_inf reduction).
        p = LorentzPole(resonance_frequency=(1e15, 2e15, 1.5e15), damping=1e13, delta_epsilon=(2.0, 1.0, 0.5))
        dt = 4e-18
        c1, c2, c3 = compute_pole_coefficients_per_axis((p,), dt=dt)
        # shape (num_poles, 3, 1) with a single spatial cell
        c1a, c2a, c3a = (x[:, :, None] for x in (c1, c2, c3))
        eps_inf = 2.25
        inv_eps = np.full((1, 1), 1.0 / eps_inf)
        omegas = np.array([0.7e15, 1.9e15])
        eps = compute_eps_spectrum_from_coefficients(c1a, c2a, c3a, inv_eps, omegas, dt)
        m = DispersionModel(poles=(p,))
        for i, omega in enumerate(omegas):
            chi_axes = m.susceptibility_axes(float(omega))
            expected = eps_inf + sum(chi_axes) / 3.0
            assert eps[i] == pytest.approx(expected, rel=1e-9)

    def test_eps_spectrum_tensor_eps_inf_inverted_properly(self):
        # inv_eps_inf stores the inverse tensor: with off-diagonal entries,
        # diag(eps) != 1/diag(eps^-1), so the helper must invert per cell.
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0, orientation=(1.0, 1.0, 0.0))
        dt = 4e-18
        c1, c2, c3 = compute_pole_coefficients_tensor((p,), dt)
        c1a, c2a, c3a = (x[:, :, None] for x in (c1, c2, c3))
        eps_inf_mat = np.array([[2.5, 0.4, 0.0], [0.4, 2.0, 0.1], [0.0, 0.1, 3.0]])
        inv_eps = np.linalg.inv(eps_inf_mat).reshape(9, 1)
        omegas = np.array([0.7e15, 1.3e15])
        eps = compute_eps_spectrum_from_coefficients(c1a, c2a, c3a, inv_eps, omegas, dt)
        m = DispersionModel(poles=(p,))
        for i, omega in enumerate(omegas):
            chi = m.susceptibility_tensor(float(omega))
            expected = np.trace(eps_inf_mat) / 3.0 + np.trace(chi) / 3.0
            assert eps[i] == pytest.approx(expected, rel=1e-6)


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
        c1, _c2, c3 = compute_allowed_dispersive_coefficients(mats, dt=1e-17, max_num_poles=1, num_components=3)
        assert c1.shape == (2, 1, 3)
        # air row is all zero
        assert np.all(c1[0] == 0.0)
        # per-axis columns differ for the anisotropic material
        assert c3[1, 0, 0] != c3[1, 0, 2]

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

    def test_tensor_coefficients_closed_form(self):
        p = LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0, orientation=(1.0, 1.0, 0.0))
        dt = 4e-18
        c1, _c2, c3 = compute_pole_coefficients_tensor((p,), dt)
        assert c1.shape == (1, 3) and c3.shape == (1, 9)
        u = np.asarray(p.orientation)
        denom = 1.0 + 0.5 * p.gamma * dt
        expected = (p.coupling_sq * dt**2 / denom) * np.outer(u, u)
        mat = c3[0].reshape(3, 3)
        assert np.allclose(mat, expected)
        assert np.allclose(mat, mat.T)
        assert np.all(np.linalg.eigvalsh(mat) >= -1e-30)

    def test_tensor_coefficients_per_axis_pole_is_diagonal(self):
        pa = LorentzPole(resonance_frequency=(1e15, 2e15, 3e15), damping=1e13, delta_epsilon=(1.0, 2.0, 3.0))
        dt = 4e-18
        _, _, c3t = compute_pole_coefficients_tensor((pa,), dt)
        _, _, c3a = compute_pole_coefficients_per_axis((pa,), dt)
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
        c1, c2, c3 = compute_pole_coefficients_tensor((p,), dt)
        omega = 1.3e15
        chi = susceptibility_from_coefficients(
            c1=c1[:, :, None], c2=c2[:, :, None], c3=c3[:, :, None], omega=omega, dt=dt
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
        c1, c2, c3 = compute_pole_coefficients_tensor((p,), dt)
        # pole only in cell 0
        c1a = jnp.zeros((1, 3, 2, 1, 1)).at[:, :, 0].set(c1[0][None, :, None, None])
        c2a = jnp.zeros((1, 3, 2, 1, 1)).at[:, :, 0].set(c2[0][None, :, None, None])
        c3a = jnp.zeros((1, 9, 2, 1, 1)).at[:, :, 0].set(c3[0][None, :, None, None])
        omega = 1.3e15
        result = effective_inv_permittivity(inv_eps, c1a, c2a, c3a, omega, dt)
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
        c1, _, c3 = compute_allowed_dispersive_coefficients(
            mats, dt=1e-17, max_num_poles=1, num_components=3, coupling_components=9
        )
        assert c1.shape == (1, 1, 3)
        assert c3.shape == (1, 1, 9)
        assert c3[0, 0, 1] != 0.0  # off-diagonal weight present

    def test_diagonal_material_reduces_exactly(self):
        mats = {
            "per_axis": Material(
                permittivity=1.0,
                dispersion=DispersionModel(
                    poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=(1.0, 2.0, 3.0)),)
                ),
            ),
        }
        _, _, c3_full = compute_allowed_dispersive_coefficients(
            mats, dt=1e-17, max_num_poles=1, num_components=3, coupling_components=9
        )
        _, _, c3_diag = compute_allowed_dispersive_coefficients(
            mats, dt=1e-17, max_num_poles=1, num_components=3, coupling_components=3
        )
        assert np.allclose(c3_full[0, 0, (0, 4, 8)], c3_diag[0, 0])
        off_diag = [i for i in range(9) if i not in (0, 4, 8)]
        assert np.allclose(c3_full[0, 0, off_diag], 0.0)

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
