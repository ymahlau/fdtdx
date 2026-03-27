"""Unit tests for fdtdx.fdtd.update"""

from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.constants import eta0
from fdtdx.fdtd.container import ArrayContainer
from fdtdx.fdtd.update import (
    add_interfaces,
    collect_interfaces,
    get_periodic_axes,
    update_detector_states,
    update_E,
    update_E_reverse,
    update_H,
    update_H_reverse,
)
from fdtdx.objects.boundaries.periodic import PeriodicBoundary

# ─── Shared helpers ───────────────────────────────────────────────────────────

NX, NY, NZ = 4, 4, 4
FIELD_SHAPE = (3, NX, NY, NZ)
PSI_SHAPE = (6, NX, NY, NZ)
CURL_ZERO = jnp.zeros(FIELD_SHAPE)
PSI_ZERO = jnp.zeros(PSI_SHAPE)


def _make_arrays(
    E=None,
    H=None,
    inv_permittivities=None,
    inv_permeabilities=None,
    electric_conductivity=None,
    magnetic_conductivity=None,
    detector_states=None,
    recording_state=None,
):
    """Build a minimal real ArrayContainer for update tests."""
    return ArrayContainer(
        E=E if E is not None else jnp.ones(FIELD_SHAPE),
        H=H if H is not None else jnp.zeros(FIELD_SHAPE),
        psi_E=jnp.zeros(PSI_SHAPE),
        psi_H=jnp.zeros(PSI_SHAPE),
        alpha=jnp.zeros(FIELD_SHAPE),
        kappa=jnp.ones(FIELD_SHAPE),
        sigma=jnp.zeros(FIELD_SHAPE),
        inv_permittivities=inv_permittivities if inv_permittivities is not None else jnp.ones(FIELD_SHAPE),
        inv_permeabilities=inv_permeabilities if inv_permeabilities is not None else jnp.ones(FIELD_SHAPE),
        detector_states=detector_states if detector_states is not None else {},
        recording_state=recording_state,
        electric_conductivity=electric_conductivity,
        magnetic_conductivity=magnetic_conductivity,
    )


def _make_objects(sources=None):
    """Build a mock ObjectContainer with no periodic boundaries."""
    obj = Mock()
    obj.boundary_objects = []
    obj.sources = sources or []
    return obj


def _make_config(c=0.5):
    cfg = Mock()
    cfg.courant_number = c
    return cfg


def _diag_anisotropic_tensor(shape):
    """Return a (9, Nx, Ny, Nz) diagonal (identity) anisotropic tensor."""
    Nx, Ny, Nz = shape
    t = jnp.zeros((9, Nx, Ny, Nz))
    # Diagonal elements of 3x3 are at flat indices 0 ([0,0]), 4 ([1,1]), 8 ([2,2])
    t = t.at[0].set(1.0)
    t = t.at[4].set(1.0)
    t = t.at[8].set(1.0)
    return t


# ─── TestGetPeriodicAxes ──────────────────────────────────────────────────────


class TestGetPeriodicAxes:
    def test_no_periodic_boundaries(self):
        obj = Mock()
        obj.boundary_objects = []
        assert get_periodic_axes(obj) == (False, False, False)

    def test_x_axis_periodic(self):
        b = Mock(spec=PeriodicBoundary)
        b.axis = 0
        obj = Mock()
        obj.boundary_objects = [b]
        assert get_periodic_axes(obj) == (True, False, False)

    def test_y_axis_periodic(self):
        b = Mock(spec=PeriodicBoundary)
        b.axis = 1
        obj = Mock()
        obj.boundary_objects = [b]
        assert get_periodic_axes(obj) == (False, True, False)

    def test_z_axis_periodic(self):
        b = Mock(spec=PeriodicBoundary)
        b.axis = 2
        obj = Mock()
        obj.boundary_objects = [b]
        assert get_periodic_axes(obj) == (False, False, True)

    def test_multiple_periodic_boundaries(self):
        b1 = Mock(spec=PeriodicBoundary)
        b1.axis = 0
        b2 = Mock(spec=PeriodicBoundary)
        b2.axis = 2
        obj = Mock()
        obj.boundary_objects = [b1, b2]
        assert get_periodic_axes(obj) == (True, False, True)

    def test_all_axes_periodic(self):
        boundaries = [Mock(spec=PeriodicBoundary) for _ in range(3)]
        for i, b in enumerate(boundaries):
            b.axis = i
        obj = Mock()
        obj.boundary_objects = boundaries
        assert get_periodic_axes(obj) == (True, True, True)

    def test_non_periodic_boundary_ignored(self):
        periodic = Mock(spec=PeriodicBoundary)
        periodic.axis = 1
        other = Mock()  # not a PeriodicBoundary
        obj = Mock()
        obj.boundary_objects = [periodic, other]
        assert get_periodic_axes(obj) == (False, True, False)


# ─── TestUpdateE ──────────────────────────────────────────────────────────────


class TestUpdateE:
    """Tests for the forward electric field update."""

    def _call(self, arrays, objects=None, config=None, curl=None, psi=None, simulate_boundaries=False):
        curl_val = curl if curl is not None else CURL_ZERO
        psi_val = psi if psi is not None else PSI_ZERO
        with patch("fdtdx.fdtd.update.curl_H", return_value=(curl_val, psi_val)):
            return update_E(
                time_step=jnp.array(0),
                arrays=arrays,
                objects=objects or _make_objects(),
                config=config or _make_config(),
                simulate_boundaries=simulate_boundaries,
            )

    def test_isotropic_zero_curl_unchanged(self):
        """With zero curl and no conductivity, E is unchanged."""
        E_init = jnp.ones(FIELD_SHAPE) * 3.0
        result = self._call(_make_arrays(E=E_init), curl=CURL_ZERO)
        assert jnp.allclose(result.E, E_init)

    def test_isotropic_no_conductivity_formula(self):
        """E^(n+1) = E^(n) + c * curl(H) * inv_eps."""
        c = 0.5
        E_init = jnp.ones(FIELD_SHAPE)
        curl = jnp.ones(FIELD_SHAPE) * 2.0
        inv_eps = jnp.ones(FIELD_SHAPE)
        result = self._call(_make_arrays(E=E_init, inv_permittivities=inv_eps), config=_make_config(c=c), curl=curl)
        expected = E_init + c * curl * inv_eps  # 1.0 + 0.5 * 2.0 * 1.0 = 2.0
        assert jnp.allclose(result.E, expected)

    def test_isotropic_with_conductivity_formula(self):
        """Lossy material forward update (Schneider 3.12): sigma_E != 0."""
        c = 0.5
        sigma_E = jnp.ones(FIELD_SHAPE) * 1e-4
        inv_eps = jnp.ones(FIELD_SHAPE)
        E_init = jnp.ones(FIELD_SHAPE) * 2.0
        arrays = _make_arrays(E=E_init, inv_permittivities=inv_eps, electric_conductivity=sigma_E)
        result = self._call(arrays, config=_make_config(c=c), curl=CURL_ZERO)
        half = c * sigma_E * eta0 * inv_eps / 2
        expected = (1 - half) * E_init / (1 + half)
        assert jnp.allclose(result.E, expected, rtol=1e-5)

    def test_psi_E_updated_from_curl_H(self):
        """psi_E in the returned container matches the psi returned by curl_H."""
        new_psi = jnp.ones(PSI_SHAPE) * 9.0
        result = self._call(_make_arrays(), psi=new_psi)
        assert jnp.allclose(result.psi_E, new_psi)

    def test_simulate_boundaries_flag_forwarded(self):
        """simulate_boundaries=True is passed as 7th positional arg to curl_H."""
        with patch("fdtdx.fdtd.update.curl_H", return_value=(CURL_ZERO, PSI_ZERO)) as mock_curl:
            update_E(
                time_step=jnp.array(0),
                arrays=_make_arrays(),
                objects=_make_objects(),
                config=_make_config(),
                simulate_boundaries=True,
            )
        # simulate_boundaries is positional arg index 6
        assert mock_curl.call_args[0][6] is True

    def test_anisotropic_full_tensor_correct_shape_and_finite(self):
        """inv_eps.shape[0]==9 triggers the anisotropic path; output is (3,N,N,N) and finite."""
        inv_eps = _diag_anisotropic_tensor((NX, NY, NZ))
        result = self._call(_make_arrays(inv_permittivities=inv_eps))
        assert result.E.shape == FIELD_SHAPE
        assert jnp.all(jnp.isfinite(result.E))

    def test_no_sources_skips_lax_cond(self):
        """With no sources, jax.lax.cond is never invoked."""
        with (
            patch("fdtdx.fdtd.update.curl_H", return_value=(CURL_ZERO, PSI_ZERO)),
            patch("fdtdx.fdtd.update.jax.lax.cond") as mock_cond,
        ):
            update_E(
                time_step=jnp.array(0),
                arrays=_make_arrays(),
                objects=_make_objects(sources=[]),
                config=_make_config(),
                simulate_boundaries=False,
            )
        mock_cond.assert_not_called()

    def test_with_source_calls_lax_cond_and_update_E(self):
        """With one source, jax.lax.cond and source.update_E are each called once."""
        source = Mock()
        source.is_on_at_time_step.return_value = jnp.array(True)
        source.adjust_time_step_by_on_off.return_value = jnp.array(0)
        source.update_E.return_value = jnp.ones(FIELD_SHAPE)

        with (
            patch("fdtdx.fdtd.update.curl_H", return_value=(CURL_ZERO, PSI_ZERO)),
            patch("fdtdx.fdtd.update.jax.lax.cond", side_effect=lambda cond, t, f: t()) as mock_cond,
        ):
            update_E(
                time_step=jnp.array(0),
                arrays=_make_arrays(),
                objects=_make_objects(sources=[source]),
                config=_make_config(),
                simulate_boundaries=False,
            )
        mock_cond.assert_called_once()
        source.is_on_at_time_step.assert_called_once()
        source.update_E.assert_called_once()

    def test_source_inverse_false_passed(self):
        """update_E passes inverse=False to source.update_E."""
        source = Mock()
        source.is_on_at_time_step.return_value = jnp.array(True)
        source.adjust_time_step_by_on_off.return_value = jnp.array(0)
        source.update_E.return_value = jnp.ones(FIELD_SHAPE)

        with (
            patch("fdtdx.fdtd.update.curl_H", return_value=(CURL_ZERO, PSI_ZERO)),
            patch("fdtdx.fdtd.update.jax.lax.cond", side_effect=lambda cond, t, f: t()),
        ):
            update_E(
                time_step=jnp.array(0),
                arrays=_make_arrays(),
                objects=_make_objects(sources=[source]),
                config=_make_config(),
                simulate_boundaries=False,
            )
        assert source.update_E.call_args[1]["inverse"] is False


# ─── TestUpdateEReverse ───────────────────────────────────────────────────────


class TestUpdateEReverse:
    """Tests for the reverse electric field update."""

    def _call(self, arrays, objects=None, config=None, curl=None):
        with patch("fdtdx.fdtd.update.curl_H", return_value=(curl if curl is not None else CURL_ZERO, PSI_ZERO)):
            return update_E_reverse(
                time_step=jnp.array(0),
                arrays=arrays,
                objects=objects or _make_objects(),
                config=config or _make_config(),
            )

    def test_isotropic_zero_curl_unchanged(self):
        """With zero curl and no conductivity, E is unchanged."""
        E_init = jnp.ones(FIELD_SHAPE) * 3.0
        result = self._call(_make_arrays(E=E_init), curl=CURL_ZERO)
        assert jnp.allclose(result.E, E_init)

    def test_isotropic_no_conductivity_formula(self):
        """Reverse update: E^(n) = E^(n+1) - c * curl * inv_eps."""
        c = 0.5
        E_fwd = jnp.ones(FIELD_SHAPE) * 2.0
        curl = jnp.ones(FIELD_SHAPE)
        inv_eps = jnp.ones(FIELD_SHAPE)
        result = self._call(_make_arrays(E=E_fwd, inv_permittivities=inv_eps), config=_make_config(c=c), curl=curl)
        expected = E_fwd - c * curl * inv_eps  # 2.0 - 0.5 = 1.5
        assert jnp.allclose(result.E, expected)

    def test_isotropic_with_conductivity_reverses_forward(self):
        """Reverse lossy update recovers the original E field (zero curl)."""
        c = 0.5
        sigma_E = jnp.ones(FIELD_SHAPE) * 1e-4
        inv_eps = jnp.ones(FIELD_SHAPE)
        E_init = jnp.ones(FIELD_SHAPE) * 2.0
        half = c * sigma_E * eta0 * inv_eps / 2
        # Simulate forward: E_fwd = (1 - half) * E_init / (1 + half)
        E_fwd = (1 - half) * E_init / (1 + half)
        arrays = _make_arrays(E=E_fwd, inv_permittivities=inv_eps, electric_conductivity=sigma_E)
        result = self._call(arrays, config=_make_config(c=c), curl=CURL_ZERO)
        assert jnp.allclose(result.E, E_init, rtol=1e-5)

    def test_anisotropic_full_tensor_correct_shape_and_finite(self):
        """inv_eps.shape[0]==9 triggers anisotropic reverse path; output is finite."""
        inv_eps = _diag_anisotropic_tensor((NX, NY, NZ))
        result = self._call(_make_arrays(inv_permittivities=inv_eps))
        assert result.E.shape == FIELD_SHAPE
        assert jnp.all(jnp.isfinite(result.E))

    def test_with_source_calls_lax_cond(self):
        """Source reverse update is invoked via jax.lax.cond with inverse=True."""
        source = Mock()
        source.is_on_at_time_step.return_value = jnp.array(True)
        source.adjust_time_step_by_on_off.return_value = jnp.array(0)
        source.update_E.return_value = jnp.ones(FIELD_SHAPE)

        with (
            patch("fdtdx.fdtd.update.curl_H", return_value=(CURL_ZERO, PSI_ZERO)),
            patch("fdtdx.fdtd.update.jax.lax.cond", side_effect=lambda cond, t, f: t()) as mock_cond,
        ):
            update_E_reverse(
                time_step=jnp.array(0),
                arrays=_make_arrays(),
                objects=_make_objects(sources=[source]),
                config=_make_config(),
            )
        mock_cond.assert_called_once()
        source.update_E.assert_called_once()
        assert source.update_E.call_args[1]["inverse"] is True


# ─── TestUpdateH ──────────────────────────────────────────────────────────────


class TestUpdateH:
    """Tests for the forward magnetic field update."""

    def _call(self, arrays, objects=None, config=None, curl=None, psi=None, simulate_boundaries=False):
        curl_val = curl if curl is not None else CURL_ZERO
        psi_val = psi if psi is not None else PSI_ZERO
        with patch("fdtdx.fdtd.update.curl_E", return_value=(curl_val, psi_val)):
            return update_H(
                time_step=jnp.array(0),
                arrays=arrays,
                objects=objects or _make_objects(),
                config=config or _make_config(),
                simulate_boundaries=simulate_boundaries,
            )

    def test_isotropic_zero_curl_unchanged(self):
        """With zero curl and no conductivity, H is unchanged."""
        H_init = jnp.ones(FIELD_SHAPE) * 3.0
        result = self._call(_make_arrays(H=H_init), curl=CURL_ZERO)
        assert jnp.allclose(result.H, H_init)

    def test_isotropic_no_conductivity_formula(self):
        """H^(n+1/2) = H^(n-1/2) - c * curl(E) * inv_mu."""
        c = 0.5
        H_init = jnp.ones(FIELD_SHAPE) * 2.0
        curl = jnp.ones(FIELD_SHAPE)
        inv_mu = jnp.ones(FIELD_SHAPE)
        result = self._call(_make_arrays(H=H_init, inv_permeabilities=inv_mu), config=_make_config(c=c), curl=curl)
        expected = H_init - c * curl * inv_mu  # 2.0 - 0.5 = 1.5
        assert jnp.allclose(result.H, expected)

    def test_isotropic_with_conductivity_formula(self):
        """Lossy magnetic material update (sigma_H != 0)."""
        c = 0.5
        sigma_H = jnp.ones(FIELD_SHAPE) * 1e-4
        inv_mu = jnp.ones(FIELD_SHAPE)
        H_init = jnp.ones(FIELD_SHAPE) * 2.0
        arrays = _make_arrays(H=H_init, inv_permeabilities=inv_mu, magnetic_conductivity=sigma_H)
        result = self._call(arrays, config=_make_config(c=c), curl=CURL_ZERO)
        half = c * sigma_H / eta0 * inv_mu / 2
        expected = (1 - half) * H_init / (1 + half)
        assert jnp.allclose(result.H, expected, rtol=1e-5)

    def test_psi_H_updated_from_curl_E(self):
        """psi_H in the returned container matches the psi returned by curl_E."""
        new_psi = jnp.ones(PSI_SHAPE) * 7.0
        result = self._call(_make_arrays(), psi=new_psi)
        assert jnp.allclose(result.psi_H, new_psi)

    def test_simulate_boundaries_flag_forwarded(self):
        """simulate_boundaries=True is passed as 7th positional arg to curl_E."""
        with patch("fdtdx.fdtd.update.curl_E", return_value=(CURL_ZERO, PSI_ZERO)) as mock_curl:
            update_H(
                time_step=jnp.array(0),
                arrays=_make_arrays(),
                objects=_make_objects(),
                config=_make_config(),
                simulate_boundaries=True,
            )
        assert mock_curl.call_args[0][6] is True

    def test_anisotropic_full_tensor_correct_shape_and_finite(self):
        """inv_mu.shape[0]==9 triggers the anisotropic path; output is (3,N,N,N) and finite."""
        inv_mu = _diag_anisotropic_tensor((NX, NY, NZ))
        result = self._call(_make_arrays(inv_permeabilities=inv_mu))
        assert result.H.shape == FIELD_SHAPE
        assert jnp.all(jnp.isfinite(result.H))

    def test_no_sources_skips_lax_cond(self):
        """With no sources, jax.lax.cond is never invoked."""
        with (
            patch("fdtdx.fdtd.update.curl_E", return_value=(CURL_ZERO, PSI_ZERO)),
            patch("fdtdx.fdtd.update.jax.lax.cond") as mock_cond,
        ):
            update_H(
                time_step=jnp.array(0),
                arrays=_make_arrays(),
                objects=_make_objects(sources=[]),
                config=_make_config(),
                simulate_boundaries=False,
            )
        mock_cond.assert_not_called()

    def test_with_source_calls_lax_cond_and_update_H(self):
        """With one source, jax.lax.cond and source.update_H are each called once."""
        source = Mock()
        source.is_on_at_time_step.return_value = jnp.array(True)
        source.adjust_time_step_by_on_off.return_value = jnp.array(0)
        source.update_H.return_value = jnp.ones(FIELD_SHAPE)

        with (
            patch("fdtdx.fdtd.update.curl_E", return_value=(CURL_ZERO, PSI_ZERO)),
            patch("fdtdx.fdtd.update.jax.lax.cond", side_effect=lambda cond, t, f: t()) as mock_cond,
        ):
            update_H(
                time_step=jnp.array(0),
                arrays=_make_arrays(),
                objects=_make_objects(sources=[source]),
                config=_make_config(),
                simulate_boundaries=False,
            )
        mock_cond.assert_called_once()
        source.is_on_at_time_step.assert_called_once()
        source.update_H.assert_called_once()

    def test_source_inverse_false_passed(self):
        """update_H passes inverse=False to source.update_H."""
        source = Mock()
        source.is_on_at_time_step.return_value = jnp.array(True)
        source.adjust_time_step_by_on_off.return_value = jnp.array(0)
        source.update_H.return_value = jnp.ones(FIELD_SHAPE)

        with (
            patch("fdtdx.fdtd.update.curl_E", return_value=(CURL_ZERO, PSI_ZERO)),
            patch("fdtdx.fdtd.update.jax.lax.cond", side_effect=lambda cond, t, f: t()),
        ):
            update_H(
                time_step=jnp.array(0),
                arrays=_make_arrays(),
                objects=_make_objects(sources=[source]),
                config=_make_config(),
                simulate_boundaries=False,
            )
        assert source.update_H.call_args[1]["inverse"] is False


# ─── TestUpdateHReverse ───────────────────────────────────────────────────────


class TestUpdateHReverse:
    """Tests for the reverse magnetic field update."""

    def _call(self, arrays, objects=None, config=None, curl=None):
        with patch("fdtdx.fdtd.update.curl_E", return_value=(curl if curl is not None else CURL_ZERO, PSI_ZERO)):
            return update_H_reverse(
                time_step=jnp.array(0),
                arrays=arrays,
                objects=objects or _make_objects(),
                config=config or _make_config(),
            )

    def test_isotropic_zero_curl_unchanged(self):
        """With zero curl and no conductivity, H is unchanged."""
        H_init = jnp.ones(FIELD_SHAPE) * 3.0
        result = self._call(_make_arrays(H=H_init), curl=CURL_ZERO)
        assert jnp.allclose(result.H, H_init)

    def test_isotropic_no_conductivity_formula(self):
        """Reverse update without conductivity: H^(n-1/2) = H^(n+1/2) + c * curl * inv_mu."""
        c = 0.5
        H_fwd = jnp.ones(FIELD_SHAPE) * 1.5
        curl = jnp.ones(FIELD_SHAPE)
        inv_mu = jnp.ones(FIELD_SHAPE)
        result = self._call(_make_arrays(H=H_fwd, inv_permeabilities=inv_mu), config=_make_config(c=c), curl=curl)
        expected = H_fwd + c * curl * inv_mu  # 1.5 + 0.5 = 2.0
        assert jnp.allclose(result.H, expected)

    def test_isotropic_with_conductivity_reverses_forward(self):
        """Reverse lossy update recovers the original H field (zero curl)."""
        c = 0.5
        sigma_H = jnp.ones(FIELD_SHAPE) * 1e-4
        inv_mu = jnp.ones(FIELD_SHAPE)
        H_init = jnp.ones(FIELD_SHAPE) * 2.0
        half = c * sigma_H / eta0 * inv_mu / 2
        H_fwd = (1 - half) * H_init / (1 + half)
        arrays = _make_arrays(H=H_fwd, inv_permeabilities=inv_mu, magnetic_conductivity=sigma_H)
        result = self._call(arrays, config=_make_config(c=c), curl=CURL_ZERO)
        assert jnp.allclose(result.H, H_init, rtol=1e-5)

    def test_anisotropic_full_tensor_correct_shape_and_finite(self):
        """inv_mu.shape[0]==9 triggers anisotropic reverse path; output is finite."""
        inv_mu = _diag_anisotropic_tensor((NX, NY, NZ))
        result = self._call(_make_arrays(inv_permeabilities=inv_mu))
        assert result.H.shape == FIELD_SHAPE
        assert jnp.all(jnp.isfinite(result.H))

    def test_with_source_calls_lax_cond(self):
        """Source reverse update is invoked via jax.lax.cond with inverse=True."""
        source = Mock()
        source.is_on_at_time_step.return_value = jnp.array(True)
        source.adjust_time_step_by_on_off.return_value = jnp.array(0)
        source.update_H.return_value = jnp.ones(FIELD_SHAPE)

        with (
            patch("fdtdx.fdtd.update.curl_E", return_value=(CURL_ZERO, PSI_ZERO)),
            patch("fdtdx.fdtd.update.jax.lax.cond", side_effect=lambda cond, t, f: t()) as mock_cond,
        ):
            update_H_reverse(
                time_step=jnp.array(0),
                arrays=_make_arrays(),
                objects=_make_objects(sources=[source]),
                config=_make_config(),
            )
        mock_cond.assert_called_once()
        source.update_H.assert_called_once()
        assert source.update_H.call_args[1]["inverse"] is True


# ─── TestUpdateDetectorStates ─────────────────────────────────────────────────


class TestUpdateDetectorStates:
    """Tests for update_detector_states."""

    def _make_detector(self, name="det1", exact_interpolation=False):
        d = Mock()
        d.name = name
        d.exact_interpolation = exact_interpolation
        d._is_on_at_time_step_arr = [True] * 100
        d.update.return_value = {"value": jnp.zeros((1,))}
        return d

    def test_forward_uses_forward_detectors(self):
        """inverse=False iterates over objects.forward_detectors."""
        det = self._make_detector()
        objects = Mock()
        objects.boundary_objects = []
        objects.forward_detectors = [det]
        arrays = _make_arrays(detector_states={det.name: {}})
        H_prev = jnp.zeros(FIELD_SHAPE)

        with (
            patch(
                "fdtdx.fdtd.update.interpolate_fields",
                return_value=(jnp.ones(FIELD_SHAPE), jnp.ones(FIELD_SHAPE)),
            ),
            patch(
                "fdtdx.fdtd.update.jax.lax.cond",
                side_effect=lambda cond, t, f, e, h, d: t(e, h, d),
            ) as mock_cond,
        ):
            update_detector_states(jnp.array(0), arrays, objects, H_prev, inverse=False)
        mock_cond.assert_called_once()

    def test_backward_uses_backward_detectors(self):
        """inverse=True iterates over objects.backward_detectors."""
        det = self._make_detector()
        objects = Mock()
        objects.boundary_objects = []
        objects.backward_detectors = [det]
        arrays = _make_arrays(detector_states={det.name: {}})
        H_prev = jnp.zeros(FIELD_SHAPE)

        with (
            patch(
                "fdtdx.fdtd.update.interpolate_fields",
                return_value=(jnp.ones(FIELD_SHAPE), jnp.ones(FIELD_SHAPE)),
            ),
            patch(
                "fdtdx.fdtd.update.jax.lax.cond",
                side_effect=lambda cond, t, f, e, h, d: t(e, h, d),
            ) as mock_cond,
        ):
            update_detector_states(jnp.array(0), arrays, objects, H_prev, inverse=True)
        mock_cond.assert_called_once()

    def test_no_forward_detectors_skips_cond(self):
        """With no forward detectors, jax.lax.cond is never called."""
        objects = Mock()
        objects.boundary_objects = []
        objects.forward_detectors = []
        arrays = _make_arrays(detector_states={})
        H_prev = jnp.zeros(FIELD_SHAPE)

        with (
            patch(
                "fdtdx.fdtd.update.interpolate_fields",
                return_value=(jnp.ones(FIELD_SHAPE), jnp.ones(FIELD_SHAPE)),
            ),
            patch("fdtdx.fdtd.update.jax.lax.cond") as mock_cond,
        ):
            update_detector_states(jnp.array(0), arrays, objects, H_prev, inverse=False)
        mock_cond.assert_not_called()

    def test_exact_interpolation_passes_interpolated_fields(self):
        """exact_interpolation=True: helper_fn receives interpolated E and H."""
        det = self._make_detector(exact_interpolation=True)
        objects = Mock()
        objects.boundary_objects = []
        objects.forward_detectors = [det]
        arrays = _make_arrays(detector_states={det.name: {}})
        H_prev = jnp.zeros(FIELD_SHAPE)
        interp_E = jnp.ones(FIELD_SHAPE) * 42.0
        interp_H = jnp.ones(FIELD_SHAPE) * 99.0
        captured = {}

        def fake_cond(cond_val, t_fn, f_fn, e_arg, h_arg, d_arg):
            captured["E"] = e_arg
            captured["H"] = h_arg
            return t_fn(e_arg, h_arg, d_arg)

        with (
            patch("fdtdx.fdtd.update.interpolate_fields", return_value=(interp_E, interp_H)),
            patch("fdtdx.fdtd.update.jax.lax.cond", side_effect=fake_cond),
        ):
            update_detector_states(jnp.array(0), arrays, objects, H_prev, inverse=False)

        assert jnp.allclose(captured["E"], interp_E)
        assert jnp.allclose(captured["H"], interp_H)

    def test_no_interpolation_passes_raw_fields(self):
        """exact_interpolation=False: helper_fn receives raw arrays.E and arrays.H."""
        det = self._make_detector(exact_interpolation=False)
        objects = Mock()
        objects.boundary_objects = []
        objects.forward_detectors = [det]
        raw_E = jnp.ones(FIELD_SHAPE) * 5.0
        raw_H = jnp.ones(FIELD_SHAPE) * 6.0
        arrays = _make_arrays(E=raw_E, H=raw_H, detector_states={det.name: {}})
        H_prev = jnp.zeros(FIELD_SHAPE)
        captured = {}

        def fake_cond(cond_val, t_fn, f_fn, e_arg, h_arg, d_arg):
            captured["E"] = e_arg
            captured["H"] = h_arg
            return t_fn(e_arg, h_arg, d_arg)

        with (
            patch(
                "fdtdx.fdtd.update.interpolate_fields",
                return_value=(jnp.ones(FIELD_SHAPE) * 99.0, jnp.ones(FIELD_SHAPE) * 99.0),
            ),
            patch("fdtdx.fdtd.update.jax.lax.cond", side_effect=fake_cond),
        ):
            update_detector_states(jnp.array(0), arrays, objects, H_prev, inverse=False)

        # Should receive the raw fields, not the interpolated ones
        assert jnp.allclose(captured["E"], raw_E)
        assert jnp.allclose(captured["H"], raw_H)

    def test_interpolate_fields_receives_h_avg(self):
        """interpolate_fields receives (H_prev + arrays.H) / 2 as H_field."""
        objects = Mock()
        objects.boundary_objects = []
        objects.forward_detectors = []
        H_field = jnp.ones(FIELD_SHAPE) * 4.0
        H_prev = jnp.ones(FIELD_SHAPE) * 2.0
        arrays = _make_arrays(H=H_field, detector_states={})

        with patch(
            "fdtdx.fdtd.update.interpolate_fields",
            return_value=(jnp.zeros(FIELD_SHAPE), jnp.zeros(FIELD_SHAPE)),
        ) as mock_interp:
            update_detector_states(jnp.array(0), arrays, objects, H_prev, inverse=False)

        expected_H_avg = (H_prev + H_field) / 2
        actual_H_arg = mock_interp.call_args[1]["H_field"]
        assert jnp.allclose(actual_H_arg, expected_H_avg)


# ─── TestCollectInterfaces ────────────────────────────────────────────────────


class TestCollectInterfaces:
    """Tests for the collect_interfaces function."""

    @pytest.fixture
    def setup(self):
        recording_state = Mock()
        arrays = _make_arrays(recording_state=recording_state)
        recorder = Mock()
        recorder.compress.return_value = Mock()
        gradient_config = Mock()
        gradient_config.recorder = recorder
        config = Mock()
        config.gradient_config = gradient_config
        objects = Mock()
        objects.pml_objects = []
        key = jax.random.PRNGKey(0)
        return arrays, config, objects, key, recorder

    def test_compress_is_called(self, setup):
        """collect_interfaces calls recorder.compress with the collected values."""
        arrays, config, objects, key, recorder = setup
        fake_values = {"pml1_E": jnp.zeros((3, 1, NX, NY))}
        with patch("fdtdx.fdtd.update.collect_boundary_interfaces", return_value=fake_values):
            result = collect_interfaces(jnp.array(0), arrays, objects, config, key)
        recorder.compress.assert_called_once()
        assert result is not None

    def test_no_gradient_config_raises(self, setup):
        arrays, config, objects, key, _ = setup
        config.gradient_config = None
        with pytest.raises(Exception, match="Need recorder to record boundaries"):
            collect_interfaces(jnp.array(0), arrays, objects, config, key)

    def test_no_recorder_raises(self, setup):
        arrays, config, objects, key, _ = setup
        config.gradient_config.recorder = None
        with pytest.raises(Exception, match="Need recorder to record boundaries"):
            collect_interfaces(jnp.array(0), arrays, objects, config, key)

    def test_no_recording_state_raises(self, setup):
        arrays, config, objects, key, _ = setup
        arrays = arrays.aset("recording_state", None)
        with pytest.raises(Exception, match="Need recording state to record boundaries"):
            collect_interfaces(jnp.array(0), arrays, objects, config, key)

    def test_recording_state_updated_in_result(self, setup):
        """collect_interfaces stores the compressed state back into recording_state."""
        arrays, config, objects, key, recorder = setup
        new_state = Mock()
        recorder.compress.return_value = new_state
        with patch("fdtdx.fdtd.update.collect_boundary_interfaces", return_value={}):
            result = collect_interfaces(jnp.array(0), arrays, objects, config, key)
        assert result.recording_state is new_state


# ─── TestAddInterfaces ────────────────────────────────────────────────────────


class TestAddInterfaces:
    """Tests for the add_interfaces function."""

    @pytest.fixture
    def setup(self):
        recording_state = Mock()
        arrays = _make_arrays(recording_state=recording_state)
        fake_values = {"pml1_E": jnp.zeros((3, 1, NX, NY)), "pml1_H": jnp.zeros((3, 1, NX, NY))}
        new_state = Mock()
        recorder = Mock()
        recorder.decompress.return_value = (fake_values, new_state)
        gradient_config = Mock()
        gradient_config.recorder = recorder
        config = Mock()
        config.gradient_config = gradient_config
        objects = Mock()
        objects.pml_objects = []
        key = jax.random.PRNGKey(0)
        return arrays, config, objects, key, recorder, new_state

    def test_decompress_and_add_interfaces_called(self, setup):
        """add_interfaces calls recorder.decompress and add_boundary_interfaces."""
        arrays, config, objects, key, recorder, _ = setup
        with patch("fdtdx.fdtd.update.add_boundary_interfaces", return_value=arrays) as mock_add:
            result = add_interfaces(jnp.array(0), arrays, objects, config, key)
        recorder.decompress.assert_called_once()
        mock_add.assert_called_once()
        assert result is not None

    def test_no_gradient_config_raises(self, setup):
        arrays, config, objects, key, _, _ = setup
        config.gradient_config = None
        with pytest.raises(Exception, match="Need recorder to record boundaries"):
            add_interfaces(jnp.array(0), arrays, objects, config, key)

    def test_no_recorder_raises(self, setup):
        arrays, config, objects, key, _, _ = setup
        config.gradient_config.recorder = None
        with pytest.raises(Exception, match="Need recorder to record boundaries"):
            add_interfaces(jnp.array(0), arrays, objects, config, key)

    def test_no_recording_state_raises(self, setup):
        arrays, config, objects, key, _, _ = setup
        arrays = arrays.aset("recording_state", None)
        with pytest.raises(Exception, match="Need recording state to record boundaries"):
            add_interfaces(jnp.array(0), arrays, objects, config, key)

    def test_recording_state_updated_from_decompress(self, setup):
        """add_interfaces stores the decompressed state into recording_state."""
        arrays, config, objects, key, _, new_state = setup
        # Pass through the arrays argument so the aset("recording_state", ...) call is preserved
        with patch(
            "fdtdx.fdtd.update.add_boundary_interfaces",
            side_effect=lambda arrays, values, pml_objects: arrays,
        ):
            result = add_interfaces(jnp.array(0), arrays, objects, config, key)
        assert result.recording_state is new_state
