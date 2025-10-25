from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.physics.modes import (
    ModeTupleType,
    compute_mode,
    compute_mode_polarization_fraction,
    sort_modes,
    tidy3d_mode_computation_wrapper,
)
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.units import A, V, m


class TestModeTupleType:
    """Test the ModeTupleType named tuple."""

    def test_mode_tuple_creation(self):
        """Test creating a ModeTupleType instance."""
        mode = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.array([1, 2, 3]),
            Ey=np.array([4, 5, 6]),
            Ez=np.array([7, 8, 9]),
            Hx=np.array([10, 11, 12]),
            Hy=np.array([13, 14, 15]),
            Hz=np.array([16, 17, 18]),
        )

        assert mode.neff == 1.5 + 0.1j
        assert np.array_equal(mode.Ex, np.array([1, 2, 3]))
        assert isinstance(mode, tuple)  # Should behave like a tuple


class TestComputeModePolarizationFraction:
    """Test the compute_mode_polarization_fraction function."""

    def test_te_polarization(self):
        """Test TE polarization fraction calculation."""
        # Create a mode with stronger E1 component (TE-like)
        mode = ModeTupleType(
            neff=1.5,
            Ex=np.array([[2.0, 2.0], [2.0, 2.0]]),  # E1 component (axis 0)
            Ey=np.array([[1.0, 1.0], [1.0, 1.0]]),  # E2 component (axis 1)
            Ez=np.array([[0.0, 0.0], [0.0, 0.0]]),
            Hx=np.array([[0.0, 0.0], [0.0, 0.0]]),
            Hy=np.array([[0.0, 0.0], [0.0, 0.0]]),
            Hz=np.array([[0.0, 0.0], [0.0, 0.0]]),
        )

        fraction = compute_mode_polarization_fraction(mode, (0, 1), "te")

        # Expected: |E1|^2 / (|E1|^2 + |E2|^2) = 16 / (16 + 4) = 0.8
        expected = 16.0 / (16.0 + 4.0)
        assert fraction == pytest.approx(expected)

    def test_tm_polarization(self):
        """Test TM polarization fraction calculation."""
        # Create a mode with stronger E2 component (TM-like)
        mode = ModeTupleType(
            neff=1.5,
            Ex=np.array([[1.0, 1.0], [1.0, 1.0]]),  # E1 component (axis 0)
            Ey=np.array([[3.0, 3.0], [3.0, 3.0]]),  # E2 component (axis 1)
            Ez=np.array([[0.0, 0.0], [0.0, 0.0]]),
            Hx=np.array([[0.0, 0.0], [0.0, 0.0]]),
            Hy=np.array([[0.0, 0.0], [0.0, 0.0]]),
            Hz=np.array([[0.0, 0.0], [0.0, 0.0]]),
        )

        fraction = compute_mode_polarization_fraction(mode, (0, 1), "tm")

        # Expected: |E2|^2 / (|E1|^2 + |E2|^2) = 36 / (4 + 36) = 0.9
        expected = 36.0 / (4.0 + 36.0)
        assert fraction == pytest.approx(expected)

    def test_invalid_pol_raises_error(self):
        """Test that invalid polarization raises ValueError."""
        mode = ModeTupleType(
            neff=1.5,
            Ex=np.array([[1.0]]),
            Ey=np.array([[1.0]]),
            Ez=np.array([[0.0]]),
            Hx=np.array([[0.0]]),
            Hy=np.array([[0.0]]),
            Hz=np.array([[0.0]]),
        )

        with pytest.raises(ValueError, match="pol must be 'te' or 'tm'"):
            compute_mode_polarization_fraction(mode, (0, 1), "invalid_pol")


class TestSortModes:
    """Test the sort_modes function."""

    def create_test_modes(self):
        """Helper function to create test modes."""
        mode1 = ModeTupleType(
            neff=2.0 + 0.1j,
            Ex=np.ones((2, 2)),
            Ey=np.ones((2, 2)) * 0.1,
            Ez=np.zeros((2, 2)),
            Hx=np.zeros((2, 2)),
            Hy=np.zeros((2, 2)),
            Hz=np.zeros((2, 2)),
        )
        mode2 = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((2, 2)) * 0.1,
            Ey=np.ones((2, 2)),
            Ez=np.zeros((2, 2)),
            Hx=np.zeros((2, 2)),
            Hy=np.zeros((2, 2)),
            Hz=np.zeros((2, 2)),
        )
        mode3 = ModeTupleType(
            neff=3.0 + 0.1j,
            Ex=np.ones((2, 2)) * 0.6,
            Ey=np.ones((2, 2)) * 0.4,
            Ez=np.zeros((2, 2)),
            Hx=np.zeros((2, 2)),
            Hy=np.zeros((2, 2)),
            Hz=np.zeros((2, 2)),
        )
        return [mode1, mode2, mode3]

    def test_sort_no_filter(self):
        """Test sorting without polarization filter."""
        modes = self.create_test_modes()
        sorted_modes = sort_modes(modes, None, (0, 1))

        # Should be sorted by descending real part of neff
        expected_order = [3.0, 2.0, 1.5]
        actual_order = [float(np.real(m.neff)) for m in sorted_modes]
        assert actual_order == expected_order

    def test_sort_te_filter(self):
        """Test sorting with TE polarization filter."""
        modes = self.create_test_modes()
        sorted_modes = sort_modes(modes, "te", (0, 1))

        # mode1 has strong Ex (TE-like), mode2 has weak Ex, mode3 is mixed but Ex > Ey
        # TE modes should come first, sorted by neff
        assert sorted_modes[0].neff == 3.0 + 0.1j  # Highest neff TE-like
        assert sorted_modes[1].neff == 2.0 + 0.1j  # TE-like
        assert sorted_modes[2].neff == 1.5 + 0.1j  # TM-like

    def test_sort_tm_filter(self):
        """Test sorting with TM polarization filter."""
        modes = self.create_test_modes()
        sorted_modes = sort_modes(modes, "tm", (0, 1))

        # mode2 has strong Ey (TM-like), should come first among TM modes
        assert sorted_modes[0].neff == 1.5 + 0.1j  # Most TM-like
        # The other modes should follow


class TestComputeMode:
    """Test the compute_mode function."""

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_compute_mode_basic(self, mock_normalize, mock_tidy3d_wrapper):
        """Test basic compute_mode functionality."""
        # Mock the tidy3d wrapper
        mock_mode = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((5, 5), dtype=np.complex64),
            Ey=np.ones((5, 5), dtype=np.complex64),
            Ez=np.ones((5, 5), dtype=np.complex64),
            Hx=np.ones((5, 5), dtype=np.complex64),
            Hy=np.ones((5, 5), dtype=np.complex64),
            Hz=np.ones((5, 5), dtype=np.complex64),
        )
        mock_tidy3d_wrapper.return_value = [mock_mode]

        # Mock the normalization function
        mock_normalize.return_value = (
            jnp.ones((3, 5, 1, 5), dtype=jnp.complex64),
            jnp.ones((3, 5, 1, 5), dtype=jnp.complex64),
        )

        # Test inputs
        frequency = 2e14  # 200 THz
        inv_permittivities = jnp.ones((1, 5, 5)) * 0.25  # eps = 4.0
        inv_permeabilities = 1.0  # mu = 1.0
        resolution = 1e-8  # 10 nm

        # Test the function
        E, H, neff = compute_mode(
            frequency=frequency,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            resolution=resolution,
            direction="+",
            mode_index=0,
        )

        # Verify the outputs
        assert E.shape == (3, 5, 1, 5)
        assert H.shape == (3, 5, 1, 5)
        assert neff.dtype == jnp.complex64

    def test_invalid_permittivities_shape(self):
        """Test that invalid permittivities shape raises exception."""
        frequency = 2e14
        # 3D array but not squeezable to 2D
        inv_permittivities = jnp.ones((2, 2, 2))
        inv_permeabilities = 1.0
        resolution = 1e-8

        with pytest.raises(Exception, match="Invalid shape of inv_permittivities"):
            compute_mode(frequency, inv_permittivities, inv_permeabilities, resolution, "+")

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    def test_different_propagation_axes(self, mock_tidy3d_wrapper):
        """Test different propagation axis configurations."""
        mock_mode = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((3, 4), dtype=np.complex64),
            Ey=np.ones((3, 4), dtype=np.complex64),
            Ez=np.ones((3, 4), dtype=np.complex64),
            Hx=np.ones((3, 4), dtype=np.complex64),
            Hy=np.ones((3, 4), dtype=np.complex64),
            Hz=np.ones((3, 4), dtype=np.complex64),
        )
        mock_tidy3d_wrapper.return_value = [mock_mode]

        # Mock normalize_by_poynting_flux to return simple arrays
        with patch("fdtdx.core.physics.modes.normalize_by_poynting_flux") as mock_normalize:
            mock_normalize.return_value = (
                jnp.ones((3, 3, 1, 4), dtype=jnp.complex64),
                jnp.ones((3, 3, 1, 4), dtype=jnp.complex64),
            )

            # Test propagation along axis 0 (shape: 1, 3, 4)
            inv_permittivities = jnp.ones((1, 3, 4))
            E, H, neff = compute_mode(2e14, inv_permittivities, 1.0, 1e-8, "+")
            assert E.shape == (3, 3, 1, 4)

            # Test propagation along axis 1 (shape: 3, 1, 4)
            inv_permittivities = jnp.ones((3, 1, 4))
            E, H, neff = compute_mode(2e14, inv_permittivities, 1.0, 1e-8, "+")
            assert E.shape == (3, 3, 1, 4)  # Note: axis handling in the function


class TestTidy3DModeComputationWrapper:
    """Test the tidy3d_mode_computation_wrapper function."""

    def create_mock_eh_data(self, shape, num_modes=1):
        """Helper to create properly structured mock EH data."""
        # Create numpy arrays with proper shape and add squeeze method
        if num_modes == 1:
            # Single mode - 2D arrays
            E_data = (
                np.ones(shape, dtype=np.complex64),
                np.ones(shape, dtype=np.complex64),
                np.ones(shape, dtype=np.complex64),
            )
            H_data = (
                np.ones(shape, dtype=np.complex64),
                np.ones(shape, dtype=np.complex64),
                np.ones(shape, dtype=np.complex64),
            )
        else:
            # Multiple modes - 3D arrays with mode dimension last
            E_data = (
                np.ones(shape + (num_modes,), dtype=np.complex64),
                np.ones(shape + (num_modes,), dtype=np.complex64),
                np.ones(shape + (num_modes,), dtype=np.complex64),
            )
            H_data = (
                np.ones(shape + (num_modes,), dtype=np.complex64),
                np.ones(shape + (num_modes,), dtype=np.complex64),
                np.ones(shape + (num_modes,), dtype=np.complex64),
            )

        # Create a mock object that behaves like the expected EH structure
        class MockEH:
            def __init__(self, E_data, H_data):
                self.E_data = E_data
                self.H_data = H_data

            def squeeze(self):
                # Return a tuple that can be unpacked as ((Ex, Ey, Ez), (Hx, Hy, Hz))
                return (self.E_data, self.H_data)

        return MockEH(E_data, H_data)

    @patch("fdtdx.core.physics.modes._compute_modes")
    def test_single_mode(self, mock_compute_modes):
        """Test single mode computation."""
        # Create properly structured mock data
        mock_EH = self.create_mock_eh_data((5, 5), num_modes=1)
        mock_neffs = 1.5 + 0.1j

        mock_compute_modes.return_value = (mock_EH, mock_neffs, None)

        # Test inputs
        frequency = 2e14
        permittivity = np.ones((5, 5))
        coords = [np.linspace(0, 1, 6), np.linspace(0, 1, 6)]  # x, y coordinates

        modes = tidy3d_mode_computation_wrapper(
            frequency=frequency, permittivity_cross_section=permittivity, coords=coords, direction="+", num_modes=1
        )

        assert len(modes) == 1
        assert modes[0].neff == 1.5 + 0.1j
        assert modes[0].Ex.shape == (5, 5)

    @patch("fdtdx.core.physics.modes._compute_modes")
    def test_multiple_modes(self, mock_compute_modes):
        """Test multiple mode computation."""
        # Create properly structured mock data for multiple modes
        mock_EH = self.create_mock_eh_data((5, 5), num_modes=3)
        mock_neffs = np.array([1.5 + 0.1j, 1.4 + 0.1j, 1.3 + 0.1j])

        mock_compute_modes.return_value = (mock_EH, mock_neffs, None)

        frequency = 2e14
        permittivity = np.ones((5, 5))
        coords = [np.linspace(0, 1, 6), np.linspace(0, 1, 6)]

        modes = tidy3d_mode_computation_wrapper(
            frequency=frequency, permittivity_cross_section=permittivity, coords=coords, direction="+", num_modes=3
        )

        assert len(modes) == 3
        assert modes[0].neff == 1.5 + 0.1j
        assert modes[1].neff == 1.4 + 0.1j
        assert modes[2].neff == 1.3 + 0.1j

    @patch("fdtdx.core.physics.modes._compute_modes")
    def test_with_permeability(self, mock_compute_modes):
        """Test mode computation with permeability cross-section."""
        mock_EH = self.create_mock_eh_data((4, 4), num_modes=1)
        mock_neffs = 1.6 + 0.2j

        mock_compute_modes.return_value = (mock_EH, mock_neffs, None)

        frequency = 2e14
        permittivity = np.ones((4, 4))
        permeability = np.ones((4, 4)) * 2.0  # Non-unity permeability
        coords = [np.linspace(0, 1, 5), np.linspace(0, 1, 5)]

        modes = tidy3d_mode_computation_wrapper(
            frequency=frequency,
            permittivity_cross_section=permittivity,
            coords=coords,
            direction="+",
            permeability_cross_section=permeability,
            num_modes=1,
        )

        assert len(modes) == 1
        assert modes[0].neff == 1.6 + 0.2j


def test_modes_basic():
    inv_perm = jnp.ones((200, 200, 1))
    inv_perm = inv_perm.at[75:125, 75:125].set(1 / 12)

    mode_E, mode_H, eff_idx = compute_mode(
        wave_character=WaveCharacter(wavelength=1.55e-6 * m),
        inv_permittivities=inv_perm,
        inv_permeabilities=1.0,
        resolution=10e-9 * m,
        direction="+",
    )
    assert isinstance(eff_idx, jax.Array)
    assert mode_E.unit.dim == (V / m).unit.dim
    assert mode_H.unit.dim == (A / m).unit.dim
