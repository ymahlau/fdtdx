import math
import jax
import plum
import pytest
from fdtdx.core.fraction import Fraction
from fdtdx.functional.numpy import sqrt, roll, transpose
from fdtdx.units.typing import SI
from fdtdx.units.unitful import Unit, Unitful
from fdtdx.units.composite import Hz, s, ms, m_per_s 
import jax.numpy as jnp

def test_sqrt_unitful_even_integer_dimensions():
    """Test sqrt with Unitful object having even integer dimensions"""
    # Create a unit with m^2 dimension (area)
    area_unit = Unit(scale=0, dim={SI.m: 2})
    area = Unitful(val=jnp.array(16.0), unit=area_unit)
    
    result = sqrt(area)
    
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 4.0)  # sqrt(16) = 4
    assert result.unit.dim == {SI.m: 1}  # sqrt(m^2) = m^1


def test_sqrt_unitful_odd_integer_dimensions():
    """Test sqrt with Unitful object having odd integer dimensions"""
    # Create a unit with kg^3 dimension 
    mass_cube_unit = Unit(scale=0, dim={SI.kg: 3})
    mass_cube = Unitful(val=jnp.array(8.0), unit=mass_cube_unit)
    
    result = jax.lax.sqrt(mass_cube)  # type: ignore
    
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 2.828427124746190)  # sqrt(8) ≈ 2.828
    assert result.unit.dim == {SI.kg: Fraction(3, 2)}  # sqrt(kg^3) = kg^(3/2)


def test_sqrt_unitful_fractional_dimensions_and_odd_scale():
    """Test sqrt with Unitful object having fractional dimensions and odd scale"""
    # Create a unit with m^(2/3) dimension and scale=3 (factor of 1000)
    fractional_unit = Unit(scale=3, dim={SI.m: Fraction(2, 3)})
    value = Unitful(val=jnp.array(4.0), unit=fractional_unit)
    
    result = jnp.sqrt(value)  # type: ignore
    
    assert isinstance(result, Unitful)
    # sqrt(4 * 1000) * sqrt(10) = 2 * sqrt(10000) = 2 * 100 = 200 (approximately)
    # But scale adjustment: sqrt(4) * sqrt(10) with scale floor(3/2) = 1
    # So: sqrt(4) * sqrt(10) * 10^1 = 2 * 3.162 * 10 ≈ 63.24
    expected_value = 2.0 * math.sqrt(10) * 10  # 2 * sqrt(10) * 10^1
    assert jnp.allclose(result.value(), expected_value)
    assert result.unit.dim == {SI.m: Fraction(1, 3)}  # sqrt(m^(2/3)) = m^(1/3)
    
def test_sqrt_jax_array():
    """Test sqrt with regular JAX array (non-Unitful)"""
    # Test that the overloaded sqrt still works with regular JAX arrays
    array = jnp.array([4.0, 9.0, 16.0, 25.0])
    
    result = jnp.sqrt(array)  # type: ignore
    
    # Expected: [sqrt(4), sqrt(9), sqrt(16), sqrt(25)] = [2, 3, 4, 5]
    expected = jnp.array([2.0, 3.0, 4.0, 5.0])
    assert jnp.allclose(result, expected)
    # Should return a regular JAX array, not a Unitful object
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)
    
    
def test_roll_1d_unitful_basic():
    """Test basic roll operation on 1D Unitful array"""
    # Create a 1D array of forces
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})  # Newtons
    forces = Unitful(val=jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]), unit=force_unit)
    
    # Roll by 2 positions to the right
    result = jnp.roll(forces, shift=2)  # type: ignore
    
    assert isinstance(result, Unitful)
    expected_vals = jnp.array([40.0, 50.0, 10.0, 20.0, 30.0])
    assert jnp.allclose(result.value(), expected_vals)
    assert result.unit.dim == {SI.kg: 1, SI.m: 1, SI.s: -2}


def test_roll_2d_unitful_with_axis_and_scale():
    """Test roll operation on 2D Unitful array with specific axis and unit scale"""
    # Create a 2D array of temperatures with millikelvin scale
    temp_unit = Unit(scale=-3, dim={SI.K: 1})  # millikelvin
    temperatures = Unitful(val=jnp.array([
        [100.0, 200.0, 300.0],
        [400.0, 500.0, 600.0],
        [700.0, 800.0, 900.0]
    ]), unit=temp_unit)
    
    # Roll along axis 1 (columns) by -1
    result = jnp.roll(temperatures, shift=-1, axis=1)  # type: ignore
    
    assert isinstance(result, Unitful)
    expected_vals = jnp.array([
        [200.0, 300.0, 100.0],
        [500.0, 600.0, 400.0],
        [800.0, 900.0, 700.0]
    ])
    assert jnp.allclose(result.value(), expected_vals * 1e-3)  # Convert to Kelvin
    assert result.unit.dim == {SI.K: 1}


def test_roll_multi_axis_fractional_dimensions():
    """Test roll operation with multiple axes on Unitful array with fractional dimensions"""
    # Create a unit with fractional dimensions: m^(3/2) * kg^(-1/4)
    complex_unit = Unit(scale=1, dim={SI.m: Fraction(3, 2), SI.kg: Fraction(-1, 4)})
    values = Unitful(val=jnp.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]]
    ]), unit=complex_unit)
    
    # Roll along multiple axes: axis 0 by 1, axis 2 by -1
    result = roll(values, shift=[1, -1], axis=[0, 2])
    
    assert isinstance(result, Unitful)
    # Expected: first roll axis 0 by 1 (shifts the 3x2x2 along first dimension)
    # then roll axis 2 by -1 (shifts the innermost dimension)
    expected_vals = jnp.array([
        [[10.0, 9.0], [12.0, 11.0]],
        [[2.0, 1.0], [4.0, 3.0]],
        [[6.0, 5.0], [8.0, 7.0]]
    ])
    assert jnp.allclose(result.value(), expected_vals * 10)  # scale=1 means *10
    assert result.unit.dim == {SI.m: Fraction(3, 2), SI.kg: Fraction(-1, 4)}
    assert result.shape == values.shape  # Shape should be preserved


def test_square_unitful_basic_dimensions():
    """Test square with Unitful object having basic dimensions"""
    # Create a unit with m dimension (length)
    length_unit = Unit(scale=0, dim={SI.m: 1})
    length = Unitful(val=jnp.array(5.0), unit=length_unit)
    
    result = jnp.square(length)  # type: ignore
    
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 25.0)  # square(5) = 25
    assert result.unit.dim == {SI.m: 2}  # square(m^1) = m^2 (area)


def test_square_unitful_complex_fractional_dimensions_with_scale():
    """Test square with Unitful object having fractional dimensions and non-zero scale"""
    # Create a unit with kg^(1/3) * s^(-2/3) dimension and scale=2 (factor of 100)
    complex_unit = Unit(scale=2, dim={SI.kg: Fraction(1, 3), SI.s: Fraction(-2, 3)})
    value = Unitful(val=jnp.array(3.0), unit=complex_unit)
    
    result = jnp.square(value)  # type: ignore
    
    assert isinstance(result, Unitful)
    # square(3 * 100) = 9 * 10000 = 90000
    # But with scale optimization: 9 * 10^4 = 9 * 10^4
    assert jnp.allclose(result.value(), 9.0 * 10**4)  # 3^2 * (10^2)^2 = 9 * 10^4
    # Dimensions: square(kg^(1/3) * s^(-2/3)) = kg^(2/3) * s^(-4/3)
    assert result.unit.dim == {SI.kg: Fraction(2, 3), SI.s: Fraction(-4, 3)}


def test_square_unitful_complex_number():
    """Test square with Unitful object containing complex Python scalar"""
    # Create a unit with electric charge dimension (Coulombs: A*s)
    charge_unit = Unit(scale=-3, dim={SI.A: 1, SI.s: 1})  # milliCoulombs
    # Complex impedance-like value: 3 + 4j
    complex_charge = Unitful(val=3.0 + 4.0j, unit=charge_unit)
    
    result = jnp.square(complex_charge)  # type: ignore
    
    assert isinstance(result, Unitful)
    # square(3 + 4j) = (3 + 4j)^2 = 9 + 24j + 16j^2 = 9 + 24j - 16 = -7 + 24j
    expected_complex = -7.0 + 24.0j
    # With scale factor: result * 10^(-3*2) = result * 10^(-6)
    assert jnp.allclose(result.value(), expected_complex * 10**(-6))
    # Dimensions: square(A^1 * s^1) = A^2 * s^2
    assert result.unit.dim == {SI.A: 2, SI.s: 2}
    
    
def test_cross_unitful_basic_physics():
    """Test cross product with physics vectors: velocity × magnetic field = electric field"""
    # Create velocity vector: m/s
    velocity_unit = Unit(scale=0, dim={SI.m: 1, SI.s: -1})
    velocity = Unitful(val=jnp.array([3.0, 0.0, 0.0]), unit=velocity_unit)
    
    # Create magnetic field vector: Tesla (kg/(A*s^2))
    b_field_unit = Unit(scale=0, dim={SI.kg: 1, SI.A: -1, SI.s: -2})
    b_field = Unitful(val=jnp.array([0.0, 2.0, 0.0]), unit=b_field_unit)
    
    result = jnp.cross(velocity, b_field)  # type: ignore
    
    assert isinstance(result, Unitful)
    # v × B = [3, 0, 0] × [0, 2, 0] = [0, 0, 6]
    expected_vals = jnp.array([0.0, 0.0, 6.0])
    assert jnp.allclose(result.value(), expected_vals)
    # Dimensions: (m/s) × (kg/(A*s^2)) = kg*m/(A*s^3) = electric field units
    assert result.unit.dim == {SI.kg: 1, SI.m: 1, SI.A: -1, SI.s: -3}


def test_cross_unitful_fractional_dimensions_with_scale():
    """Test cross product with fractional dimensions and different scales"""
    # First vector: kg^(1/3) * m^(2/3) with scale=1 (factor of 10)
    unit_a = Unit(scale=1, dim={SI.kg: Fraction(1, 3), SI.m: Fraction(2, 3)})
    vector_a = Unitful(val=jnp.array([1.0, 2.0, 0.0]), unit=unit_a)
    
    # Second vector: s^(-1/2) * A^(3/4) with scale=-2 (factor of 0.01)
    unit_b = Unit(scale=-2, dim={SI.s: Fraction(-1, 2), SI.A: Fraction(3, 4)})
    vector_b = Unitful(val=jnp.array([0.0, 1.0, 3.0]), unit=unit_b)
    
    result = jnp.cross(vector_a, vector_b)  # type: ignore
    
    assert isinstance(result, Unitful)
    # Cross product: [1, 2, 0] × [0, 1, 3] = [6, -3, 1]
    # Scale: 10 * 0.01 = 0.1, so values are multiplied by 0.1
    expected_vals = jnp.array([6.0, -3.0, 1.0]) * 0.1
    assert jnp.allclose(result.value(), expected_vals)
    # Combined dimensions: kg^(1/3) * m^(2/3) * s^(-1/2) * A^(3/4)
    expected_dims = {
        SI.kg: Fraction(1, 3),
        SI.m: Fraction(2, 3), 
        SI.s: Fraction(-1, 2),
        SI.A: Fraction(3, 4)
    }
    assert result.unit.dim == expected_dims


def test_cross_unitful_with_axis_parameter():
    """Test cross product with axis parameter on higher dimensional arrays"""
    # Create force vectors: Newton = kg*m/s^2
    force_unit = Unit(scale=2, dim={SI.kg: 1, SI.m: 1, SI.s: -2})  # scale=2 -> factor of 100
    # 2x3 array of force vectors
    forces = Unitful(val=jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]), unit=force_unit)
    
    # Create position vectors: meters
    position_unit = Unit(scale=-1, dim={SI.m: 1})  # scale=-1 -> factor of 0.1
    # 2x3 array of position vectors  
    positions = Unitful(val=jnp.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]), unit=position_unit)
    
    result = jnp.cross(forces, positions, axis=1)  # type: ignore
    
    assert isinstance(result, Unitful)
    # First cross: [1, 0, 0] × [0, 1, 0] = [0, 0, 1]
    # Second cross: [0, 1, 0] × [0, 0, 1] = [1, 0, 0]
    # Scale factor: 100 * 0.1 = 10
    expected_vals = jnp.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ]) * 10
    assert jnp.allclose(result.value(), expected_vals)
    # Dimensions: (kg*m/s^2) × m = kg*m^2/s^2 (torque units)
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.s: -2}
    
    
def test_conj_unitful_complex_impedance():
    """Test complex conjugate with Unitful object containing complex impedance values"""
    # Create impedance unit: Ohms = kg*m^2/(A^2*s^3)
    impedance_unit = Unit(scale=-3, dim={SI.kg: 1, SI.m: 2, SI.A: -2, SI.s: -3})  # milliohms
    # Complex impedance array: resistance + j*reactance
    complex_impedances = Unitful(val=jnp.array([
        3.0 + 4.0j,    # Z1 = 3 + 4j milliohms
        -2.0 + 5.0j,   # Z2 = -2 + 5j milliohms  
        1.0 - 3.0j,    # Z3 = 1 - 3j milliohms
        6.0 + 0.0j     # Z4 = 6 + 0j milliohms (purely resistive)
    ]), unit=impedance_unit)
    
    result = jnp.conj(complex_impedances)  # type: ignore
    
    assert isinstance(result, Unitful)
    # Complex conjugates: conj(a + bj) = a - bj
    expected_vals = jnp.array([
        3.0 - 4.0j,    # conj(3 + 4j) = 3 - 4j
        -2.0 - 5.0j,   # conj(-2 + 5j) = -2 - 5j
        1.0 + 3.0j,    # conj(1 - 3j) = 1 + 3j
        6.0 - 0.0j     # conj(6 + 0j) = 6 - 0j
    ])
    # Scale factor: 10^(-3) for milliohms
    assert jnp.allclose(result.value(), expected_vals * 10**(-3))
    # Units should remain unchanged
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.A: -2, SI.s: -3}


def test_conj_jax_array():
    """Test complex conjugate with regular JAX array (non-Unitful)"""
    # Create a complex JAX array with mixed real and complex values
    complex_array = jnp.array([
        2.0 + 3.0j,
        -1.5 - 2.5j, 
        4.0 + 0.0j,
        0.0 - 7.0j,
        8.5 + 1.2j
    ])
    
    result = jnp.conj(complex_array)
    
    # Expected conjugates
    expected = jnp.array([
        2.0 - 3.0j,    # conj(2 + 3j) = 2 - 3j
        -1.5 + 2.5j,   # conj(-1.5 - 2.5j) = -1.5 + 2.5j
        4.0 - 0.0j,    # conj(4 + 0j) = 4 - 0j
        0.0 + 7.0j,    # conj(0 - 7j) = 0 + 7j  
        8.5 - 1.2j     # conj(8.5 + 1.2j) = 8.5 - 1.2j
    ])
    
    assert jnp.allclose(result, expected)
    # Should return a regular JAX array, not a Unitful object
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)


def test_dot_unitful_energy_calculation():
    """Test dot product with Unitful objects: force · displacement = work/energy"""
    # Create force vector: Newtons = kg*m/s^2 with scale=1 (factor of 10)
    force_unit = Unit(scale=1, dim={SI.kg: 1, SI.m: 1, SI.s: -2})
    force = Unitful(val=jnp.array([3.0, 4.0, 0.0]), unit=force_unit)
    
    # Create displacement vector: meters with scale=-2 (factor of 0.01, i.e., centimeters)
    displacement_unit = Unit(scale=-2, dim={SI.m: 1})
    displacement = Unitful(val=jnp.array([2.0, 1.5, 0.0]), unit=displacement_unit)
    
    result = jnp.dot(force, displacement)  # type: ignore
    
    assert isinstance(result, Unitful)
    # Dot product: [3, 4, 0] · [2, 1.5, 0] = 3*2 + 4*1.5 + 0*0 = 6 + 6 = 12
    # Scale factor: 10 * 0.01 = 0.1 (divide by 10 for scale)
    expected_value = 12.0 * 0.1
    assert jnp.allclose(result.value(), expected_value)
    # Dimensions: (kg*m/s^2) · m = kg*m^2/s^2 (energy/work units: Joules)
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.s: -2}


def test_dot_jax_array_matrix_vector():
    """Test dot product with regular JAX arrays (matrix-vector multiplication)"""
    # Create a 3x4 matrix
    matrix = jnp.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0], 
        [9.0, 10.0, 11.0, 12.0]
    ])
    
    # Create a 4-element vector
    vector = jnp.array([2.0, -1.0, 3.0, 0.5])
    
    result = jnp.dot(matrix, vector)
    
    # Expected result: matrix @ vector
    # Row 1: 1*2 + 2*(-1) + 3*3 + 4*0.5 = 2 - 2 + 9 + 2 = 11
    # Row 2: 5*2 + 6*(-1) + 7*3 + 8*0.5 = 10 - 6 + 21 + 4 = 29  
    # Row 3: 9*2 + 10*(-1) + 11*3 + 12*0.5 = 18 - 10 + 33 + 6 = 47
    expected = jnp.array([11.0, 29.0, 47.0])
    
    assert jnp.allclose(result, expected)
    # Should return a regular JAX array, not a Unitful object
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)
    # Result should be 1D with shape (3,)
    assert result.shape == (3,)
    
    
def test_transpose_unitful_stress_tensor():
    """Test transpose with Unitful object containing a stress tensor (2D matrix)"""
    # Create stress tensor unit: Pascals = kg/(m*s^2) with scale=6 (factor of 10^6, i.e., MPa)
    stress_unit = Unit(scale=6, dim={SI.kg: 1, SI.m: -1, SI.s: -2})
    # 3x3 stress tensor matrix
    stress_tensor = Unitful(val=jnp.array([
        [100.0, 50.0, 25.0],
        [50.0, 200.0, 75.0],
        [25.0, 75.0, 150.0]
    ]), unit=stress_unit)
    
    result = jnp.transpose(stress_tensor)  # type: ignore
    
    assert isinstance(result, Unitful)
    # Transpose of the matrix: rows become columns
    expected_vals = jnp.array([
        [100.0, 50.0, 25.0],
        [50.0, 200.0, 75.0], 
        [25.0, 75.0, 150.0]
    ])
    # Scale factor: 10^6 for MPa
    assert jnp.allclose(result.value(), expected_vals * 10**6)
    assert result.unit.dim == {SI.kg: 1, SI.m: -1, SI.s: -2}
    # Shape should be transposed: (3,3) -> (3,3) (symmetric in this case)
    assert result.shape == (3, 3)


def test_transpose_unitful_with_axes_parameter():
    """Test transpose with Unitful object using specific axes parameter on 3D array"""
    # Create velocity field unit: m/s with fractional dimensions and scale=-1 (factor of 0.1)
    velocity_unit = Unit(scale=-1, dim={SI.m: 1, SI.s: -1, SI.kg: Fraction(1, 4)})
    # 2x3x4 velocity field array
    velocity_field = Unitful(val=jnp.array([
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0]],
        [[13.0, 14.0, 15.0, 16.0],
         [17.0, 18.0, 19.0, 20.0],
         [21.0, 22.0, 23.0, 24.0]]
    ]), unit=velocity_unit)
    
    # Transpose with axes=(2, 0, 1): swap dimensions 0 and 2
    result = transpose(velocity_field, axes=(2, 0, 1))
    
    assert isinstance(result, Unitful)
    # Original shape: (2, 3, 4) -> New shape: (4, 2, 3) 
    assert result.shape == (4, 2, 3)
    # Check a few values to ensure correct transposition
    # Original [0, 1, 2] should be at new position [2, 0, 1]
    assert jnp.allclose(result.value()[2, 0, 1], 7.0 * 0.1)  # type: ignore
    # Original [1, 2, 3] should be at new position [3, 1, 2]
    assert jnp.allclose(result.value()[3, 1, 2], 24.0 * 0.1)  # type: ignore
    assert result.unit.dim == {SI.m: 1, SI.s: -1, SI.kg: Fraction(1, 4)}


def test_transpose_jax_array():
    """Test transpose with regular JAX array (non-Unitful)"""
    # Create a 4x3 matrix
    matrix = jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])
    
    result = transpose(matrix)
    
    # Expected transpose: 4x3 -> 3x4
    expected = jnp.array([
        [1.0, 4.0, 7.0, 10.0],
        [2.0, 5.0, 8.0, 11.0],
        [3.0, 6.0, 9.0, 12.0]
    ])
    
    assert jnp.allclose(result, expected)
    # Should return a regular JAX array, not a Unitful object
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)
    # Shape should be transposed: (4, 3) -> (3, 4)
    assert result.shape == (3, 4)
    

def test_pad_unitful_temperature_field():
    """Test pad with Unitful object containing a temperature field with fractional dimensions"""
    # Create temperature unit: Kelvin with fractional dimension and scale=-3 (millikelvin)
    temp_unit = Unit(scale=-3, dim={SI.K: 1, SI.m: Fraction(-1, 2)})  # K/sqrt(m)
    # 2x3 temperature field array
    temp_field = Unitful(val=jnp.array([
        [100.0, 200.0, 300.0],
        [400.0, 500.0, 600.0]
    ]), unit=temp_unit)
    
    # Pad with constant value and symmetric padding
    result: Unitful = jnp.pad(temp_field, pad_width=((1, 1), (2, 1)), mode='constant', constant_values=0.0)  # type: ignore
    
    assert isinstance(result, Unitful)
    # Expected shape: original (2, 3) + padding ((1,1), (2,1)) = (4, 6)
    assert result.shape == (4, 6)
    # Check padded values (should be 0 with proper scale factor)
    assert jnp.allclose(result.value()[0, :], 0.0)  # type: ignore First row should be zeros
    assert jnp.allclose(result.value()[-1, :], 0.0)  # type: ignore Last row should be zeros
    assert jnp.allclose(result.value()[:, :2], 0.0)  # type: ignore First two columns should be zeros
    assert jnp.allclose(result.value()[:, -1], 0.0)  # type: ignore Last column should be zeros
    # Check original values are preserved in the center with scale factor 10^(-3)
    assert jnp.allclose(result.value()[1, 2], 100.0 * 1e-3)  # type: ignore Original [0,0] -> [1,2]
    assert jnp.allclose(result.value()[2, 4], 600.0 * 1e-3)  # type: ignore Original [1,2] -> [2,4]
    # Units should be preserved
    assert result.unit.dim == {SI.K: 1, SI.m: Fraction(-1, 2)}


def test_pad_jax_array():
    """Test pad with regular JAX array (non-Unitful) using edge mode"""
    # Create a 3x2 matrix
    matrix = jnp.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    
    # Pad with edge values (replicate border values)
    result = jnp.pad(matrix, pad_width=((1, 2), (1, 1)), mode='edge')  # type: ignore
    
    # Expected result: 3x2 -> 6x4 with edge padding
    # Top padding: replicate first row
    # Bottom padding: replicate last row  
    # Left padding: replicate first column
    # Right padding: replicate last column
    expected = jnp.array([
        [1.0, 1.0, 2.0, 2.0],  # Top padding: replicate [1, 2]
        [1.0, 1.0, 2.0, 2.0],  # Original row 0 with side padding
        [3.0, 3.0, 4.0, 4.0],  # Original row 1 with side padding
        [5.0, 5.0, 6.0, 6.0],  # Original row 2 with side padding
        [5.0, 5.0, 6.0, 6.0],  # Bottom padding: replicate [5, 6]
        [5.0, 5.0, 6.0, 6.0]   # Bottom padding: replicate [5, 6]
    ])
    
    assert jnp.allclose(result, expected)
    # Should return a regular JAX array, not a Unitful object
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)
    # Shape should be padded: (3, 2) + ((1,2), (1,1)) = (6, 4)
    assert result.shape == (6, 4)
    
    
def test_stack_unitful_same_unit():
    """Test stack with multiple Unitful objects having identical units"""
    # Create force vectors with same unit: Newtons = kg*m/s^2 with scale=0
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})
    
    force1 = Unitful(val=jnp.array([10.0, 20.0, 30.0]), unit=force_unit)
    force2 = Unitful(val=jnp.array([40.0, 50.0, 60.0]), unit=force_unit)
    force3 = Unitful(val=jnp.array([70.0, 80.0, 90.0]), unit=force_unit)
    
    # Stack along axis 0 (default)
    result = jnp.stack([force1, force2, force3])  # type: ignore
    
    assert isinstance(result, Unitful)
    # Expected shape: 3 arrays of shape (3,) stacked -> (3, 3)
    assert result.shape == (3, 3)
    # Check values are correctly stacked
    expected_vals = jnp.array([
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
        [70.0, 80.0, 90.0]
    ])
    assert jnp.allclose(result.value(), expected_vals)
    # Units should be preserved
    assert result.unit.dim == {SI.kg: 1, SI.m: 1, SI.s: -2}


def test_stack_unitful_same_dimension_different_scale():
    """Test stack with Unitful objects having same dimensions but different scales"""
    # Create pressure values with same dimensions but different scales
    pressure_dim: dict[SI, int | Fraction] = {SI.kg: 1, SI.m: -1, SI.s: -2}  # Pascal dimensions
    
    # Pressure in Pascals (scale=0)
    pressure_pa = Unitful(val=jnp.array([1000.0, 2000.0]), unit=Unit(scale=0, dim=pressure_dim))
    # Pressure in kilopascals (scale=3)
    pressure_kpa = Unitful(val=jnp.array([5.0, 8.0]), unit=Unit(scale=3, dim=pressure_dim))
    # Pressure in megapascals (scale=6)
    pressure_mpa = Unitful(val=jnp.array([0.003, 0.007]), unit=Unit(scale=6, dim=pressure_dim))
    
    # Stack along axis 1
    result = jnp.stack([pressure_pa, pressure_kpa, pressure_mpa], axis=1)  # type: ignore
    
    assert isinstance(result, Unitful)
    # Expected shape: 3 arrays of shape (2,) stacked along axis 1 -> (2, 3)
    assert result.shape == (2, 3)
    # All values should be converted to the same scale
    # Expected: scale normalization should bring all to a common scale
    # pressure_pa: [1000, 2000] Pa (scale=0)
    # pressure_kpa: [5000, 8000] Pa (scale=0 equivalent)
    # pressure_mpa: [3000, 7000] Pa (scale=0 equivalent)
    expected_vals = jnp.array([
        [1000.0, 5000.0, 3000.0],  # First element from each array
        [2000.0, 8000.0, 7000.0]   # Second element from each array
    ])
    assert jnp.allclose(result.value(), expected_vals)
    assert result.unit.dim == pressure_dim


def test_stack_unitful_different_units_should_raise():
    """Test stack with Unitful objects having different dimensions should raise exception"""
    # Create objects with incompatible dimensions
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})  # Newton
    mass_unit = Unit(scale=0, dim={SI.kg: 1})  # kilogram
    
    force = Unitful(val=jnp.array([10.0, 20.0]), unit=force_unit)
    mass = Unitful(val=jnp.array([5.0, 8.0]), unit=mass_unit)
    
    # Should raise an exception due to incompatible dimensions
    with pytest.raises(Exception):
        jnp.stack([force, mass])  # type: ignore


def test_stack_jax_array():
    """Test stack with regular JAX arrays (non-Unitful)"""
    # Create regular JAX arrays
    array1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    array2 = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    array3 = jnp.array([[9.0, 10.0], [11.0, 12.0]])
    
    # Stack along axis 0 (default)
    result = jnp.stack([array1, array2, array3])  # type: ignore
    
    # Expected result: 3 arrays of shape (2, 2) stacked -> (3, 2, 2)
    expected = jnp.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]]
    ])
    
    assert jnp.allclose(result, expected)
    # Should return a regular JAX array, not a Unitful object
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)
    # Shape should be (3, 2, 2)
    assert result.shape == (3, 2, 2)
    

def test_isfinite_unitful_with_special_values():
    """Test isfinite with Unitful object containing finite, infinite, and NaN values"""
    # Create energy unit: Joules = kg*m^2/s^2 with scale=3 (kilojoules)
    energy_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: 2, SI.s: -2})
    # Array with mix of finite, infinite, and NaN values
    energies = Unitful(val=jnp.array([
        10.5,           # finite
        jnp.inf,        # positive infinity
        -25.0,          # finite negative
        jnp.nan,        # NaN
        -jnp.inf,       # negative infinity
        0.0,            # finite zero
        1e-10           # finite small value
    ]), unit=energy_unit)
    
    result = jnp.isfinite(energies)  # type: ignore
    
    # Expected: only finite values should return True
    expected = jnp.array([
        True,   # 10.5 is finite
        False,  # inf is not finite
        True,   # -25.0 is finite
        False,  # nan is not finite
        False,  # -inf is not finite
        True,   # 0.0 is finite
        True    # 1e-10 is finite
    ])
    
    assert jnp.array_equal(result, expected)
    # Should return a regular JAX array of booleans, not a Unitful object
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)
    # Result should have boolean dtype
    assert result.dtype == jnp.bool_
    # Shape should match input
    assert result.shape == energies.shape


def test_isfinite_jax_array():
    """Test isfinite with regular JAX array (non-Unitful) containing special values"""
    # Create array with various finite and non-finite values
    array = jnp.array([
        [1.5, jnp.inf, -3.7],
        [jnp.nan, 0.0, -jnp.inf],
        [42.0, 1e20, -1e-15]
    ])
    
    result = jnp.isfinite(array)  # type: ignore
    
    # Expected: finite values return True, inf and nan return False
    expected = jnp.array([
        [True, False, True],    # 1.5 finite, inf not finite, -3.7 finite
        [False, True, False],   # nan not finite, 0.0 finite, -inf not finite
        [True, True, True]      # all finite (including very large and very small)
    ])
    
    assert jnp.array_equal(result, expected)
    # Should return a regular JAX array, not a Unitful object
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)
    # Result should have boolean dtype
    assert result.dtype == jnp.bool_
    # Shape should match input: (3, 3)
    assert result.shape == (3, 3)
    