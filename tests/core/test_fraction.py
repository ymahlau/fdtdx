import unittest
import math

from fdtdx.core.fraction import Fraction


class TestFraction(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.f1 = Fraction(1, 2)  # 1/2
        self.f2 = Fraction(1, 3)  # 1/3
        self.f3 = Fraction(2, 4)  # 2/4 = 1/2
        self.zero = Fraction(0, 1)
        self.one = Fraction(1, 1)
        self.negative = Fraction(-1, 2)  # -1/2
    
    def test_string_representation(self):
        """Test __str__ and __repr__ methods."""
        self.assertEqual(str(self.f1), "1 / 2")
        self.assertEqual(str(self.f2), "1 / 3")
        self.assertEqual(str(self.zero), "0")
        self.assertEqual(str(self.one), "1")
        self.assertEqual(str(Fraction(5, 1)), "5")
        self.assertEqual(repr(self.f1), "1 / 2")
    
    def test_reduction(self):
        """Test fraction reduction."""
        self.assertEqual(self.f3.reduced(), Fraction(1, 2))
        self.assertEqual(Fraction(6, 9).reduced(), Fraction(2, 3))
        self.assertEqual(Fraction(0, 5).reduced(), Fraction(0, 1))
        self.assertEqual(Fraction(-4, 8).reduced(), Fraction(-1, 2))
        self.assertEqual(Fraction(4, -8).reduced(), Fraction(-1, 2))  # Negative denominator
        self.assertEqual(Fraction(-4, -8).reduced(), Fraction(1, 2))  # Double negative
    
    def test_value(self):
        """Test conversion to float."""
        self.assertAlmostEqual(self.f1.value(), 0.5)
        self.assertAlmostEqual(self.f2.value(), 1/3)
        self.assertEqual(self.zero.value(), 0.0)
        self.assertEqual(self.one.value(), 1.0)
    
    def test_equality(self):
        """Test equality comparisons."""
        # Fraction equality
        self.assertTrue(self.f1 == self.f3)
        self.assertFalse(self.f1 == self.f2)
        
        # Integer equality
        self.assertTrue(self.one == 1)
        self.assertFalse(self.f1 == 1)
        self.assertTrue(self.zero == 0)
        
        # Float equality
        self.assertTrue(self.f1 == 0.5)
        self.assertFalse(self.f1 == 0.33)
        
        # Non-numeric types
        self.assertFalse(self.f1 == "1/2")
        self.assertFalse(self.f1 == [1, 2])
    
    def test_inequality(self):
        """Test inequality operator."""
        self.assertTrue(self.f1 != self.f2)
        self.assertFalse(self.f1 != self.f3)
        self.assertTrue(self.f1 != 1)
        self.assertTrue(self.f1 != 0.33)
    
    def test_addition(self):
        """Test addition operations."""
        # Fraction + Fraction
        result = self.f1 + self.f2
        expected = Fraction(5, 6)  # 1/2 + 1/3 = 3/6 + 2/6 = 5/6
        self.assertEqual(result, expected)
        
        # Fraction + int
        result = self.f1 + 2
        expected = Fraction(5, 2)  # 1/2 + 2 = 1/2 + 4/2 = 5/2
        self.assertEqual(result, expected)
        
        # int + Fraction (reverse addition)
        result = 2 + self.f1
        expected = Fraction(5, 2)
        self.assertEqual(result, expected)
        
        # Fraction + float
        result = self.f1 + 0.5
        self.assertAlmostEqual(result, 1.0)
        
        # float + Fraction
        result = 0.5 + self.f1
        self.assertAlmostEqual(result, 1.0)
    
    def test_subtraction(self):
        """Test subtraction operations."""
        # Fraction - Fraction
        result = self.f1 - self.f2
        expected = Fraction(1, 6)  # 1/2 - 1/3 = 3/6 - 2/6 = 1/6
        self.assertEqual(result, expected)
        
        # Fraction - int
        result = self.f1 - 1
        expected = Fraction(-1, 2)  # 1/2 - 1 = 1/2 - 2/2 = -1/2
        self.assertEqual(result, expected)
        
        # int - Fraction (reverse subtraction)
        result = 1 - self.f1
        expected = Fraction(1, 2)  # 1 - 1/2 = 2/2 - 1/2 = 1/2
        self.assertEqual(result, expected)
        
        # Fraction - float
        result = self.f1 - 0.25
        self.assertAlmostEqual(result, 0.25)
        
        # float - Fraction
        result = 1.0 - self.f1
        self.assertAlmostEqual(result, 0.5)
    
    def test_multiplication(self):
        """Test multiplication operations."""
        # Fraction * Fraction
        result = self.f1 * self.f2
        expected = Fraction(1, 6)  # 1/2 * 1/3 = 1/6
        self.assertEqual(result, expected)
        
        # Fraction * int
        result = self.f1 * 3
        expected = Fraction(3, 2)  # 1/2 * 3 = 3/2
        self.assertEqual(result, expected)
        
        # int * Fraction (reverse multiplication)
        result = 3 * self.f1
        expected = Fraction(3, 2)
        self.assertEqual(result, expected)
        
        # Fraction * float
        result = self.f1 * 2.0
        self.assertAlmostEqual(result, 1.0)
        
        # float * Fraction
        result = 2.0 * self.f1
        self.assertAlmostEqual(result, 1.0)
    
    def test_division(self):
        """Test division operations."""
        # Fraction / Fraction
        result = self.f1 / self.f2
        expected = Fraction(3, 2)  # (1/2) / (1/3) = (1/2) * (3/1) = 3/2
        self.assertEqual(result, expected)
        
        # Fraction / int
        result = self.f1 / 2
        expected = Fraction(1, 4)  # (1/2) / 2 = 1/4
        self.assertEqual(result, expected)
        
        # int / Fraction (reverse division)
        result = 2 / self.f1
        expected = Fraction(4, 1)  # 2 / (1/2) = 2 * 2 = 4
        self.assertEqual(result, expected)
        
        # Fraction / float
        result = self.f1 / 0.5
        self.assertAlmostEqual(result, 1.0)
        
        # float / Fraction
        result = 1.0 / self.f1
        self.assertAlmostEqual(result, 2.0)
    
    def test_division_by_zero(self):
        """Test division by zero raises appropriate exceptions."""
        with self.assertRaises(ZeroDivisionError):
            self.f1 / self.zero
        
        with self.assertRaises(ZeroDivisionError):
            self.f1 / 0
        
        with self.assertRaises(ZeroDivisionError):
            self.f1 / 0.0
        
        with self.assertRaises(ZeroDivisionError):
            2 / self.zero
        
        with self.assertRaises(ZeroDivisionError):
            2.0 / self.zero
    
    def test_comparison_operators(self):
        """Test comparison operations."""
        # Less than
        self.assertFalse(self.f1 < self.f2)  # 1/2 < 1/3 is False
        self.assertTrue(self.f2 < self.f1)   # 1/3 < 1/2 is True
        self.assertTrue(self.f1 < 1)         # 1/2 < 1 is True
        self.assertFalse(self.f1 < 0.25)     # 1/2 < 0.25 is False
        
        # Less than or equal
        self.assertTrue(self.f1 <= self.f3)  # 1/2 <= 1/2 is True
        self.assertTrue(self.f2 <= self.f1)  # 1/3 <= 1/2 is True
        self.assertFalse(self.f1 <= self.f2) # 1/2 <= 1/3 is False
        
        # Greater than
        self.assertTrue(self.f1 > self.f2)   # 1/2 > 1/3 is True
        self.assertFalse(self.f2 > self.f1)  # 1/3 > 1/2 is False
        self.assertFalse(self.f1 > 1)        # 1/2 > 1 is False
        self.assertTrue(self.f1 > 0.25)      # 1/2 > 0.25 is True
        
        # Greater than or equal
        self.assertTrue(self.f1 >= self.f3)  # 1/2 >= 1/2 is True
        self.assertTrue(self.f1 >= self.f2)  # 1/2 >= 1/3 is True
        self.assertFalse(self.f2 >= self.f1) # 1/3 >= 1/2 is False
    
    def test_power_operations(self):
        """Test exponentiation operations."""
        # Positive integer exponents
        result = self.f1 ** 2
        expected = Fraction(1, 4)  # (1/2)^2 = 1/4
        self.assertEqual(result, expected)
        
        result = self.f2 ** 3
        expected = Fraction(1, 27)  # (1/3)^3 = 1/27
        self.assertEqual(result, expected)
        
        # Zero exponent
        result = self.f1 ** 0
        expected = Fraction(1, 1)  # Any non-zero number to power 0 is 1
        self.assertEqual(result, expected)
        
        result = self.f2 ** 0
        expected = Fraction(1, 1)
        self.assertEqual(result, expected)
        
        # Negative exponents
        result = self.f1 ** -2
        expected = Fraction(4, 1)  # (1/2)^(-2) = (2/1)^2 = 4/1
        self.assertEqual(result, expected)
        
        result = self.f2 ** -1
        expected = Fraction(3, 1)  # (1/3)^(-1) = 3/1
        self.assertEqual(result, expected)
    
    def test_power_edge_cases(self):
        """Test edge cases for power operations."""
        # 0^0 should raise ZeroDivisionError
        with self.assertRaises(ZeroDivisionError):
            self.zero ** 0
        
        # 0^(negative) should raise ZeroDivisionError
        with self.assertRaises(ZeroDivisionError):
            self.zero ** -1
        
        # Non-integer exponents should return NotImplemented
        result = self.f1.__pow__(2.5)
        self.assertEqual(result, NotImplemented)
