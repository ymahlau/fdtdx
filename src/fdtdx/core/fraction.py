import math
from typing import Any, NamedTuple, Union, overload

class Fraction(NamedTuple):
    num: int
    denom: int
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        if self.num == 0:
            return "0"
        if self.denom == 1:
            return str(self.num)
        return f"({self.num}/{self.denom})"
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, int | float | Fraction):
            return False
        if isinstance(other, Fraction):
            self_red = self.reduced()
            other_red = other.reduced()
            return self_red.num == other_red.num and self_red.denom == other_red.denom
        if isinstance(other, int):
            self_red = self.reduced()
            if self_red.denom != 1:
                return False
            return self_red.num == other
        if isinstance(other, float):
            # Compare float values (be careful with floating point precision)
            return abs(self.value() - other) < 1e-15
    
    def value(self) -> float:
        return self.num / self.denom
        
    def reduced(self) -> "Fraction":
        if self.num == 0:
            return Fraction(0, 1)
        
        common_divisor = math.gcd(abs(self.num), abs(self.denom))
        reduced_num = self.num // common_divisor
        reduced_denom = self.denom // common_divisor
        
        # Ensure denominator is positive
        if reduced_denom < 0:
            reduced_num = -reduced_num
            reduced_denom = -reduced_denom
        
        return Fraction(reduced_num, reduced_denom)
    
    # Addition
    @overload
    def __add__(self, other: "Fraction") -> "Fraction": ...
    
    @overload
    def __add__(self, other: int) -> "Fraction": ...  # type: ignore
    
    @overload
    def __add__(self, other: float) -> float: ...
    
    def __add__(self, other: Any) -> Union["Fraction", float]:  # type: ignore
        if isinstance(other, Fraction):
            new_num = self.num * other.denom + other.num * self.denom
            new_denom = self.denom * other.denom
            return Fraction(new_num, new_denom).reduced()
        elif isinstance(other, int):
            return self + Fraction(other, 1)
        elif isinstance(other, float):
            return self.value() + other
        return NotImplemented
    
    @overload
    def __radd__(self, other: "Fraction") -> "Fraction": ...
    
    @overload
    def __radd__(self, other: int) -> "Fraction": ...  # type: ignore
    
    @overload
    def __radd__(self, other: float) -> float: ...
    
    def __radd__(self, other: Any) -> Union["Fraction", float]:
        return self.__add__(other)
    
    # Multiplication
    @overload
    def __mul__(self, other: "Fraction") -> "Fraction": ...
    
    @overload
    def __mul__(self, other: int) -> "Fraction": ...  # type: ignore
    
    @overload
    def __mul__(self, other: float) -> float: ...
    
    def __mul__(self, other: Any) -> Union["Fraction", float]:  # type: ignore
        if isinstance(other, Fraction):
            new_num = self.num * other.num
            new_denom = self.denom * other.denom
            return Fraction(new_num, new_denom).reduced()
        elif isinstance(other, int):
            return Fraction(self.num * other, self.denom).reduced()
        elif isinstance(other, float):
            return self.value() * other
        return NotImplemented
    
    @overload
    def __rmul__(self, other: "Fraction") -> "Fraction": ...
    
    @overload
    def __rmul__(self, other: int) -> "Fraction": ...  # type: ignore
    
    @overload
    def __rmul__(self, other: float) -> float: ...
    
    def __rmul__(self, other: Any) -> Union["Fraction", float]:  # type: ignore
        return self.__mul__(other)
    
    # Division
    @overload
    def __truediv__(self, other: "Fraction") -> "Fraction": ...
    
    @overload
    def __truediv__(self, other: int) -> "Fraction": ...  # type: ignore
    
    @overload
    def __truediv__(self, other: float) -> float: ...
    
    def __truediv__(self, other: Any) -> Union["Fraction", float]:
        if isinstance(other, Fraction):
            if other.num == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            new_num = self.num * other.denom
            new_denom = self.denom * other.num
            return Fraction(new_num, new_denom).reduced()
        elif isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return Fraction(self.num, self.denom * other).reduced()
        elif isinstance(other, float):
            if other == 0.0:
                raise ZeroDivisionError("Cannot divide by zero")
            return self.value() / other
        return NotImplemented
    
    @overload
    def __rtruediv__(self, other: "Fraction") -> "Fraction": ...
    
    @overload
    def __rtruediv__(self, other: int) -> "Fraction": ...  # type: ignore
    
    @overload
    def __rtruediv__(self, other: float) -> float: ...
    
    def __rtruediv__(self, other: Any) -> Union["Fraction", float]:
        if isinstance(other, Fraction):
            return other.__truediv__(self)
        elif isinstance(other, int):
            if self.num == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return Fraction(other * self.denom, self.num).reduced()
        elif isinstance(other, float):
            if self.num == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return other / self.value()
        return NotImplemented
    
    # Subtraction
    @overload
    def __sub__(self, other: "Fraction") -> "Fraction": ...
    
    @overload
    def __sub__(self, other: int) -> "Fraction": ...  # type: ignore
    
    @overload
    def __sub__(self, other: float) -> float: ...
    
    def __sub__(self, other: Any) -> Union["Fraction", float]:
        if isinstance(other, Fraction):
            new_num = self.num * other.denom - other.num * self.denom
            new_denom = self.denom * other.denom
            return Fraction(new_num, new_denom).reduced()
        elif isinstance(other, int):
            return self - Fraction(other, 1)
        elif isinstance(other, float):
            return self.value() - other
        return NotImplemented
    
    @overload
    def __rsub__(self, other: "Fraction") -> "Fraction": ...
    
    @overload
    def __rsub__(self, other: int) -> "Fraction": ...  # type: ignore
    
    @overload
    def __rsub__(self, other: float) -> float: ...
    
    def __rsub__(self, other: Any) -> Union["Fraction", float]:
        if isinstance(other, Fraction):
            return other.__sub__(self)
        elif isinstance(other, int):
            return Fraction(other, 1) - self
        elif isinstance(other, float):
            return other - self.value()
        return NotImplemented
    
    # Comparison operators
    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Fraction):
            # Cross multiply to avoid division
            return self.num * other.denom < other.num * self.denom
        elif isinstance(other, int):
            return self.num < other * self.denom
        elif isinstance(other, float):
            return self.value() < other
        return NotImplemented
    
    def __le__(self, other: Any) -> bool:
        return self < other or self == other
    
    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)
    
    def __gt__(self, other: Any) -> bool:
        if isinstance(other, Fraction):
            return self.num * other.denom > other.num * self.denom
        elif isinstance(other, int):
            return self.num > other * self.denom
        elif isinstance(other, float):
            return self.value() > other
        return NotImplemented
    
    def __ge__(self, other: Any) -> bool:
        return self > other or self == other
    
    # Integer power
    def __pow__(self, exponent: int) -> "Fraction":
        if not isinstance(exponent, int):
            return NotImplemented
        
        if exponent == 0:
            if self.num == 0:
                raise ZeroDivisionError("0^0 is undefined")
            return Fraction(1, 1)
        elif exponent > 0:
            new_num = self.num ** exponent
            new_denom = self.denom ** exponent
            return Fraction(new_num, new_denom).reduced()
        else:  # exponent < 0
            if self.num == 0:
                raise ZeroDivisionError("Cannot raise zero to negative power")
            # For negative exponents, flip the fraction and use positive exponent
            new_num = self.denom ** (-exponent)
            new_denom = self.num ** (-exponent)
            return Fraction(new_num, new_denom).reduced()
