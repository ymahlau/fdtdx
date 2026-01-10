"""Color module for fdtdx framework.

This module provides a Color class for representing and converting between different
color formats, along with a comprehensive collection of named colors from the XKCD
color survey. The Color class is a subclass of fdtdx.TreeClass and supports:

- Multiple input formats: RGB tuples (normalized or 8-bit), hex strings, named colors
- Multiple output formats: normalized RGB, 8-bit RGB, hex, matplotlib-compatible
- Type checking and validation
- Integration with the fdtdx framework

The module includes all XKCD colors as predefined constants for consistent visualization.

Example:
    >>> from fdtdx import colors
    >>> # Create colors from different formats
    >>> c1 = colors.Color.from_rgb(255, 0, 0)  # 8-bit RGB
    >>> c2 = colors.Color.from_hex("#00FF00")
    >>> c3 = colors.XKCD_MINT_GREEN
    >>>
    >>> # Convert to different formats
    >>> c1.to_hex()  # "#FF0000"
    >>> c1.to_rgb_normalized()  # (1.0, 0.0, 0.0)
    >>> c1.to_rgb_255()  # (255, 0, 0)
"""

from fdtdx.core.jax.pytrees import TreeClass, autoinit


@autoinit
class Color(TreeClass):
    """Color representation with multiple format support.

    This class represents a color and provides methods to convert between
    different color formats. Internally, colors are stored as normalized
    RGB values in the range [0, 1].

    Attributes:
        r (float): Red component, normalized to [0, 1]
        g (float): Green component, normalized to [0, 1]
        b (float): Blue component, normalized to [0, 1]
    """

    r: float
    g: float
    b: float

    def __post_init__(self):
        """Validate RGB components are in valid range."""
        if not (0 <= self.r <= 1 and 0 <= self.g <= 1 and 0 <= self.b <= 1):
            raise ValueError(f"RGB components must be in range [0, 1], got ({self.r}, {self.g}, {self.b})")

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int) -> "Color":
        """Create a Color from 8-bit RGB values (0-255).

        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)

        Returns:
            Color instance
        """
        return cls(r=r / 255.0, g=g / 255.0, b=b / 255.0)

    @classmethod
    def from_hex(cls, hex_string: str) -> "Color":
        """Create a Color from a hexadecimal color string.

        Args:
            hex_string: Hex color string (e.g., "#FF0000", "FF0000", "#F00")

        Returns:
            Color instance

        Raises:
            ValueError: If hex string is invalid
        """
        hex_string = hex_string.lstrip("#")

        # Handle 3-character hex codes
        if len(hex_string) == 3:
            hex_string = "".join([c * 2 for c in hex_string])

        if len(hex_string) != 6:
            raise ValueError(f"Invalid hex color: {hex_string}")

        try:
            r = int(hex_string[0:2], 16)
            g = int(hex_string[2:4], 16)
            b = int(hex_string[4:6], 16)
            return cls.from_rgb(r, g, b)
        except ValueError:
            raise ValueError(f"Invalid hex color: {hex_string}")

    def to_rgb_normalized(self) -> tuple[float, float, float]:
        """Return color as normalized RGB tuple [0, 1].

        Returns:
            Tuple of (r, g, b) with values in [0, 1]
        """
        return self.r, self.g, self.b

    def to_rgb_255(self) -> tuple[int, int, int]:
        """Return color as 8-bit RGB tuple (0-255).

        Returns:
            Tuple of (r, g, b) with integer values in [0, 255]
        """
        return int(round(self.r * 255)), int(round(self.g * 255)), int(round(self.b * 255))

    def to_hex(self) -> str:
        """Return color as hexadecimal string.

        Returns:
            Hex color string with leading # (e.g., "#FF0000")
        """
        r, g, b = self.to_rgb_255()
        return f"#{r:02X}{g:02X}{b:02X}"

    def to_mpl(self) -> tuple[float, float, float]:
        """Return color in matplotlib-compatible format.

        This is an alias for to_rgb_normalized() for clarity when using
        with matplotlib.

        Returns:
            Tuple of (r, g, b) with values in [0, 1]
        """
        return self.to_rgb_normalized()

    def __str__(self) -> str:
        """Return hex string representation of the Color."""
        return self.to_hex()


# ============================================================================
# XKCD Color Survey Colors
# ============================================================================
# The following colors are from the XKCD color survey:
# https://xkcd.com/color/rgb.txt
#
# These colors represent the results of asking 200,000+ participants to name
# random colors. They provide a comprehensive, crowdsourced palette with
# intuitive names.
# ============================================================================

# Bright and Vibrant Colors
XKCD_CLOUDY_BLUE = Color.from_rgb(172, 194, 217)
XKCD_DARK_PASTEL_GREEN = Color.from_rgb(86, 174, 87)
XKCD_DUST = Color.from_rgb(178, 153, 110)
XKCD_ELECTRIC_LIME = Color.from_rgb(168, 255, 4)
XKCD_FRESH_GREEN = Color.from_rgb(105, 216, 79)
XKCD_LIGHT_EGGPLANT = Color.from_rgb(137, 69, 133)
XKCD_NASTY_GREEN = Color.from_rgb(112, 178, 63)
XKCD_REALLY_LIGHT_BLUE = Color.from_rgb(212, 255, 255)
XKCD_TEA = Color.from_rgb(101, 171, 124)
XKCD_WARM_PURPLE = Color.from_rgb(149, 46, 143)
XKCD_YELLOWISH_TAN = Color.from_rgb(252, 252, 129)
XKCD_CEMENT = Color.from_rgb(165, 163, 145)
XKCD_DARK_GRASS_GREEN = Color.from_rgb(56, 128, 4)
XKCD_DUSTY_TEAL = Color.from_rgb(76, 144, 133)
XKCD_GREY_TEAL = Color.from_rgb(94, 155, 138)
XKCD_MACARONI_AND_CHEESE = Color.from_rgb(239, 180, 53)
XKCD_PINKISH_TAN = Color.from_rgb(217, 155, 130)
XKCD_SPRUCE = Color.from_rgb(10, 95, 56)
XKCD_STRONG_BLUE = Color.from_rgb(12, 6, 247)
XKCD_TOXIC_GREEN = Color.from_rgb(97, 222, 42)
XKCD_WINDOWS_BLUE = Color.from_rgb(55, 120, 191)
XKCD_BLUE_BLUE = Color.from_rgb(34, 66, 199)
XKCD_BLUE_WITH_A_HINT_OF_PURPLE = Color.from_rgb(83, 60, 198)
XKCD_BOOGER = Color.from_rgb(155, 181, 60)
XKCD_BRIGHT_SEA_GREEN = Color.from_rgb(5, 255, 166)
XKCD_DARK_GREEN_BLUE = Color.from_rgb(31, 99, 87)
XKCD_DEEP_TURQUOISE = Color.from_rgb(1, 115, 116)
XKCD_GREEN_TEAL = Color.from_rgb(12, 181, 119)
XKCD_STRONG_PINK = Color.from_rgb(255, 7, 137)
XKCD_BLAND = Color.from_rgb(175, 168, 139)
XKCD_DEEP_AQUA = Color.from_rgb(8, 120, 127)
XKCD_LAVENDER_PINK = Color.from_rgb(221, 133, 215)
XKCD_LIGHT_MOSS_GREEN = Color.from_rgb(166, 200, 117)
XKCD_LIGHT_SEAFOAM_GREEN = Color.from_rgb(167, 255, 181)
XKCD_OLIVE_YELLOW = Color.from_rgb(194, 183, 9)
XKCD_PIG_PINK = Color.from_rgb(231, 142, 165)
XKCD_DEEP_LILAC = Color.from_rgb(150, 110, 189)
XKCD_DESERT = Color.from_rgb(204, 173, 96)
XKCD_DUSTY_LAVENDER = Color.from_rgb(172, 134, 168)
XKCD_PURPLY_GREY = Color.from_rgb(148, 126, 148)
XKCD_PURPLY = Color.from_rgb(152, 63, 178)
XKCD_CANDY_PINK = Color.from_rgb(255, 99, 233)
XKCD_LIGHT_PASTEL_GREEN = Color.from_rgb(178, 251, 165)
XKCD_BORING_GREEN = Color.from_rgb(99, 179, 101)
XKCD_KIWI_GREEN = Color.from_rgb(142, 229, 63)
XKCD_LIGHT_GREY_GREEN = Color.from_rgb(183, 225, 161)
XKCD_ORANGE_PINK = Color.from_rgb(255, 111, 82)
XKCD_TEA_GREEN = Color.from_rgb(189, 248, 163)
XKCD_VERY_LIGHT_BROWN = Color.from_rgb(211, 182, 131)
XKCD_EGG_SHELL = Color.from_rgb(255, 252, 196)
XKCD_EGGPLANT_PURPLE = Color.from_rgb(67, 5, 65)

# Primary and Secondary Colors
XKCD_RED = Color.from_rgb(229, 0, 0)
XKCD_ORANGE = Color.from_rgb(249, 115, 6)
XKCD_YELLOW = Color.from_rgb(255, 255, 20)
XKCD_GREEN = Color.from_rgb(21, 176, 26)
XKCD_BLUE = Color.from_rgb(3, 67, 223)
XKCD_PURPLE = Color.from_rgb(126, 30, 156)
XKCD_PINK = Color.from_rgb(255, 129, 192)
XKCD_BROWN = Color.from_rgb(101, 55, 0)
XKCD_GREY = Color.from_rgb(146, 149, 145)
XKCD_WHITE = Color.from_rgb(255, 255, 255)
XKCD_BLACK = Color.from_rgb(0, 0, 0)

# Shades of Red
XKCD_DARK_RED = Color.from_rgb(132, 0, 0)
XKCD_LIGHT_RED = Color.from_rgb(255, 71, 76)
XKCD_BRIGHT_RED = Color.from_rgb(255, 0, 13)
XKCD_DEEP_RED = Color.from_rgb(154, 2, 0)
XKCD_BLOOD_RED = Color.from_rgb(152, 0, 2)
XKCD_CRIMSON = Color.from_rgb(140, 0, 15)
XKCD_SCARLET = Color.from_rgb(190, 1, 25)
XKCD_CHERRY_RED = Color.from_rgb(247, 2, 42)
XKCD_WINE_RED = Color.from_rgb(123, 3, 35)
XKCD_BRICK_RED = Color.from_rgb(143, 20, 2)

# Shades of Orange
XKCD_DARK_ORANGE = Color.from_rgb(198, 81, 2)
XKCD_LIGHT_ORANGE = Color.from_rgb(253, 170, 72)
XKCD_BRIGHT_ORANGE = Color.from_rgb(255, 91, 0)
XKCD_BURNT_ORANGE = Color.from_rgb(192, 78, 1)
XKCD_RUST_ORANGE = Color.from_rgb(196, 85, 8)
XKCD_PUMPKIN_ORANGE = Color.from_rgb(251, 125, 7)
XKCD_TANGERINE = Color.from_rgb(255, 148, 8)

# Shades of Yellow
XKCD_DARK_YELLOW = Color.from_rgb(213, 182, 10)
XKCD_LIGHT_YELLOW = Color.from_rgb(255, 254, 122)
XKCD_BRIGHT_YELLOW = Color.from_rgb(255, 253, 1)
XKCD_GOLDEN_YELLOW = Color.from_rgb(254, 193, 7)
XKCD_MUSTARD_YELLOW = Color.from_rgb(210, 189, 10)
XKCD_LEMON_YELLOW = Color.from_rgb(253, 255, 82)
XKCD_SUNSHINE_YELLOW = Color.from_rgb(255, 253, 55)

# Shades of Green
XKCD_DARK_GREEN = Color.from_rgb(3, 53, 0)
XKCD_LIGHT_GREEN = Color.from_rgb(150, 249, 123)
XKCD_BRIGHT_GREEN = Color.from_rgb(1, 255, 7)
XKCD_FOREST_GREEN = Color.from_rgb(6, 71, 12)
XKCD_LIME_GREEN = Color.from_rgb(137, 254, 5)
XKCD_EMERALD_GREEN = Color.from_rgb(2, 143, 30)
XKCD_KELLY_GREEN = Color.from_rgb(2, 171, 46)
XKCD_GRASS_GREEN = Color.from_rgb(63, 155, 11)
XKCD_OLIVE_GREEN = Color.from_rgb(103, 122, 4)
XKCD_MINT_GREEN = Color.from_rgb(143, 255, 159)
XKCD_SEA_GREEN = Color.from_rgb(83, 252, 161)
XKCD_SAGE_GREEN = Color.from_rgb(136, 179, 120)
XKCD_HUNTER_GREEN = Color.from_rgb(11, 64, 8)

# Shades of Blue
XKCD_DARK_BLUE = Color.from_rgb(0, 3, 91)
XKCD_LIGHT_BLUE = Color.from_rgb(149, 208, 252)
XKCD_BRIGHT_BLUE = Color.from_rgb(1, 101, 252)
XKCD_NAVY_BLUE = Color.from_rgb(0, 17, 70)
XKCD_SKY_BLUE = Color.from_rgb(117, 187, 253)
XKCD_ROYAL_BLUE = Color.from_rgb(5, 4, 170)
XKCD_COBALT_BLUE = Color.from_rgb(3, 10, 167)
XKCD_AZURE = Color.from_rgb(6, 154, 243)
XKCD_ELECTRIC_BLUE = Color.from_rgb(6, 82, 255)
XKCD_TURQUOISE = Color.from_rgb(6, 194, 172)
XKCD_CYAN = Color.from_rgb(0, 255, 255)
XKCD_TEAL = Color.from_rgb(2, 147, 134)
XKCD_AQUA = Color.from_rgb(19, 234, 201)

# Shades of Purple
XKCD_DARK_PURPLE = Color.from_rgb(53, 6, 62)
XKCD_LIGHT_PURPLE = Color.from_rgb(191, 119, 246)
XKCD_BRIGHT_PURPLE = Color.from_rgb(190, 3, 253)
XKCD_DEEP_PURPLE = Color.from_rgb(54, 1, 63)
XKCD_LAVENDER = Color.from_rgb(199, 159, 239)
XKCD_VIOLET = Color.from_rgb(154, 14, 234)
XKCD_MAGENTA = Color.from_rgb(194, 0, 120)
XKCD_PLUM = Color.from_rgb(88, 15, 65)
XKCD_INDIGO = Color.from_rgb(56, 2, 130)
XKCD_MAUVE = Color.from_rgb(174, 113, 129)

# Shades of Pink
XKCD_DARK_PINK = Color.from_rgb(203, 65, 107)
XKCD_LIGHT_PINK = Color.from_rgb(255, 209, 223)
XKCD_BRIGHT_PINK = Color.from_rgb(254, 1, 177)
XKCD_HOT_PINK = Color.from_rgb(255, 2, 141)
XKCD_ROSE = Color.from_rgb(207, 98, 117)
XKCD_FUCHSIA = Color.from_rgb(237, 13, 217)
XKCD_SALMON = Color.from_rgb(255, 121, 108)
XKCD_CORAL = Color.from_rgb(252, 90, 80)
XKCD_PEACH = Color.from_rgb(255, 176, 124)

# Shades of Brown
XKCD_DARK_BROWN = Color.from_rgb(52, 28, 2)
XKCD_LIGHT_BROWN = Color.from_rgb(173, 129, 80)
XKCD_CHOCOLATE_BROWN = Color.from_rgb(65, 25, 0)
XKCD_COFFEE = Color.from_rgb(166, 129, 76)
XKCD_CARAMEL = Color.from_rgb(175, 111, 9)
XKCD_TAN = Color.from_rgb(209, 178, 111)
XKCD_BEIGE = Color.from_rgb(230, 218, 166)
XKCD_TAUPE = Color.from_rgb(185, 162, 129)
XKCD_KHAKI = Color.from_rgb(170, 166, 98)
XKCD_SAND = Color.from_rgb(226, 202, 118)

# Shades of Grey
XKCD_DARK_GREY = Color.from_rgb(54, 55, 55)
XKCD_LIGHT_GREY = Color.from_rgb(216, 220, 214)
XKCD_CHARCOAL = Color.from_rgb(52, 56, 55)
XKCD_SLATE_GREY = Color.from_rgb(89, 101, 109)
XKCD_SILVER = Color.from_rgb(197, 201, 199)
XKCD_ASH_GREY = Color.from_rgb(178, 190, 181)

# Nature-inspired Colors
XKCD_LEAF_GREEN = Color.from_rgb(92, 169, 4)
XKCD_MOSS_GREEN = Color.from_rgb(101, 139, 56)
XKCD_SEAFOAM = Color.from_rgb(128, 249, 173)
XKCD_OCEAN_BLUE = Color.from_rgb(3, 113, 156)
XKCD_STONE = Color.from_rgb(172, 170, 148)
XKCD_DIRT = Color.from_rgb(138, 110, 69)
XKCD_CLAY = Color.from_rgb(182, 106, 80)
XKCD_RUST = Color.from_rgb(168, 60, 9)

# Pastel Colors
XKCD_PASTEL_BLUE = Color.from_rgb(162, 191, 254)
XKCD_PASTEL_GREEN = Color.from_rgb(176, 255, 157)
XKCD_PASTEL_PINK = Color.from_rgb(255, 186, 205)
XKCD_PASTEL_PURPLE = Color.from_rgb(202, 160, 255)
XKCD_PASTEL_YELLOW = Color.from_rgb(255, 254, 113)
XKCD_PASTEL_ORANGE = Color.from_rgb(255, 150, 79)

# Neon Colors
XKCD_NEON_BLUE = Color.from_rgb(4, 217, 255)
XKCD_NEON_GREEN = Color.from_rgb(12, 255, 12)
XKCD_NEON_PINK = Color.from_rgb(254, 1, 154)
XKCD_NEON_PURPLE = Color.from_rgb(188, 19, 254)
XKCD_NEON_YELLOW = Color.from_rgb(207, 255, 4)

# Metallic Colors
XKCD_GOLD = Color.from_rgb(219, 180, 12)
XKCD_BRONZE = Color.from_rgb(168, 121, 0)
XKCD_COPPER = Color.from_rgb(182, 99, 37)

# Additional Popular Colors
XKCD_CREAM = Color.from_rgb(255, 255, 194)
XKCD_IVORY = Color.from_rgb(255, 255, 203)
XKCD_BURGUNDY = Color.from_rgb(97, 0, 35)
XKCD_MAROON = Color.from_rgb(101, 0, 33)
XKCD_AVOCADO = Color.from_rgb(144, 177, 52)
XKCD_PERIWINKLE = Color.from_rgb(142, 130, 254)
XKCD_LILAC = Color.from_rgb(206, 162, 253)

# ============================================================================
# Convenience Lists
# ============================================================================

PRIMARY_COLORS = [XKCD_RED, XKCD_ORANGE, XKCD_YELLOW, XKCD_GREEN, XKCD_BLUE, XKCD_PURPLE, XKCD_PINK]

VIBRANT_COLORS = [
    XKCD_BRIGHT_RED,
    XKCD_BRIGHT_ORANGE,
    XKCD_BRIGHT_YELLOW,
    XKCD_BRIGHT_GREEN,
    XKCD_BRIGHT_BLUE,
    XKCD_BRIGHT_PURPLE,
    XKCD_BRIGHT_PINK,
]

PASTEL_COLORS = [
    XKCD_PASTEL_BLUE,
    XKCD_PASTEL_GREEN,
    XKCD_PASTEL_PINK,
    XKCD_PASTEL_PURPLE,
    XKCD_PASTEL_YELLOW,
    XKCD_PASTEL_ORANGE,
]

NEON_COLORS = [XKCD_NEON_BLUE, XKCD_NEON_GREEN, XKCD_NEON_PINK, XKCD_NEON_PURPLE, XKCD_NEON_YELLOW]

EARTH_TONES = [XKCD_BROWN, XKCD_TAN, XKCD_BEIGE, XKCD_SAND, XKCD_CLAY, XKCD_DIRT, XKCD_STONE]

GREYSCALE = [XKCD_BLACK, XKCD_CHARCOAL, XKCD_DARK_GREY, XKCD_GREY, XKCD_SILVER, XKCD_LIGHT_GREY, XKCD_WHITE]
