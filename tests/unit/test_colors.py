"""Unit tests for fdtdx.colors module."""

import pytest

from fdtdx.colors import (
    EARTH_TONES,
    GREYSCALE,
    NEON_COLORS,
    PASTEL_COLORS,
    PRIMARY_COLORS,
    VIBRANT_COLORS,
    XKCD_BLACK,
    XKCD_BLUE,
    XKCD_GREEN,
    XKCD_RED,
    XKCD_WHITE,
    Color,
)


class TestColorConstruction:
    """Tests for Color class construction."""

    def test_construct_with_normalized_values(self):
        """Test creating a Color with normalized RGB values."""
        color = Color(r=0.5, g=0.25, b=0.75)
        assert color.r == 0.5
        assert color.g == 0.25
        assert color.b == 0.75

    def test_construct_with_boundary_values(self):
        """Test creating a Color with boundary values (0 and 1)."""
        black = Color(r=0.0, g=0.0, b=0.0)
        white = Color(r=1.0, g=1.0, b=1.0)
        assert (black.r, black.g, black.b) == (0.0, 0.0, 0.0)
        assert (white.r, white.g, white.b) == (1.0, 1.0, 1.0)

    def test_construct_with_invalid_values_raises(self):
        """Test that out-of-range values raise ValueError."""
        with pytest.raises(ValueError, match="RGB components must be in range"):
            Color(r=1.1, g=0.5, b=0.5)
        with pytest.raises(ValueError, match="RGB components must be in range"):
            Color(r=0.5, g=-0.1, b=0.5)


class TestColorFromRgb:
    """Tests for Color.from_rgb class method."""

    def test_from_rgb_converts_to_normalized(self):
        """Test that from_rgb converts 8-bit values to normalized."""
        color = Color.from_rgb(100, 150, 200)
        assert abs(color.r - 100 / 255) < 1e-10
        assert abs(color.g - 150 / 255) < 1e-10
        assert abs(color.b - 200 / 255) < 1e-10

    def test_from_rgb_boundary_values(self):
        """Test from_rgb with boundary values."""
        black = Color.from_rgb(0, 0, 0)
        white = Color.from_rgb(255, 255, 255)
        assert (black.r, black.g, black.b) == (0.0, 0.0, 0.0)
        assert (white.r, white.g, white.b) == (1.0, 1.0, 1.0)


class TestColorFromHex:
    """Tests for Color.from_hex class method."""

    def test_from_hex_six_char(self):
        """Test creating a color from 6-character hex string."""
        color = Color.from_hex("#AbCdEf")
        assert abs(color.r - 0xAB / 255) < 1e-10
        assert abs(color.g - 0xCD / 255) < 1e-10
        assert abs(color.b - 0xEF / 255) < 1e-10

    def test_from_hex_without_hash(self):
        """Test creating a color from hex string without hash prefix."""
        color = Color.from_hex("00FF00")
        assert (color.r, color.g, color.b) == (0.0, 1.0, 0.0)

    def test_from_hex_short_form(self):
        """Test creating a color from 3-character hex string."""
        color = Color.from_hex("#F00")
        assert (color.r, color.g, color.b) == (1.0, 0.0, 0.0)

    def test_from_hex_invalid_raises(self):
        """Test that invalid hex strings raise ValueError."""
        with pytest.raises(ValueError, match="Invalid hex color"):
            Color.from_hex("#FF00")  # Wrong length
        with pytest.raises(ValueError, match="Invalid hex color"):
            Color.from_hex("#GGGGGG")  # Invalid characters


class TestColorConversions:
    """Tests for Color conversion methods."""

    def test_to_rgb_normalized(self):
        """Test to_rgb_normalized returns correct tuple."""
        color = Color(r=0.5, g=0.25, b=0.75)
        assert color.to_rgb_normalized() == (0.5, 0.25, 0.75)

    def test_to_rgb_255_roundtrip(self):
        """Test that from_rgb and to_rgb_255 roundtrip correctly."""
        original = (100, 150, 200)
        color = Color.from_rgb(*original)
        assert color.to_rgb_255() == original

    def test_to_hex_roundtrip(self):
        """Test that from_hex and to_hex roundtrip correctly."""
        original = "#ABCDEF"
        color = Color.from_hex(original)
        assert color.to_hex() == original

    def test_to_mpl_equals_to_rgb_normalized(self):
        """Test that to_mpl returns same as to_rgb_normalized."""
        color = Color(r=0.5, g=0.25, b=0.75)
        assert color.to_mpl() == color.to_rgb_normalized()


class TestColorDunderMethods:
    """Tests for Color special methods."""

    def test_str_returns_hex(self):
        """Test that str() returns hex representation."""
        color = Color(r=1.0, g=0.0, b=0.0)
        assert str(color) == "#FF0000"

    def test_iter_unpacking(self):
        """Test that Color can be unpacked via iteration."""
        color = Color(r=0.5, g=0.25, b=0.75)
        r, g, b = color
        assert (r, g, b) == (0.5, 0.25, 0.75)

    def test_len_returns_three(self):
        """Test that len() returns 3."""
        color = Color(r=0.5, g=0.25, b=0.75)
        assert len(color) == 3


class TestPredefinedColors:
    """Tests for predefined XKCD colors."""

    def test_xkcd_primary_colors_values(self):
        """Test XKCD primary colors have expected values."""
        assert XKCD_RED.to_rgb_255() == (229, 0, 0)
        assert XKCD_GREEN.to_rgb_255() == (21, 176, 26)
        assert XKCD_BLUE.to_rgb_255() == (3, 67, 223)
        assert XKCD_WHITE.to_rgb_255() == (255, 255, 255)
        assert XKCD_BLACK.to_rgb_255() == (0, 0, 0)


class TestColorLists:
    """Tests for predefined color lists."""

    def test_color_lists_contain_colors(self):
        """Test all color lists contain Color instances."""
        all_lists = [PRIMARY_COLORS, VIBRANT_COLORS, PASTEL_COLORS, NEON_COLORS, EARTH_TONES, GREYSCALE]
        for color_list in all_lists:
            assert len(color_list) > 0
            assert all(isinstance(c, Color) for c in color_list)

    def test_greyscale_ordering(self):
        """Test GREYSCALE starts with black and ends with white."""
        assert GREYSCALE[0] == XKCD_BLACK
        assert GREYSCALE[-1] == XKCD_WHITE
