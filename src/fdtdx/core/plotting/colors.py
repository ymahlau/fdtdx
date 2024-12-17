"""Color constants for visualization and plotting.

This module provides a collection of predefined RGB color tuples normalized to the range [0,1].
Colors are organized into categories: primary/bright colors, grayscale, and earth tones.
These colors are designed to provide a consistent and visually appealing palette for plotting
and visualization tasks throughout the FDTDX framework.

Each color is defined as a tuple of (red, green, blue) values normalized to [0,1].
The normalization is done by dividing 8-bit RGB values (0-255) by 255.

Categories:
    - Bright and primary colors: Vibrant colors for emphasis and contrast
    - Grayscale colors: Various shades of gray for backgrounds and subtle elements
    - Earth tones: Natural, warm colors for material representations

Example:
    >>> import matplotlib.pyplot as plt
    >>> plt.plot([0, 1], [0, 1], color=GREEN)  # Plot a line in vibrant green
    >>> plt.fill_between([0, 1], [0, 1], color=LIGHT_BLUE, alpha=0.3)  # Fill with pale blue
"""

# Bright and primary colors
GREEN = (21 / 255, 176 / 255, 26 / 255)  # Vibrant green
PINK = (255 / 255, 129 / 255, 192 / 255)  # Soft pink
MAGENTA = (194 / 255, 0, 120 / 255)  # Deep magenta
CYAN = (0, 1, 1)  # Pure cyan
LIGHT_BLUE = (149 / 255, 208 / 255, 252 / 255)  # Pale blue
ORANGE = (249 / 255, 115 / 255, 6 / 255)  # Bright orange
LIGHT_GREEN = (150 / 255, 249 / 255, 123 / 255)  # Pale green

# Grayscale colors
GREY = (146 / 255, 149 / 255, 145 / 255)  # Medium gray
DARK_GREY = (54 / 255, 55 / 255, 55 / 255)  # Deep gray
LIGHT_GREY = (216 / 255, 220 / 255, 214 / 255)  # Pale gray

# Earth tones
BROWN = (101 / 255, 55 / 255, 0)  # Deep brown
LIGHT_BROWN = (173 / 255, 129 / 255, 80 / 255)  # Lighter brown
TAN = (209 / 255, 178 / 255, 111 / 255)  # Warm tan
BEIGE = (230 / 255, 218 / 255, 166 / 255)  # Light beige
