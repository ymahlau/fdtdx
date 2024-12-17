"""Core functionality for the FDTDX package.

This package contains the fundamental building blocks and abstractions used throughout
FDTDX, including:

- JAX-based data structures and utilities for efficient computation and automatic differentiation
- Physics computations and metrics for electromagnetic simulations
- Base classes for simulation components like sources, detectors, and materials
- Configuration and state management for FDTD simulations
- Core algorithms for time-domain electromagnetic field updates

The core module provides the essential infrastructure needed to run FDTD simulations
and perform inverse design optimization. It implements memory-efficient automatic
differentiation based on the time-reversibility of Maxwell's equations.
"""
