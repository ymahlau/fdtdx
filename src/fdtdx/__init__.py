"""FDTDX: A JAX-based Finite-Difference Time-Domain (FDTD) Electromagnetic Simulation Package.

A high-performance framework for electromagnetic simulations and inverse design optimization
using the FDTD method, implemented in JAX for GPU acceleration and automatic differentiation.

Key Features:
    - Memory-efficient automatic differentiation using time-reversibility
    - Multi-GPU support for large-scale simulations
    - Intuitive object positioning with relative coordinates
    - Built-in optimization capabilities for inverse design
    - Comprehensive boundary condition implementations (e.g., PML)
    - Field detectors and visualization tools
    - JIT-compiled operations for maximum performance

Example:
    Basic simulation setup:
    ```python
    import fdtdx
    from fdtdx.objects import Source, Detector

    # Create simulation components
    source = Source(...)
    detector = Detector(...)

    # Run simulation
    fields = fdtdx.simulate(source, detector)
    ```

Notes:
    The package is designed for both research and industrial applications in nanophotonics,
    metamaterials, and photonic integrated circuits. It supports both 2D and 3D simulations
    with automatic gradient computation for optimization tasks.

References:
    - Paper: https://github.com/ymahlau/fdtdx
    - Documentation: https://github.com/ymahlau/fdtdx/docs
"""
