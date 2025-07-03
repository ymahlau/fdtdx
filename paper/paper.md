---
title: 'FDTDX: Simulating Maxwells Equations in the Time Domain'
tags:
  - Python
  - Electromagnetics
  - Photonics
  - Numerical Simulation
authors:
  - name: Yannik Mahlau
    corresponding: true
    orcid: 0000-0003-0425-5003
    affiliation: 1
  - name: Frederik Schubert
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Bodo Rosenhahn
    orcid: 0000-0003-3861-1424
    affiliation: 1
affiliations:
 - name: Institute of Information Processing, Leibniz University Hannover, Germany
   index: 1
date: 03 July 2025
bibliography: paper/paper.bib
---

# Summary

The behavior of electromagnetic fields in complex media and structures under time-varying conditions leads to the propagation, scattering, and interaction of electromagnetic waves across diverse environments. 
The temporal and spatial evolution of these fields is essential to understanding wave phenomena, antenna design, or photonic devices.
Aside from analytical solutions for simple geometries, the majority of practical problems require robust numerical methods such as the finite-difference time-domain (FDTD) method for discretizing Maxwell's equations in both space and time.
FDTDX is an efficient implementation of the FDTD method with GPU acceleration through the JAX framework.
It provides a simple user interface for specifying a simulation scene and tools for inverse design of geometric components using automatic differentiation.


# Statement of need

FDTDX implements the FDTD algorithm, which aims to simulate maxwell's equations $\frac{\partial H}{\partial t} &= - \frac{1}{\mu} \nabla \times E$ and $\frac{\partial E}{\partial t} &= \frac{1}{\epsilon} \nabla \times H$.
It discretizes the differential equations in space and time according to the Yee grid [@kaneyeeNumericalSolutionInitial1966].
This algorithm has been used in a number of research applications, for example in the field of photonic integrated circuits [@schubertmahlau2025quantized], optical computing[@mahlau2025multi] or quantum computing [@larsen2025integrated].

The FDTD algorithm has been well known for a long time and a number of open source packages already implement it.
However, due to their age, most previous packages implement the algorithm only for CPU, while GPU acceleration offers massive speedups.




<!-- `Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike. -->

# Acknowledgements

We acknowledge contributions from Antonio Calà Lesina, Reinhard Caspary and Konrad Bethmann for understanding the physics behind Maxwell's equations and how to implement them within FDTD. Additionally, we acknowledge Fabian Hartmann for the initial idea of implementing a GPU accelerated FDTD algorithm.

This work was supported by the Federal Ministry of Education and Research (BMBF), Germany under the AI service center KISSKI (grant no. 01IS22093C), the Lower Saxony Ministry of Science and Culture (MWK) through the zukunft.niedersachsen program of the Volkswagen Foundation and the Deutsche Forschungsgemeinschaft (DFG) under Germany’s Excellence Strategy within the Cluster of Excellence PhoenixD (EXC 2122) and (RO2497/17-1). Additionally, this was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 517733257.

# References