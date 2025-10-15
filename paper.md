---
title: 'FDTDX: High-Performance Open-Source FDTD Simulation with Automatic Differentiation'
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
    orcid: 0000-0001-8312-3943
    affiliation: 1
  - name: Lukas Berg
    orcid: 0000-0002-5684-6975
    affiliation: 1
  - name: Bodo Rosenhahn
    orcid: 0000-0003-3861-1424
    affiliation: 1
affiliations:
 - name: Institute of Information Processing, Leibniz University Hannover, Germany
   index: 1
date: 03 July 2025
bibliography: paper.bib
---

# Summary

The ability to precisely manipulate the propagation of light is an increasingly critical factor in many modern applications.
Aside from analytical solutions for simple geometries, the majority of practical problems require robust numerical methods such as the finite-difference time-domain (FDTD) method for discretizing Maxwell's equations in both space and time.
We present FDTDX, an efficient implementation of the FDTD method with GPU acceleration through the JAX framework.
It provides a simple user interface for specifying a simulation scene as well as a suite of tools for inverse design.
Using the time reversibility of maxwell's equations, gradients for optimizing the geometry of a design can be computated efficiently within FDTDX.

# Statement of Need

FDTDX implements the FDTD algorithm, which aims to simulate Maxwell's equations $\frac{\partial H}{\partial t} = - \frac{1}{\mu} \nabla \times E$ and $\frac{\partial E}{\partial t} = \frac{1}{\epsilon} \nabla \times H$, where $E$ and $H$ are the electric and magnetic field components.
This algorithm has been used in a number of research applications, for example in the field of photonic integrated circuits [@augenstein2020inverse], optical computing [@mahlau2025multi] or quantum computing [@larsen2025integrated].

The FDTD algorithm has been well known for a long time and a number of open-source packages already implement it.
However, most previous packages implement the algorithm only for CPU, which misses out on massive speedups through GPU acceleration.
Additionally, the implementation of the FDTD algorithm in JAX allows for automatic differentiation using a specialized algorithm based on the time reversibility of Maxwell's equations [@schubertmahlau2025quantized].
In contrast to the adjoint method, our custom gradient algorithm can calculate a gradient in the time-domain without the need to save electric and magnetic field after every time step.
This enables memory efficient inverse design, i.e. topological optimization of optical components using gradient descent.
The simulation capabilities of FDTDX are useful for physicists from all kinds of backgrounds, while the inverse design capabilities are targeted more towards computational scientists in particular.

A non-exhaustive list of FDTD implementations includes the popular Meep [@meep], which was developed almost 20 years ago for execution on CPU and is still widely used today.
Other frameworks for only CPU include OpenEMS [@openEMS], fdtd [@fdtd_laporte] and Ceviche [@ceviche].
Existing open-source packages that support execution on GPU are Khronos [@khronos] and FDTD-Z [@fdtdz], but both packages are not maintained.
Additionally, various commercial implementations of FDTD exist.
Notably, Tidy3D [@tidy3d] is an extremely fast commercial software due to its GPU acceleration.
A comparison between the different software frameworks can be seen in \autoref{comparison}.

![Feature comparison between different FDTD software frameworks. \label{comparison} ](img/comparison.png){height="70pt"}

<!-- | Feature | Meep | Ceviche | openEMS | Tidy3D | FDTDX |
|---------|------|---------|---------|--------|-------|
| 3D-Simulation | yes | yes | yes | yes | yes |
| Open-Source | yes | yes | yes | no | yes |
| Time Reversibility | no | no | no | no | yes |
| GPU/TPU-capable | no | no | no | yes | yes | -->

<!-- +-------------------+------------+----------+----------+----------+----------+
| Feature           | Meep       | Ceviche  | openEMS  | Tidy3D   |FDTDX     |
|                   |            |          |          |          |          |
+:=================:+:==========:+:========:+:========:+:========:|:========:|
| 3D-Simulation     | yes        | yes      | yes      | yes      | yes      |
+-------------------+------------+----------+----------+----------+----------+
| Open-Source       | yes        | yes      | yes      | no       | yes      |
+-------------------+------------+----------+----------+----------+----------+
| Time Reversibility| no         | no       | no       | no       | yes      |
+-------------------+ -----------+----------+----------|----------+----------+
| GPU/TPU-capable   | no         | no       | no       | yes      | yes      |
+===================+============+==========+==========+==========+==========+
| Table 1: Feature Comparison between different FDTD Software frameworks     |
+============================================================================+ -->


# Implementation 

As the name suggests, the Finite-Difference Time-Domain (FDTD) algorithm discretizes Maxwell's equations in space and time.
To compute the curl operation efficiently using only a single finite difference, the electric and magnetic fields are staggered in both space and time according to the Yee grid [@kaneyeeNumericalSolutionInitial1966].
The initial electric and magnetic fields are updated in a leapfrog pattern.
Firstly, the electric field is updated based on the magnetic field.
Afterwards, the newly computed electric field is the basis for updating the magnetic field.
The staggering through the Yee grids makes these updates very efficient, but as a consequence fields need to be interpolated for accurate measurements.
In FDTDX, physical values can be measured through different detectors, which automatically implement this interpolation.

To inject light into the simulation, the Total-Field Scattered-Field (TFSF) [@taflove] formulation of a source is used in FDTDX. 
This formulation allows injecting light in a single direction into the simulation.
In contrast, a naive additive source implementation would emit light in both directions perpendicular to the injection plane.

<!-- To prevent unwanted reflections at the boundary of the simulation volume, there exist two different boundary conditions in FDTDX. -->
Two different boundary objects can be used to prevent unwanted reflections within the simulation.
A periodic boundary wraps the fields around the simulation volume and automatically injects them on the other side.
This is useful for simulating large repeating areas through a single unit cell, for example in metamaterials [@metamaterial].
The other boundary object is a perfectly matched layer (PML), which absorbs incoming light.
Specifically, the PML in fdtdx is implemented in form of a convolutional PML [@cpml].
<!-- Secondly, reflections can also be prevented by using an absorbing boundary condition, implemented in the form of convolutional perfectly matched layers [@cpml]. -->

![Visualization of a simulation scene using the ```fdtdx.plot_setup``` function. \label{fig_setup} ](img/setup.png)

In FDTDX, the specification of a simulation is simplified by the implementation of a constraint system.
The position and size of sources, detectors or any other simulation objects can be specified using relative constraints.
For example, it might make sense to position a detector next to a source for measuring the input energy in the simulation.
If both detector and source are placed independently, then moving one of the objects also requires moving the other.
With only two objects this is manageable, but with more objects such adaptations quickly become a burden.
In contrast, in FDTDX the position between objects can be specified relative to each other.
Consequently, if one of the objects is moved, the other object automatically moves as well.
Additionally, FDTDX implements utility functions for easily plotting a visualization of the simulation scene.
Such a visualization can be seen in \autoref{fig_setup}.
Similarly, plotting functions are implemented for detectors to visualize the results of a simulation in form of an image or video.
The $E_z$ for the same simulation scene is visualized in \autoref{fig_sim}.

![Visualization of the $E_z$ field in a simulation as output of an ```fdtdx.FieldDetector```. \label{fig_sim} ](img/simulation.png){height="150pt"}

<!-- \begin{figure}[h]
\centering
\includegraphics{img/setup.png}
\caption*{**Figure 1** Visualization of a simulation scene using the fdtdx.plot_setup function.\label{setup}}
\end{figure} -->

# Limitations and Future Work

At the time of publication, FDTDX only supports simulation of linear, non-dispersive materials.
Additionally, lossy materials are currently only partially supported.
In the future, an implementation of dispersive material models [@taflove] is planned.
Simulating non-linear materials is a difficult task due to the necessary theoretical physical knowledge as well as the capabilitiy to experimentally verify the simulation results in the lab.
We are determined to tackle this, but acknowledge that this will require significant effort and time.

# Further Information

The full API and tutorials can be found at the FDTDX [documentation](https://ymahlau.github.io/fdtdx/). 
The source code is publicly available via the corresponding [Github repository](https://github.com/ymahlau/fdtdx).
Additionally, our conference paper on large-scale FDTD simulations [@mahlau2025flexible] provides a good introduction to FDTDX.

# Acknowledgements

We thank [Antonio Calà Lesina](https://www.hot.uni-hannover.de/de/calalesina), [Reinhard Caspary](https://www.phoenixd.uni-hannover.de/de/caspary) and [Konrad Bethmann](https://www.tnt.uni-hannover.de/de/staff/bethmann/) for helping us understand the physics behind Maxwell's equations and how to implement them within FDTD. 
Additionally, we acknowledge [Fabian Hartmann](https://www.tnt.uni-hannover.de/de/staff/hartmann/) for the initial idea of implementing a GPU accelerated FDTD algorithm.
Moreover, community contributions from [Marko Simic](https://github.com/msimicphysics), [Tianxiang Dai](https://github.com/txdai), [Vatsal Limbachia](https://github.com/renaissancenerd), [Agustín Galante](https://github.com/galcerte) and [Robin Giesecke](https://github.com/TheDarkchip) improved features of FDTDX.

This work was supported by the Federal Ministry of Education and Research (BMBF), Germany under the AI service center KISSKI (grant no. 01IS22093C), the European Union within the Horizon Europe research and innovation programme under grant agreement no. 101136006 – XTREME, the Lower Saxony Ministry of Science and Culture (MWK) through the zukunft.niedersachsen program of the Volkswagen Foundation and the Deutsche Forschungsgemeinschaft (DFG) under Germany’s Excellence Strategy within the Cluster of Excellence PhoenixD (EXC 2122) and (RO2497/17-1). Additionally, this was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 517733257.

# References