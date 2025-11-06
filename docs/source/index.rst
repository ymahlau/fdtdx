.. fdtdx documentation master file, created by
   sphinx-quickstart on Fri Aug 29 13:08:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



.. image:: _static/logo.png

FDTDX Documentation
===================
**FDTDX** is an efficient open-source Python package for the simulation and design of three-dimensional photonic nanostructures using the Finite-Difference Time-Domain (FDTD) method. Built on JAX, it provides native GPU support and automatic differentiation capabilities, making it ideal for large-scale design tasks.

Key Features
---------------

The key features differentiating FDTDX from other simulation software packages like Meep (which is also great!) are the following:

- **High Performance**: GPU-accelerated FDTD simulations with multi-GPU scaling capabilities
- **Memory Efficient**: Leverages time-reversibility in Maxwell's equations for efficient gradient computation
- **Automatic Differentiation**: Built-in gradient-based optimization for complex 3D structures
- **User-Friendly API**: Intuitive positioning and sizing of objects in absolute or relative coordinates
- **Large-Scale Design**: Capable of handling simulations with billions of grid cells
- **Open Source**: Freely available for research, development and commercial use.

Check out the Quickstart Guides for an introduction into FDTDX and the examples in the github repository!

Installation
-------------

Install FDTDX using pip:

.. code-block:: bash

   pip install fdtdx  # Basic CPU-Installation
   pip install fdtdx[cuda12]  # GPU-Acceleration (Highly Recommended!)
   pip install fdtdx[rocm]   # AMD-GPU (only python<=3.12)


For development installation, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/ymahlau/fdtdx
   cd fdtdx
   pip install -e . --extra dev



Citation
--------------

If you find this repository helpful for your work, please consider citing:

.. code-block:: bibtex

   @article{schubert2025quantized,
      title={Quantized inverse design for photonic integrated circuits},
      author={Schubert, Frederik and Mahlau, Yannik and Bethmann, Konrad and Hartmann, Fabian and Caspary, Reinhard and Munderloh, Marco and Ostermann, J{\"o}rn and Rosenhahn, Bodo},
      journal={ACS omega},
      volume={10},
      number={5},
      pages={5080--5086},
      year={2025},
      publisher={ACS Publications}
   }


.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   quickstart
   advanced
   api
   contributing
   faq

