.. fdtdx documentation master file, created by
   sphinx-quickstart on Fri Aug 29 13:08:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



.. image:: _static/logo.png

FDTDX Documentation
===================
**FDTDX** is a high-performance framework for electromagnetic simulations and inverse design of photonic devices. Built on JAX, it provides GPU-accelerated FDTD (Finite-Difference Time-Domain) simulations with automatic differentiation capabilities.

Key Features
---------------
- Native GPU acceleration through JAX
- Multi-GPU scaling for large simulations 
- Memory-efficient time-reversal implementation
- Optimized for large-scale inverse design
- Flexible boundary conditions with PML support

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


Guides
--------------
- More guides will follow shortly

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

