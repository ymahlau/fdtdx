
<div align="center">
<img src="docs/source/_static/logo.png" alt="logo"></img>
</div>

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://fdtdx.readthedocs.io/en/latest/)
[![arXiv](https://img.shields.io/badge/arXiv-2407.10273-b31b1b.svg)](https://arxiv.org/abs/2407.10273)
[![arXiv](https://img.shields.io/badge/arXiv-2412.12360-b31b1b.svg)](https://arxiv.org/abs/2412.12360)
[![codecov](https://codecov.io/gh/ymahlau/fdtdx/branch/main/graph/badge.svg)](https://codecov.io/gh/ymahlau/fdtdx)
[![PyPI version](https://img.shields.io/pypi/v/fdtdx)](https://pypi.org/project/fdtdx/)
[![Continuous integration](https://github.com/ymahlau/fdtdx/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/ymahlau/fdtdx/actions/workflows/main.yml/badge.svg?branch=main)
[![status](https://joss.theoj.org/papers/d0844d2ee5a573165bcc61fb51f575ae/status.svg)](https://joss.theoj.org/papers/d0844d2ee5a573165bcc61fb51f575ae)

# FDTDX: Electromagnetic Simulations in JAX

FDTDX is an efficient open-source Python package for the simulation and design of three-dimensional photonic nanostructures using the Finite-Difference Time-Domain (FDTD) method. Built on JAX, it provides native GPU support and automatic differentiation capabilities, making it ideal for large-scale design tasks.

## Key Features
The key features differentiating FDTDX from other simulation software packages like Meep (which is also great!) are the following:

- **High Performance**: GPU-accelerated FDTD simulations with multi-GPU scaling capabilities
- **Memory Efficient**: Leverages time-reversibility in Maxwell's equations for efficient gradient computation
- **Automatic Differentiation**: Built-in gradient-based optimization for complex 3D structures
- **User-Friendly API**: Intuitive positioning and sizing of objects in absolute or relative coordinates
- **Large-Scale Design**: Capable of handling simulations with billions of grid cells
- **Open Source**: Freely available for research, development and commercial use.

## Documentation

Visit our [documentation](https://fdtdx.readthedocs.io/en/latest/) for:
- Detailed API reference
- Tutorial guides
- Best practices

Also check out our [whitepaper](https://arxiv.org/abs/2412.12360) for some examples and a comparison to other popular FDTD-frameworks.

## Installation

Install FDTDX using pip:

```bash
pip install fdtdx  # Basic CPU-Installation
pip install fdtdx[cuda12]  # GPU-Acceleration (Highly Recommended!)
pip install fdtdx[rocm]   # AMD-GPU (only python<=3.12)
```

For development installation, see the contributing guidelines!

## Multi-GPU

```bash
# The following lines often lead to better memory usage in JAX
# when using multiple GPU.
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export NCCL_LL128_BUFFSIZE="-2"
export NCCL_LL_BUFFSIZE="-2"
export NCCL_PROTO="SIMPLE,LL,LL128"
```

## Citation
If you find this repository helpful for you work, please consider citing:
```
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
```

## Acknowedgement
This project was developed at the [Institute of Information Processing](https://www.tnt.uni-hannover.de/) at Leibniz University Hannover, Germany and sponsored by the cluster of excellence [PhoenixD](https://www.phoenixd.uni-hannover.de/en/) (Photonics, Optics, Engineering, Innovation across Disciplines).
