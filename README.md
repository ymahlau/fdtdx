

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://ymahlau.github.io/fdtdx)
[![arXiv](https://img.shields.io/badge/arXiv-2407.10273-b31b1b.svg)](https://arxiv.org/abs/2407.10273)
[![arXiv](https://img.shields.io/badge/arXiv-2412.12360-b31b1b.svg)](https://arxiv.org/abs/2412.12360)

![image](/docs/img/logo.png)

FDTDX is an efficient open-source Python package for the inverse design of three-dimensional photonic nanostructures using the Finite-Difference Time-Domain (FDTD) method. Built on JAX, it provides native GPU support and automatic differentiation capabilities, making it ideal for large-scale 3D design in nanophotonics.

## Key Features

- **High Performance**: GPU-accelerated FDTD simulations with multi-GPU scaling capabilities
- **Memory Efficient**: Leverages time-reversibility in Maxwell's equations for efficient gradient computation
- **Automatic Differentiation**: Built-in gradient-based optimization for complex 3D structures
- **User-Friendly API**: Intuitive positioning and sizing of objects in absolute or relative coordinates
- **Large-Scale Design**: Capable of handling simulations with billions of grid cells
- **Open Source**: Freely available for research and development

## Documentation

Visit our [documentation](https://ymahlau.github.io/fdtdx) for:
- Detailed API reference
- Tutorial guides
- Best practices

Also check out our [whitepaper](https://arxiv.org/abs/2412.12360) for some examples and a comparison to other popular FDTD-frameworks.

## Installation

Install FDTDX using pip:

```bash
pip install fdtdx
```

For development installation, clone the repository and install in editable mode:

```bash
git clone https://github.com/ymahlau/fdtdx
cd fdtdx
pip install -e .
```

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
@article{schubertmahlau2025quantized,
  title={Quantized Inverse Design for Photonic Integrated Circuits},
  author={Schubert, Frederik and Mahlau, Yannik and Bethmann, Konrad and Hartmann, Fabian and Caspary, Reinhard and Munderloh, Marco and Ostermann, J{\"o}rn and Rosenhahn, Bodo},
  journal={ACS Omega},
  doi={10.1021/acsomega.4c10958},
  year={2025}
}
```

## Acknowedgement
This project was developed at the [Institute of Information Processing](https://www.tnt.uni-hannover.de/) at Leibniz University Hannover, Germany and sponsored by the cluster of excellence [PhoenixD](https://www.phoenixd.uni-hannover.de/en/) (Photonics, Optics, Engineering, Innovation across Disciplines).
