# This Repository is currently under construction. We will add proper Documentation and Installation Mid-December.

# FDTDX

FDTDX is a JAX implementation of the finite-difference time-domain FDTD method for solving Maxwell's equations.
See the [Paper](https://arxiv.org/abs/2407.10273) for more details.
Since this repository is currently under construction expect a lot of changes in the following weeks.
The scripts folder already contains a few examples on how to use this respository.


## Installation

```bash
# if you want faster env creation, use [mamba] as a drop-in replacement for conda
# (https://mamba.readthedocs.io/en/latest/) 
conda env create -f environment.yaml
conda activate fdtdx
pip install -e .
# GPUs are optional, but recommended
pip install -U "jax[cuda12]"
```

## Multi-GPU

```bash
# The following lines often lead to better memory usage in JAX
# when using multiple GPU.
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NCCL_LL128_BUFFSIZE="-2"
export NCCL_LL_BUFFSIZE="-2"
export NCCL_PROTO="SIMPLE,LL,LL128"
```

## Citation
If you find this repository helpful for you work, please consider citing:
```
@article{schubert2024quantized,
  title={Quantized Inverse Design for Photonic Integrated Circuits},
  author={Schubert, Frederik and Mahlau, Yannik  and Bethmann, Konrad and Hartmann, Fabian and Caspary, Reinhard and Munderloh, Marco and Ostermann, J{\"o}rn and Rosenhahn, Bodo},
  journal={arXiv preprint arXiv:2407.10273},
  year={2024}
}
```