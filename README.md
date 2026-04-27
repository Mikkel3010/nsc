# Mandelbrot Project

## Description:
This repo have examples for mandelbrot computations for the following:
- Naive
- Numpy
- Numba
- Numba Parallel
- Dask Local
- Dask Cluster
- GPU

Each implementation can be found in the .py files with the prefix mandelbrot_"xxxx".py.
Each implementation will be able to be run independently in their files.
## Setup and requirements
The code requires the following enviroment to run properly. Package management is done using Mamba.
```bash
mamba env create -f environment.yml
conda activate nsc2026