# Coding Exercises for Machine Learning Algorithms

This repository contains implementation-focused coding exercises for algorithms and computational patterns commonly used in machine learning.

The repository is intended to grow into a structured collection of machine learning algorithm exercises implemented across multiple execution environments, with an emphasis on correctness, clarity, and performance trade-offs.

## Objectives

The goals of this repository are to:

- Implement core algorithms used in machine learning from first principles where practical.
- Compare equivalent implementations across Python, NumPy, PyTorch, and CUDA.
- Build intuition for the relationship between mathematical operations, software abstractions, and hardware execution.
- Provide small, focused examples that are easy to inspect, run, and extend.

## Repository Structure

```text
.
├── CUDA/      # CUDA implementations for selected exercises
├── numpy/     # NumPy implementations
├── python/    # Pure Python reference implementations
└── torch/     # PyTorch implementations
```

Each directory contains standalone implementations of individual exercises. When practical, the same exercise is implemented in multiple frameworks to make comparison straightforward.

## Current Exercises

The current exercise set includes:

- Dot product
- Matrix-vector multiplication
- Matrix transpose
- Scalar multiplication
- Matrix mean
- Eigenvalue calculation
- Cosine similarity

## Requirements

### Python

Use Python 3.10 or newer where possible.

### NumPy

```bash
pip install numpy
```

### PyTorch

PyTorch is required only for exercises in the `torch/` directory.

```bash
pip install torch
```

### CUDA

CUDA exercises require:

- An NVIDIA GPU
- NVIDIA CUDA Toolkit
- `nvcc` compiler available on the system path

## Running Exercises

Run Python, NumPy, or PyTorch exercises directly from the repository root:

```bash
python python/calculate_dot_product.py
python numpy/calculate_dot_product.py
python torch/calculate_dot_product.py
```

Compile and run CUDA exercises with `nvcc`:

```bash
nvcc CUDA/matrix_dot_vector.cu -o matrix_dot_vector
./matrix_dot_vector
```
