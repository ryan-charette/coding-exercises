# Scientific Machine Learning Coding Exercises

This repository is a curated set of 50 coding exercises covering the core foundations of scientific machine learning, numerical computing, reinforcement learning, control, uncertainty quantification, and deep learning.

Inspired by LeetCode's Blind 75, it prioritizes high-leverage concepts and implementation patterns over exhaustive coverage. The goal is to make the most important ideas easy to practice, inspect, and revisit for engineers, researchers, and practitioners working at the intersection of machine learning and scientific computing.

Each exercise focuses on one algorithm, metric, layer, update rule, numerical method, or computational pattern. Some files are complete reference implementations, while others are intentionally structured as prompts or skeletons to fill in.

## Goals

- Practice implementing scientific machine learning and numerical computing concepts from first principles.
- Build intuition for the math behind common ML, optimization, and simulation workflows.
- Compare CPU, PyTorch, and CUDA implementations where useful.
- Keep examples small enough to inspect, test, and modify quickly.
- Create a focused reference set for interview preparation, coursework, research onboarding, and hands-on review.

## Repository Structure

```text
.
|-- classical_ml/            # Regression, classification, clustering, SVMs, trees, and KNN
|-- data_processing/         # Encoding, batching, scaling, validation, and evaluation metrics
|-- linear_algebra/          # Matrix operations, decompositions, sparse matrices, PCA, and CUDA kernels
|-- neural_networks/         # Layers, losses, autograd, CNNs, probabilistic models, and distances
|-- optimizers_training/     # Gradient descent, Adam, momentum, early stopping, and mixed precision
`-- reinforcement_learning/  # Bandits, Bellman updates, policy evaluation, gradients, and Q-learning
```

## Exercise Areas

### Classical Machine Learning

- Linear regression with the normal equation
- Logistic regression and softmax regression
- Decision trees
- K-nearest neighbors
- K-means clustering
- Elastic net gradient descent
- Pegasos kernel SVM

### Data Processing and Metrics

- Feature scaling
- One-hot encoding
- Data shuffling
- Batch iteration
- K-fold cross validation
- RMSE
- F-score

### Linear Algebra

- 2x2 matrix inverse
- Eigenvalue calculation
- Singular values for a 2x2 matrix
- Orthonormal basis construction
- Jacobi solver
- Covariance matrix calculation
- PCA
- Compressed row sparse matrix representation
- CUDA matrix-vector multiplication
- CUDA matrix multiplication

### Neural Networks

- Dense layers
- Dropout
- Batch normalization
- Cross-entropy loss
- Manual autograd
- Simple neuron training
- Simple CNN training with backpropagation
- Residual blocks
- Dice score
- Gaussian processes
- Markov chain simulation
- KL divergence
- Bhattacharyya distance
- CUDA 2D convolution

### Optimizers and Training

- Gradient descent
- Momentum optimizer
- Adam optimizer
- Early stopping
- Mixed precision training concepts

### Reinforcement Learning

- Incremental mean updates
- Epsilon-greedy action selection
- Bellman updates
- Gridworld policy evaluation
- Policy gradient computation
- Q-learning

## Requirements

Most Python exercises use Python 3.10 or newer.

Install dependencies as needed for the files you want to run:

```bash
pip install numpy torch
```

CUDA exercises require:

- An NVIDIA GPU
- NVIDIA CUDA Toolkit
- `nvcc` available on your system path

There is currently no shared package manifest. Dependencies are intentionally lightweight and are imported directly by individual exercise files.

## Running Exercises

Run Python exercises directly from the repository root:

```bash
python classical_ml/linear_regression_normal_equation.py
python data_processing/feature_scaling.py
python linear_algebra/pca.py
python reinforcement_learning/bellman_update.py
```

PyTorch-based exercises can also be run directly when they include executable examples:

```bash
python neural_networks/dense_layer.py
python optimizers_training/adam_optimizer.py
```

Compile CUDA exercises with `nvcc`:

```bash
nvcc linear_algebra/matrixmul.cu -o matrixmul
./matrixmul
```

On Windows PowerShell, run the compiled executable with:

```powershell
.\matrixmul.exe
```

## Working on an Exercise

1. Open a file in the topic area you want to practice.
2. Read the function signature and any inline comments or docstrings.
3. Implement the missing logic or modify the existing implementation.
4. Add a small example, assertion, or test case to validate the result.
5. Compare your output against a known formula, NumPy/PyTorch equivalent, or hand-calculated case.

## Notes

- Files are intentionally small and standalone.
- Some exercises are prompts with placeholder implementations.
- CUDA files focus on kernel structure and host-device data movement, not production-level error handling.
- The repository is organized by topic rather than by framework so related concepts stay together.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
