import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    eigenvalues = torch.linalg.eigvals(matrix)

    sorted_eigvals, _ = torch.sort(eigenvalues.real, descending=True)

    return sorted_eigvals
