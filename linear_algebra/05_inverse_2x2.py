import numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    arr = np.array(matrix, dtype=float)

    determinant = np.linalg.det(arr)

    if np.isclose(determinant, 0.0):
        return None

    inverse = np.linalg.inv(arr)
    return inverse.tolist()
