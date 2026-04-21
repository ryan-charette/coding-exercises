import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	arr = np.array(matrix)

	eigvals = np.linalg.eigvals(arr)

	sorted_eigvals = np.sort(eigvals.real)[::-1]

	return sorted_eigvals.tolist()
