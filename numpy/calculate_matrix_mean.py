import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	arr = np.array(matrix, dtype=float)

	if mode == "row":
		means = np.mean(arr, axis=1)
	elif mode == "column":
		means = np.mean(arr, axis=0)

	return means.tolist()
