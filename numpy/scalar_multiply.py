import numpy as np

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	arr = np.array(matrix)
	scaled = arr * scalar
	result = scaled.tolist()
	
	return result
