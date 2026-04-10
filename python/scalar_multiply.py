def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	result = []
	for row in matrix:
		new_row = []
		for value in row:
			product = value * scalar
			new_row.append(product)
		result.append(new_row)
	return result
