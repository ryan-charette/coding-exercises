def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	means = []

	if mode == "row":
		for row in matrix:
			row_mean = sum(row) / len(row)
			means.append(row_mean)

	if mode == "column":
		num_cols = len(matrix[0])

		for col in range(num_cols):
			col_total = 0.0

			for row in matrix:
				col_total += row[col]

			means.append(col_total / len(matrix))

	return means
