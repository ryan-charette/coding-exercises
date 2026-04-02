def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	if len(a[0]) != len(b):
		return -1

	result = []

	for row in a:
		dot = 0

		for i in range(len(b)):
			dot += row[i] * b[i]

		result.append(dot)

	return result
