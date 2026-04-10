def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		v1: 1D array representing the first vector.
		v2: 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	dot_product = 0
	norm_v1 = 0
	norm_v2 = 0
	
	for i in range(len(v1)):
		dot_product += v1[i] * v2[i]
		norm_v1 += v1[i] ** 2
		norm_v2 += v2[i] ** 2

	norm_v1 **= 0.5
	norm_v2 **= 0.5

	return dot_product / (norm_v1 * norm_v2)
