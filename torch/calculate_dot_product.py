import torch

def calculate_dot_product(vec1, vec2):
	"""
	Calculate the dot product of two vectors.
	Args:
		vec1 (torch.tensor): 1D tensor representing the first vector.
		vec2 (torch.tensor): 1D tensor representing the second vector.
	Returns:
		The dot product of the two vectors.
	"""
	return torch.dot(vec1, vec2)
