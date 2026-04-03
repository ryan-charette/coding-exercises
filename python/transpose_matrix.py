def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    a_T = []
    
    for col in range(len(a[0])):
        curr_row = []
        for row in range(len(a)):
            curr_row.append(a[row][col])
        a_T.append(curr_row)

    return a_T
