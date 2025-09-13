# utils.py
# A new file with a simple utility function to demonstrate the CI/CD test runner.

def multiply_matrices(matrix_a, matrix_b):
    """
    Multiplies two matrices.

    Args:
        matrix_a (list of lists): The first matrix.
        matrix_b (list of lists): The second matrix.

    Returns:
        list of lists: The resulting matrix from the multiplication.

    Raises:
        ValueError: If the matrices cannot be multiplied due to incompatible dimensions.
    """
    # Check for empty matrices
    if not matrix_a or not matrix_b or not matrix_a[0] or not matrix_b[0]:
        raise ValueError("Matrices cannot be empty.")

    # Get the dimensions of the matrices
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    # Check for compatible dimensions
    if cols_a != rows_b:
        raise ValueError("Matrices cannot be multiplied. Number of columns in the first matrix must equal the number of rows in the second matrix.")

    # Initialize the result matrix with zeros
    result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    # Perform matrix multiplication
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result_matrix
