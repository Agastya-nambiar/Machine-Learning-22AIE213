def transpose_matrix(input_matrix):
    rows = len(input_matrix)
    cols = len(input_matrix[0])

    # Zero matrix for transpose
    transpose = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transpose[j][i] = input_matrix[i][j]

    return transpose