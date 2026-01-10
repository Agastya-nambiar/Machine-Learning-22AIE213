import random
import statistics

def generate_statistics():
    # random function to make 100 values
    random_numbers = [random.randint(100, 150) for _ in range(100)]

    mean_value = statistics.mean(random_numbers)
    median_value = statistics.median(random_numbers)
    mode_value = statistics.mode(random_numbers)

    return random_numbers, mean_value, median_value, mode_value

def count_common_elements(list_one, list_two):
    #  set to not have duplicates
    common_elements = set(list_one).intersection(set(list_two))
    return len(common_elements)

def count_vowels_and_consonants(input_string):
    # convert to lowercase to compare
    input_string = input_string.lower()

    vowels = "aeiou"

    vowel_count = 0
    consonnt_count = 0

    for character in input_string:
        #  only alphabet chars
        if character.isalpha():
            if character in vowels:
                vowel_count += 1
            else:
                consonnt_count += 1

                
    return vowel_count, consonnt_count


def multiply_matrices(matrix_a, matrix_b):
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    # check if possible
    if cols_a != rows_b:
        return None

    # create matrix with zeros
    result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    # matrix multiplication
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result_matrix




def transpose_matrix(input_matrix):
    rows = len(input_matrix)
    cols = len(input_matrix[0])

    # Zero matrix for transpose
    transpose = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transpose[j][i] = input_matrix[i][j]

    return transpose


#main code


# Q!
input_string = "My name is Agastya Nambiar"
vowels, consonants = count_vowels_and_consonants(input_string)

print("Q1:")
print("Input String:", input_string)
print("Vowels:", vowels)
print("Consonants:", consonants)
print()

# Q2
matrix_a = [
    [1, 2, 3],
    [4, 5, 6]
]

matrix_b = [
    [7, 8],
    [9, 10],
    [11, 12]
]

product = multiply_matrices(matrix_a, matrix_b)

print("Q2:")
if product is None:
    print("Matrices cannot be multiplied")
else:
    print("Final Matrix:")
    for row in product:
        print(row)
print()

# Q3
list_one = [1, 2, 3, 4, 5, 6]
list_two = [4, 5, 6, 7, 8]

common_count = count_common_elements(list_one, list_two)

print("Q3:")
print("Common Elements Count:", common_count)
print()

#Q4
matrix = [
    [1, 2, 3],
    [4, 5, 6]
]

transpose = transpose_matrix(matrix)

print("Q4:")
print("Original Matrix:")
for row in matrix:
    print(row)

print("Transpose Matrix:")
for row in transpose:
    print(row)
print()

# Q5
numbers, mean_value, median_value, mode_value = generate_statistics()

print("Q5:")
print("Random Numbers:", numbers)
print("Mean:", mean_value)
print("Median:", median_value)
print("Mode:", mode_value)