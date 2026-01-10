from count_of_vowels_and_consonants import count_vowels_and_consonants
from multiply_matrices import multiply_matrices
from Transpose_matrix import transpose_matrix
from common_elements_count import count_common_elements
from Generate_statistics import generate_statistics

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