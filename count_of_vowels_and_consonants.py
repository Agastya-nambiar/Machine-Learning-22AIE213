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