import random
import statistics

def generate_statistics():
    # random function to make 100 values
    random_numbers = [random.randint(100, 150) for _ in range(100)]

    mean_value = statistics.mean(random_numbers)
    median_value = statistics.median(random_numbers)
    mode_value = statistics.mode(random_numbers)

    return random_numbers, mean_value, median_value, mode_value