import numpy as np
from collections import defaultdict
import random
import string

def calculate_markov_entropy(input_string):
    # Count transitions (bigram frequencies)
    transition_counts = defaultdict(lambda: defaultdict(int))
    state_counts = defaultdict(int)

    for i in range(len(input_string) - 1):
        current_char = input_string[i]
        next_char = input_string[i + 1]
        transition_counts[current_char][next_char] += 1
        state_counts[current_char] += 1

    # Compute transition probabilities P(j | i)
    transition_probs = {}
    for current_char, next_chars in transition_counts.items():
        total_transitions = state_counts[current_char]
        transition_probs[current_char] = {
            next_char: count / total_transitions
            for next_char, count in next_chars.items()
        }

    # Compute Markov entropy
    markov_entropy = 0
    total_states = sum(state_counts.values())
    
    for current_char, next_chars in transition_probs.items():
        P_i = state_counts[current_char] / total_states  # P(i)
        for next_char, P_j_given_i in next_chars.items():  # P(j | i)
            markov_entropy -= P_i * P_j_given_i * np.log2(P_j_given_i)

    return markov_entropy

def generate_random_string(length):
    letters = "abcde"
    return ''.join(random.choice(letters) for _ in range(length))

# Example usage
n = 30  # Length of the random string
random_string = generate_random_string(n)
entropy = calculate_markov_entropy(random_string)
print(f"Random String: {random_string}")
print(f"Markov Entropy: {entropy:.4f}")
