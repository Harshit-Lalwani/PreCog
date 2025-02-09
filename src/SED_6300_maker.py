"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# Load dataset
df = pd.read_csv("exploration_results.csv")

# 1. Stratified Sampling (70%) - Preserves pattern_score distribution
bins = np.linspace(df["pattern_score"].min(), df["pattern_score"].max(), 20)  # Create bins for pattern_score
labels = range(len(bins)-1)
df["score_bin"] = pd.cut(df["pattern_score"], bins=bins, labels=labels)
stratified_sample = df.groupby("score_bin", group_keys=False).apply(lambda x: x.sample(frac=0.7, random_state=42))

# 2. Max-Min Diversity Sampling (30%) - Ensures rare points
remaining_data = df.drop(stratified_sample.index)
selected_indices = [np.random.randint(0, len(remaining_data))]  # Start with a random point

for _ in range(30):  # Select 30 diverse points
    dists = cdist(remaining_data.iloc[selected_indices][["pattern_score"]], remaining_data[["pattern_score"]])
    min_dists = np.min(dists, axis=0)  # Distance to the closest selected point
    selected_indices.append(np.argmax(min_dists))

diverse_sample = remaining_data.iloc[selected_indices]

# Combine both sets to create the final dataset
final_sample = pd.concat([stratified_sample, diverse_sample]).sample(n=100, random_state=42)

# Save the selected 100 points
final_sample.to_csv("representative_sample.csv", index=False)
print("Representative dataset saved as representative_sample.csv")
"""

import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from pipeline_utils import calculate_markov_entropy, get_string_sampler, get_transition_array_sampler, generate_single_path_for_pipeline

def generate_datasets():
    datasets = []
    for i in range (1, 10):
        for n in range(1, 11):
            for t in range(1, 8):
                for d in range(t, 3 * t + 1):
                    string_sampler = get_string_sampler(n, 0)  # Assuming 0 is a valid sampler type
                    transition_array_sampler = get_transition_array_sampler(n, 0)  # Assuming 0 is a valid sampler type
                    initial_string = string_sampler()
                    transitions = transition_array_sampler(t, initial_string)
                    G, root, transitions, transition_history = generate_single_path_for_pipeline(
                        n=n,
                        t=t,
                        d=d,
                        string_sampler=string_sampler,
                        transition_array_sampler=transition_array_sampler
                    )
                    datasets.append((n, t, d, initial_string, transitions, transition_history))
    return datasets

def save_puzzles_and_solutions(datasets, base_dir):
    puzzles_dir = base_dir / "puzzles"
    solutions_dir = base_dir / "solutions"
    puzzles_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (n, t, d, initial_string, transitions, transition_history) in enumerate(datasets):
        markov_entropy = calculate_markov_entropy(transition_history)
        if len(transition_history) == 1:
            pattern_score = 1
        else:
            pattern_score = 1 - markov_entropy / np.log2(len(transition_history))

        puzzle = {
            "problem_id": f"{i:03d}",
            "initial_string": initial_string,
            "transitions": transitions,
            "markov_entropy": markov_entropy,
            "pattern_score": pattern_score
        }

        solution = {
            "problem_id": f"{i:03d}",
            "solution": transition_history
        }

        with open(puzzles_dir / f"{i:03d}.json", "w") as f:
            json.dump(puzzle, f, indent=4)
        with open(solutions_dir / f"{i:03d}.json", "w") as f:
            json.dump(solution, f, indent=4)

        results.append((i, n, t, d, markov_entropy, pattern_score))

    return results

def save_results_to_csv(results, filepath):
    df = pd.DataFrame(results, columns=['puzzle_id', 'n', 't', 'd', 'markov_entropy', 'pattern_score'])
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    base_dir = Path("SED_6300")
    datasets = generate_datasets()
    results = save_puzzles_and_solutions(datasets, base_dir)
    save_results_to_csv(results, base_dir / "exploration_results.csv")