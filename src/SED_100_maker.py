import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
import shutil

# Load dataset
df = pd.read_csv("SED_1000/analysis/exploration_results.csv")

# Create directory structure
base_dir = Path("SED_100")
analysis_dir = base_dir / "analysis"
data_dir = base_dir / "data"
puzzles_dir = data_dir / "puzzles"
solutions_dir = data_dir / "solutions"

for dir in [analysis_dir, puzzles_dir, solutions_dir]:
    dir.mkdir(parents=True, exist_ok=True)

# 1. Stratified Sampling (80%) - Preserves pattern_score distribution
bins = np.linspace(df["pattern_score"].min(), df["pattern_score"].max(), 20)  # Create bins for pattern_score
labels = range(len(bins)-1)
df["score_bin"] = pd.cut(df["pattern_score"], bins=bins, labels=labels)
stratified_sample = df.groupby("score_bin", group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=42))

# 2. Rare Points Sampling (20%) - Based on class frequencies
# Create classes of size 0.05
class_bins = np.arange(0, 1.05, 0.05)
df["class"] = pd.cut(df["pattern_score"], bins=class_bins)

# Calculate class frequencies and sort in ascending order (rarest first)
class_freq = df["class"].value_counts().sort_values()

# Sample rare points
rare_sample = pd.DataFrame()
remaining_data = df.drop(stratified_sample.index) # remove already sampled points
points_needed = 20

# Consider the 17 rarest classes first
rare_classes = class_freq.index[:17]

for class_name in rare_classes:
    class_data = remaining_data[remaining_data["class"] == class_name]
    if len(class_data) > 0:
        # Sample up to 2 points from each class
        n_sample = min(2, len(class_data))
        sampled = class_data.sample(n=n_sample, random_state=42)
        rare_sample = pd.concat([rare_sample, sampled])
        if len(rare_sample) >= points_needed:
            rare_sample = rare_sample.head(points_needed)
            break

# If not enough points are sampled, consider the next classes
if len(rare_sample) < points_needed:
    next_classes = class_freq.index[17:]
    for class_name in next_classes:
        class_data = remaining_data[remaining_data["class"] == class_name]
        if len(class_data) > 0:
            # Sample up to 2 points from each class
            n_sample = min(2, len(class_data))
            sampled = class_data.sample(n=n_sample, random_state=42)
            rare_sample = pd.concat([rare_sample, sampled])
            if len(rare_sample) >= points_needed:
                rare_sample = rare_sample.head(points_needed)
                break


# Create CSV files for analysis
# Overall dataset
common_mask = df["class"].isin(class_freq.index[-10:])  # Top 10 most frequent classes
rare_mask = df["class"].isin(class_freq.index[:10])     # Top 10 rarest classes

# Sampled dataset
final_sample = pd.concat([stratified_sample, rare_sample])
final_sample = final_sample.sample(n=100, random_state=42)

sampled_common_mask = final_sample["class"].isin(class_freq.index[-10:])
sampled_rare_mask = final_sample["class"].isin(class_freq.index[:10])

# Save analysis files
class_freq.to_csv(analysis_dir / "class_frequencies.csv")
df[rare_mask].to_csv(analysis_dir / "rare_points_overall.csv", index=False)
df[common_mask].to_csv(analysis_dir / "common_points_overall.csv", index=False)
final_sample[sampled_rare_mask].to_csv(analysis_dir / "rare_points_sampled.csv", index=False)
final_sample[sampled_common_mask].to_csv(analysis_dir / "common_points_sampled.csv", index=False)

# Save the selected 100 points
final_sample.to_csv(analysis_dir / "representative_sample.csv", index=False)
print("Representative dataset saved as representative_sample.csv")

# Copy (not move) the selected puzzles and solutions to the new directory
source_dir = Path("SED_1000")
for puzzle_id in final_sample["puzzle_id"]:
    # Source files in SED_1000
    puzzle_src = source_dir / "puzzles" / f"{puzzle_id:03d}.json"
    solution_src = source_dir / "solutions" / f"{puzzle_id:03d}.json"
    
    # Destination files in SED_100
    puzzle_dest = puzzles_dir / f"{puzzle_id:03d}.json"
    solution_dest = solutions_dir / f"{puzzle_id:03d}.json"
    
    # Copy files while preserving metadata
    if puzzle_src.exists():
        shutil.copy2(puzzle_src, puzzle_dest)  # copy2 preserves metadata
    else:
        print(f"Warning: Puzzle file not found: {puzzle_src}")
        
    if solution_src.exists():
        shutil.copy2(solution_src, solution_dest)  # copy2 preserves metadata
    else:
        print(f"Warning: Solution file not found: {solution_src}")
