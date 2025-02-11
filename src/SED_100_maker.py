import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
import shutil

# Load dataset and filter out first 100 problems
df = pd.read_csv("SED_1000/exploration_results.csv")
df = df[df["puzzle_id"] >= 100]  # Filter out first 100 problems

# Create directory structure
base_dir = Path("SED_50")
analysis_dir = base_dir / "analysis"
data_dir = base_dir / "data"
puzzles_dir = data_dir / "puzzles"
solutions_dir = data_dir / "solutions"

for dir in [analysis_dir, puzzles_dir, solutions_dir]:
    dir.mkdir(parents=True, exist_ok=True)

# 1. Common Points Sampling (20 points)
bins = np.linspace(df["pattern_score"].min(), df["pattern_score"].max(), 20)
labels = range(len(bins)-1)
df["score_bin"] = pd.cut(df["pattern_score"], bins=bins, labels=labels)
stratified_sample = df.groupby("score_bin", group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=42))
common_sample = stratified_sample.sample(n=20, random_state=42)  # Select 20 common points

# 2. Rare Points Sampling (10 points) - One from each of the 10 rarest classes
class_bins = np.arange(0, 1.05, 0.05)
df["class"] = pd.cut(df["pattern_score"], bins=class_bins)

# Calculate class frequencies and get 10 rarest classes
class_freq = df["class"].value_counts().sort_values()
rare_classes = class_freq.index[:10]  # Get 10 rarest classes

# Sample one point from each rare class
rare_sample = pd.DataFrame()
remaining_data = df.drop(common_sample.index)

for class_name in rare_classes:
    class_data = remaining_data[remaining_data["class"] == class_name]
    if len(class_data) > 0:
        sampled = class_data.sample(n=1, random_state=42)  # Take exactly one sample
        rare_sample = pd.concat([rare_sample, sampled])

# Combine samples
final_sample = pd.concat([common_sample, rare_sample])

# Create CSV files for analysis
# Overall dataset
common_mask = df["class"].isin(class_freq.index[-10:])  # Top 10 most frequent classes
rare_mask = df["class"].isin(class_freq.index[:10])     # Top 10 rarest classes

# Sampled dataset
sampled_common_mask = final_sample["class"].isin(class_freq.index[-10:])
sampled_rare_mask = final_sample["class"].isin(class_freq.index[:10])

# Save analysis files
class_freq.to_csv(analysis_dir / "class_frequencies.csv")
df[rare_mask].to_csv(analysis_dir / "rare_points_overall.csv", index=False)
df[common_mask].to_csv(analysis_dir / "common_points_overall.csv", index=False)
final_sample[sampled_rare_mask].to_csv(analysis_dir / "rare_points_sampled.csv", index=False)
final_sample[sampled_common_mask].to_csv(analysis_dir / "common_points_sampled.csv", index=False)

# Save the selected 30 points
final_sample.to_csv(analysis_dir / "representative_sample.csv", index=False)
print("Representative dataset saved as representative_sample.csv")

# Copy selected puzzles and solutions
source_dir = Path("SED_1000")
for puzzle_id in final_sample["puzzle_id"]:
    puzzle_src = source_dir / "puzzles" / f"{puzzle_id:03d}.json"
    solution_src = source_dir / "solutions" / f"{puzzle_id:03d}.json"
    
    puzzle_dest = puzzles_dir / f"{puzzle_id:03d}.json"
    solution_dest = solutions_dir / f"{puzzle_id:03d}.json"
    
    if puzzle_src.exists():
        shutil.copy2(puzzle_src, puzzle_dest)
    else:
        print(f"Warning: Puzzle file not found: {puzzle_src}")
        
    if solution_src.exists():
        shutil.copy2(solution_src, solution_dest)
    else:
        print(f"Warning: Solution file not found: {solution_src}")

print(f"Total samples: {len(final_sample)}")
print(f"Common samples: {len(common_sample)}")
print(f"Rare samples: {len(rare_sample)}")
