import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_distribution_analysis(csv_path: str | Path):
    """Create distribution analysis of pattern scores"""
    # Read the dataset
    df = pd.read_csv(csv_path)
    
    # Create classes of size 0.05
    class_bins = np.arange(0, 1.05, 0.05)
    df["class"] = pd.cut(df["pattern_score"], 
                        bins=class_bins, 
                        labels=[f"{i:.2f}-{i+0.05:.2f}" for i in class_bins[:-1]])
    
    # Calculate class frequencies
    class_freq = df["class"].value_counts().sort_index()
    
    # Create bar plot
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(x=class_freq.index, y=class_freq.values)
    
    # Customize plot
    # plt.title("Distribution of Pattern Scores in SED-1000", fontsize=14)
    plt.xlabel("Pattern Score Range", fontsize=12)
    plt.ylabel("Number of Examples", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(class_freq.values):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    
    # Create frequency table
    freq_table = pd.DataFrame({
        'Score Range': class_freq.index,
        'Count': class_freq.values,
        'Percentage': (class_freq.values / len(df) * 100).round(2)
    })
    
    # Save frequency table
    freq_table.to_csv("class_frequencies_table.csv", index=False)
    
    # Print table
    print("\nClass Distribution:")
    print(freq_table.to_string(index=False))
    
    return freq_table

if __name__ == "__main__":
    # Analyze SED-100 dataset
    csv_path = Path("LESS_SED_1000/exploration_results.csv")
    freq_table = create_distribution_analysis(csv_path)