import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse

def plot_results(df: pd.DataFrame, output_dir: Path):
    """
    Generate visualization plots for the dataset
    
    Args:
        df: DataFrame with columns ['n', 't', 'd', 'markov_entropy', 'pattern_score']
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dt_ratios = df['d'] / df['t']
    df['dt_ratio'] = dt_ratios

    # Plot frequency distributions
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    sns.kdeplot(data=df['n'], fill=True)
    plt.xlabel('n')
    plt.ylabel('Density')
    plt.title('Frequency Distribution of n')
    
    plt.subplot(1, 4, 2)
    sns.kdeplot(data=df['t'], fill=True)
    plt.xlabel('t')
    plt.ylabel('Density')
    plt.title('Frequency Distribution of t')
    
    plt.subplot(1, 4, 3)
    sns.kdeplot(data=df['d'], fill=True)
    plt.xlabel('d')
    plt.ylabel('Density')
    plt.title('Frequency Distribution of d')
    
    plt.subplot(1, 4, 4)
    sns.kdeplot(data=df['dt_ratio'], fill=True)
    plt.xlabel('d/t ratio')
    plt.ylabel('Density')
    plt.title('Frequency Distribution of d/t')
    
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distributions.png")
    plt.close()

    # Plot average markov_entropy vs n, t, d
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    df.groupby('n')['markov_entropy'].mean().plot(kind='line')
    plt.xlabel('n')
    plt.ylabel('Markov Entropy')
    plt.title('Average Markov Entropy vs n')

    plt.subplot(1, 3, 2)
    df.groupby('t')['markov_entropy'].mean().plot(kind='line')
    plt.xlabel('t')
    plt.ylabel('Markov Entropy')
    plt.title('Average Markov Entropy vs t')

    plt.subplot(1, 3, 3)
    df.groupby('d')['markov_entropy'].mean().plot(kind='line')
    plt.xlabel('d')
    plt.ylabel('Markov Entropy')
    plt.title('Average Markov Entropy vs d')

    plt.tight_layout()
    plt.savefig(output_dir / "average_markov_entropy_vs_ntd.png")
    plt.close()

    # Add distribution plots for entropy and pattern scores
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df['markov_entropy'], fill=True)
    plt.xlabel('Markov Entropy')
    plt.ylabel('Density')
    plt.title('Frequency Distribution of Markov Entropy')
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df['pattern_score'], fill=True)
    plt.xlabel('Pattern Score')
    plt.ylabel('Density')
    plt.title('Frequency Distribution of Pattern Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png")
    plt.close()

    # Plot average pattern_score vs n, t, d
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    df.groupby('n')['pattern_score'].mean().plot(kind='line')
    plt.xlabel('n')
    plt.ylabel('Pattern Score')
    plt.title('Average Pattern Score vs n')

    plt.subplot(1, 3, 2)
    df.groupby('t')['pattern_score'].mean().plot(kind='line')
    plt.xlabel('t')
    plt.ylabel('Pattern Score')
    plt.title('Average Pattern Score vs t')

    plt.subplot(1, 3, 3)
    df.groupby('d')['pattern_score'].mean().plot(kind='line')
    plt.xlabel('d')
    plt.ylabel('Pattern Score')
    plt.title('Average Pattern Score vs d')

    plt.tight_layout()
    plt.savefig(output_dir / "average_pattern_score_vs_ntd.png")
    plt.close()

    # Plot per-n visualizations
    for n in df['n'].unique():
        subset = df[df['n'] == n]
        if subset.empty:
            continue

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        subset.groupby('t')['pattern_score'].mean().plot(kind='line')
        plt.xlabel('t')
        plt.ylabel('Pattern Score')
        plt.title(f'Average Pattern Score vs t for n={n}')

        plt.subplot(1, 2, 2)
        subset.groupby('d')['pattern_score'].mean().plot(kind='line')
        plt.xlabel('d')
        plt.ylabel('Pattern Score')
        plt.title(f'Average Pattern Score vs d for n={n}')

        plt.tight_layout()
        plt.savefig(output_dir / f"average_pattern_score_vs_td_n{n}.png")
        plt.close()

def main():
    # Set input and output paths
    input_path = "SED_100/analysis/representative_sample.csv"
    output_dir = "SED_100/analysis/plots/representative_sample"

    # Load and process data
    df = pd.read_csv(input_path)
    plot_results(df, Path(output_dir))
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    main()