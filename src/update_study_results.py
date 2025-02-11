from pathlib import Path
import pandas as pd
import json
import numpy as np

def get_pattern_score(puzzle_id: str, dataset_path: Path) -> float:
    """Get pattern score for a puzzle from dataset"""
    puzzle_file = dataset_path / "puzzles" / f"{int(puzzle_id):03d}.json"
    with open(puzzle_file, 'r') as f:
        puzzle_data = json.load(f)
    return puzzle_data['pattern_score']

def update_study_results(study_path: Path, dataset_path: Path):
    """Update study results with pattern scores and weighted accuracies"""
    study_path = Path(study_path)
    dataset_path = Path(dataset_path)
    
    # Process each experiment
    experiments_dir = study_path / "Experiments"
    all_exp_results = []
    
    for exp_dir in experiments_dir.glob("Experiment*"):
        exp_results = []
        
        # Process each run
        for run_dir in exp_dir.glob("Run*"):
            # Update run results with pattern scores
            run_results_path = run_dir / "results.csv"
            if run_results_path.exists():
                run_df = pd.read_csv(run_results_path)
                
                # Add pattern scores
                pattern_scores = [get_pattern_score(pid, dataset_path) for pid in run_df['puzzle_id']]
                run_df['pattern_score'] = pattern_scores
                
                # Calculate weighted accuracy for run
                weighted_acc = (run_df['is_valid'] * run_df['pattern_score']).mean()
                
                # Save updated run results
                run_df.to_csv(run_results_path, index=False)
                exp_results.append(weighted_acc)
        
        # Update experiment results
        exp_results_path = exp_dir / "results.csv"
        if exp_results_path.exists():
            exp_df = pd.read_csv(exp_results_path)
            exp_df['pattern_weighted_accuracy'] = exp_results
            exp_df.to_csv(exp_results_path, index=False)
            
            # Store overall experiment result
            all_exp_results.append({
                'experiment_id': int(exp_dir.name.replace('Experiment', '')),  # Convert to int
                'pattern_weighted_accuracy': np.mean(exp_results)
            })
    
    # Update study results
    results_dir = study_path / "Results"
    if results_dir.exists():
        study_results_path = results_dir / "Results.csv"
        if study_results_path.exists():
            study_df = pd.read_csv(study_results_path)
            
            # Ensure experiment_id is int in both DataFrames
            study_df['experiment_id'] = study_df['experiment_id'].astype(int)
            exp_results_df = pd.DataFrame(all_exp_results)
            exp_results_df['experiment_id'] = exp_results_df['experiment_id'].astype(int)
            
            # Add pattern weighted accuracies
            study_df = study_df.merge(
                exp_results_df,
                on='experiment_id',
                how='left'
            )
            
            # Save updated study results
            study_df.to_csv(study_results_path, index=False)

if __name__ == "__main__":
    # Example usage
    study_path = Path("Deliverables/Task2/SED_10_symbolic_results")
    dataset_path = Path("Deliverables/Task1/SED_10")
    
    update_study_results(study_path, dataset_path)
    print("Study results updated successfully!")