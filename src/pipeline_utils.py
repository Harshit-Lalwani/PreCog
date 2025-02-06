import itertools
import os
from pathlib import Path
import json
import pandas as pd
from utils import *
from parse_utils import *
from typing import List, Dict, Any, Callable

def generate_experiment_id(params_indices: List[int], num_params: int) -> str:
    """Convert list of parameter indices to experiment ID with leading zeros"""
    return ''.join(str(i).zfill(1) for i in params_indices)

def create_directory_structure(base_dir: Path, experiment_id: str, run_id: int) -> Dict[str, Path]:
    """Create experiment and run directory structure"""
    exp_dir = base_dir / f"Experiment{experiment_id}"
    run_dir = exp_dir / f"Run{run_id}"
    
    paths = {
        'train_puzzles': run_dir / "Data/Train/puzzles",
        'train_solutions': run_dir / "Data/Train/solutions",
        'test_puzzles': run_dir / "Data/Test/puzzles",
        'test_solutions': run_dir / "Data/Test/solutions",
        'predictions': run_dir / "Data/Test/Predictions",
        'run_results': run_dir,
        'exp_results': exp_dir
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        
    return paths

def get_string_sampler(n: int) -> Callable:
    """Returns a function that samples strings of length n"""
    def sampler():
        return sample_random_string(n)
    return sampler

def get_transition_sampler(n: int) -> Callable:
    """Returns a function that creates transitions for strings of length n"""
    def sampler():
        return create_transition(n)
    return sampler

def generate_dataset(count: int, t: int, d: int, 
                    M: int,  # Added n parameter
                    output_dir: Path) -> List[Dict]:
    """Generate dataset using n-based samplers"""
    
    # Create sampler functions
    n = sample_gaussian_n(M)
    string_sampler = get_string_sampler(n)
    transition_sampler = get_transition_sampler(n)
    
    puzzles = []
    for i in range(count):
        puzzle = {
            'problem_id': f"{i:03d}",
            'initial_string': string_sampler(),
            'transitions': [transition_sampler() for _ in range(t)]
        }
        
        # Save puzzle
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{puzzle['problem_id']}.json", 'w') as f:
            json.dump(puzzle, f, indent=4)
            
        puzzles.append(puzzle)
    return puzzles

def save_run_results(results_path: Path, predictions: List[Dict], params: Dict) -> None:
    """Save results for a single run"""
    # Ensure directory exists
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame and save
    df = pd.DataFrame(predictions)
    df['run_id'] = params['run_id']
    for k, v in params.items():
        if k != 'run_id':
            df[k] = str(v)
            
    output_file = results_path / f"Results_{params['run_id']}.csv"
    df.to_csv(output_file, index=False)

def save_experiment_results(exp_dir: Path, all_runs: List[Dict], params: Dict) -> None:
    """Save aggregated results for an experiment"""
    # Ensure experiment directory exists
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with results
    df = pd.DataFrame(all_runs)
    df['experiment_id'] = params['experiment_id']
    for k, v in params.items():
        if k != 'experiment_id':
            df[k] = str(v)
    
    df['accuracy'] = df['valid_predictions'] / df['total_predictions']
    
    # Save to proper path
    results_file = exp_dir / f"Results_{params['experiment_id']}.csv"
    df.to_csv(results_file, index=False)

def save_study_results(study_dir: Path, all_experiments: List[Dict]) -> None:
    """Save overall study results"""
    # Create Results directory
    results_dir = study_dir / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    df = pd.DataFrame(all_experiments)
    df['self_consistency'] = df.apply(calculate_self_consistency, axis=1, study_dir=study_dir)
    df.to_csv(results_dir / "Results.csv", index=False)

def calculate_self_consistency(row: pd.Series, study_dir: Path) -> float:
    """Calculate self-consistency score for an experiment using the formula:
    1 - (std dev of accuracies / mean accuracy)
    
    A higher score indicates more consistent performance across runs.
    Returns 1 if mean accuracy is 0 to avoid division by zero.
    """
    # Load the experiment's results file to get all run accuracies
    exp_dir = study_dir / "Experiments" / f"Experiment{row['experiment_id']}"
    exp_results = pd.read_csv(exp_dir / f"Results_{row['experiment_id']}.csv")
    
    accuracies = exp_results['valid_predictions'] / exp_results['total_predictions']
    mean_accuracy = accuracies.mean()
    
    if mean_accuracy == 0:
        return 1.0
        
    std_accuracy = accuracies.std()
    return 1.0 - (std_accuracy / mean_accuracy)

def parse_json_solutions(solutions_str: str) -> Solutions:
    """Parse JSON formatted solution string into Solutions object"""
    try:
        # Remove any extra quotes that might wrap the JSON string
        solutions_str = solutions_str.strip('"\'')
        
        # Parse JSON
        data = json.loads(solutions_str)
        
        # Extract solutions array
        if 'solutions' in data:
            found_solutions = []
            for sol in data['solutions']:
                found_solutions.append(
                    Solution(
                        problem_id=sol['problem_id'],
                        solution=sol['solution']
                    )
                )
            return Solutions(solutions=found_solutions)
        raise ValueError("No 'solutions' key in JSON")
        
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to parse JSON solution: {e}")

def parse_solution_with_fallback(solution_text: str) -> Solutions:
    """Try JSON parse first, fallback to regex parse"""
    try:
        return parse_json_solutions(solution_text)
    except ValueError:
        return parse_solutions_string(solution_text)
