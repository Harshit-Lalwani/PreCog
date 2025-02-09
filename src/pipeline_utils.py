import itertools
import os
from pathlib import Path
import json
import pandas as pd
from utils import *
from parse_utils import *
from typing import List, Dict, Any, Callable, Tuple
from enum import Enum
import networkx as nx
import random
import numpy as np

class StringSamplerType(Enum):
    GAUSSIAN_simple = 0
    GAUSSIAN_shuffled = 1

class TransitionArraySamplerType(Enum):
    STANDARD = 0      # Independent transitions
    SHUFFLED = 1     # Shuffled transitions

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

def get_string_sampler(n: int, sampler_type: int) -> Callable:
    """Returns a function that generates strings based on specified approach"""
    if sampler_type == StringSamplerType.GAUSSIAN_simple.value:
        def sampler():
            return sample_random_string(n)
        return sampler
    elif sampler_type == StringSamplerType.GAUSSIAN_shuffled.value:
        def sampler():
            s = sample_random_string(n)
            # for each unique character in the string, sample a random letter from the alphabet. In new string, replace all instances of the unique character with the sampled letter
            unique_chars = list(set(s))
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            for char in unique_chars:
                new_char = random.choice(alphabet)
                s = s.replace(char, new_char)
            return s

        return sampler

def get_transition_array_sampler(n: int, sampler_type: int) -> Callable:
    """Returns a function that generates transition arrays based on specified approach"""
    if sampler_type == TransitionArraySamplerType.STANDARD.value:
        def sampler(t: int, initial_string: str):  # Modified to accept initial_string
            return create_transitions_array(n, t, initial_string)
        return sampler
        
    elif sampler_type == TransitionArraySamplerType.SHUFFLED.value:
        def sampler(t: int, initial_string: str):  # Modified to accept initial_string
            T = create_transitions_array(n, t, initial_string)
            random.shuffle(T)
            return T
        return sampler
    else:
        raise ValueError(f"Unknown transition array sampler type: {sampler_type}")

# ! this generate_dataset isn't used in the pipeline
# def generate_dataset(count: int, t: int, d: int, 
#                     M: int,  # Added n parameter
#                     output_dir: Path) -> List[Dict]:
#     """Generate dataset using n-based samplers"""
    
#     # Create sampler functions
#     n = sample_gaussian_n(M)
#     string_sampler = get_string_sampler(n)
#     transition_sampler = get_transition_sampler(n)
    
#     puzzles = []
#     for i in range(count):
#         puzzle = {
#             'problem_id': f"{i:03d}",
#             'initial_string': string_sampler(),
#             'transitions': [transition_sampler() for _ in range(t)],
#         }
        
#         # Save puzzle
#         output_dir.mkdir(parents=True, exist_ok=True)
#         with open(output_dir / f"{puzzle['problem_id']}.json", 'w') as f:
#             json.dump(puzzle, f, indent=4)
            
#         puzzles.append(puzzle)
#     return puzzles

def save_run_results(results_path: Path, predictions: List[Dict], run_params: Dict) -> pd.DataFrame:
    """Save run results to CSV and return DataFrame"""
    # Create basic results DataFrame with additional metrics
    basic_results = []
    for p in predictions:
        # Load original puzzle from Data directory using dataset_id and run_id
        dataset_id = run_params['dataset_id']
        run_id = run_params['run_id']
        
        # Fixed path to include Study directory
        puzzle_file = (results_path.parent.parent.parent / "Data" / 
                      f"Data_{dataset_id}" / f"Run{run_id}" / 
                      f"{dataset_id}_{run_id}" / "test" / "puzzles" / 
                      f"{p['puzzle_id']}.json")
        
        # Load puzzle to get the metrics
        with open(puzzle_file) as f:
            puzzle = json.load(f)
            
        result = {
            'problem_id': p['puzzle_id'],
            'is_valid': p['is_valid'],
            'predicted_solution': p['predicted_solution'],
            'markov_entropy': puzzle['markov_entropy'],
            'pattern_score': puzzle['pattern_score']
        }
        basic_results.append(result)
    
    basic_results_df = pd.DataFrame(basic_results)
    
    # Save basic results CSV in run folder
    run_results_file = results_path / "predictions.csv"
    basic_results_df.to_csv(run_results_file, index=False)
    
    # Create full results DataFrame with all parameters
    df = pd.DataFrame(basic_results)
    
    # Calculate accuracy for this run
    valid_count = sum(p['is_valid'] for p in predictions)
    total_count = len(predictions)
    accuracy = valid_count / total_count if total_count > 0 else 0
    
    # Add run parameters as columns
    for k, v in run_params.items():
        df[k] = str(v)
    
    df['accuracy'] = accuracy
    df['valid_predictions'] = valid_count
    df['total_predictions'] = total_count
    
    return df

def save_experiment_results(exp_dir: Path, all_runs: List[Dict], params: Dict) -> None:
    """Save aggregated results for an experiment"""
    # Ensure experiment directory exists
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with results
    df = pd.DataFrame(all_runs)
    
    # Add all parameters including experiment_id
    for k, v in params.items():
        df[k] = str(v)
    
    df['accuracy'] = df['valid_predictions'] / df['total_predictions']
    
    # Save to proper path
    results_file = exp_dir / f"Results_{params['experiment_id']}.csv"
    df.to_csv(results_file, index=False)

def save_study_results(study_dir: Path, all_experiments: List[Dict]) -> None:
    """Save aggregated results in a single CSV file"""
    results_dir = study_dir / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Process each experiment
    for exp in all_experiments:
        exp_dir = study_dir / "Experiments" / f"Experiment{exp['experiment_id']}"
        num_runs = exp.get('num_runs', 1)
        
        # Collect accuracies for all runs
        run_accuracies = []
        for run_id in range(num_runs):
            run_dir = exp_dir / f"Run{run_id}"
            pred_files = list((run_dir / "Output").glob("*.json"))
            
            if pred_files:
                valid_count = sum(1 for f in pred_files if json.loads(open(f).read())['is_valid'])
                total_count = len(pred_files)
                run_accuracy = valid_count / total_count if total_count > 0 else 0
                run_accuracies.append(run_accuracy)
        
        # Calculate experiment-level metrics
        experiment_accuracy = sum(run_accuracies) / len(run_accuracies) if run_accuracies else 0
        std_accuracy = np.std(run_accuracies) if len(run_accuracies) > 1 else 0
        self_consistency = 1.0 - (std_accuracy / experiment_accuracy) if experiment_accuracy > 0 else 1.0
        
        # Create a result row for each run
        for run_id, run_accuracy in enumerate(run_accuracies):
            result = {
                'experiment_id': exp['experiment_id'],
                'run_id': run_id,
                'run_accuracy': run_accuracy,
                'experiment_accuracy': experiment_accuracy,
                'self_consistency': self_consistency,
                'train_count': exp['train_count'],
                'train_t': exp['train_t'],
                'train_d': exp['train_d'],
                'test_count': exp['test_count'],
                'test_t': exp['test_t'],
                'test_d': exp['test_d'],
                'M': exp['M'],
                'string_sampler_type': StringSamplerType(exp['string_sampler_types']).name,
                'transition_array_type': TransitionArraySamplerType(exp['transition_array_sampler_types']).name,
                'prompt_title': exp['prompt_title'],  # Store only the title
                'model': exp['model'],
                'dataset_id': exp['dataset_id']
            }
            all_results.append(result)
    
    # Create DataFrame and save
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_dir / "results.csv", index=False)

def calculate_self_consistency(row: pd.Series, study_dir: Path) -> float:
    """Calculate self-consistency score for an experiment"""
    exp_dir = study_dir / "Experiments" / f"Experiment{row['experiment_id']}"
    results_file = exp_dir / "results.csv"  # Look for file directly in experiment dir
    exp_results = pd.read_csv(results_file)
    
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

def generate_single_path_for_pipeline(n: int, t: int, d: int, 
                                    string_sampler: Callable,
                                    transition_array_sampler: Callable) -> Tuple[nx.DiGraph, str, List[Dict], List[int]]:
    """
    Pipeline-specific version of generate_single_path that uses provided samplers
    
    Args:
        n: Length of strings
        t: Number of transitions
        d: Maximum path depth
        string_sampler: Function to generate initial string
        transition_array_sampler: Function to generate array of transitions
    """
    # Get root string first
    root = string_sampler()
    
    # Initialize graph
    G = nx.DiGraph()
    G.add_node((-1,), string=root)
    current_node = (-1,)
    current_string = root
    
    # Get transitions using initial string
    transitions = transition_array_sampler(t, root)  # Modified to pass root string
    
    transition_history = []
    
    # Build path through graph
    for i in range(d):
        j = random.choice(range(t))
        transition = transitions[j]
        new_string = apply_transition(current_string, transition)
        
        if new_string != "-1":
            transition_history.append(j)
            new_node = tuple(list(current_node) + [i])
            G.add_node(new_node, string=new_string)
            G.add_edge(current_node, new_node, label=f"{transition['src']} -> {transition['tgt']}")
            current_node = new_node
            current_string = new_string
    
    # Add final empty transition
    transitions.append({"src": current_string, "tgt": ""})
    transition_history.append(t)
    
    return G, root, transitions, transition_history

def calculate_markov_entropy(sequence: List[int]) -> float:
    """Calculate the Markov entropy of a sequence of transitions."""
    if not sequence:
        return 0.0

    transition_counts = {}
    for i in range(len(sequence) - 1):
        pair = (sequence[i], sequence[i + 1])
        if pair not in transition_counts:
            transition_counts[pair] = 0
        transition_counts[pair] += 1

    total_transitions = sum(transition_counts.values())
    entropy = 0.0
    for count in transition_counts.values():
        probability = count / total_transitions
        entropy -= probability * np.log2(probability)

    return entropy
