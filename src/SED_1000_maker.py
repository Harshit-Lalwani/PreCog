import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from pipeline_utils import *
from puzzle_generator import *
from add_explanation import *

def generate_dataset(size = 1000):
    datasets = []
    # Generate samples

    for i in range(size):
        # Sample n from Gaussian, bounded between 

        n_min = 1
        n_max = 4
        n_ave = (n_min + n_max) / 2
    
        n = int(round(np.random.normal(n_ave, 2)))
        n = max(n_min, min(n_max, n))

        t_min = 1
        t_max = 4
        t_ave = (t_min + t_max) / 2
        
        # Sample t from Gaussian, bounded between 1 and 7
        t = int(round(np.random.normal(t_ave, 2)))
        t = max(t_min, min(t_max, t))

        d_min = 1
        d_max = 4
        d_ave = (d_min + d_max) / 2
        
        # Sample d from Gaussian between t and 4*t
        d = int(round(np.random.normal(d_ave, 2)))
        d = max(int(d_min), min(int(d_max), d))  # bound between t and 4*t
        
        G, root, transitions, transition_history = generate_single_path(
            n=n,
            t=t,
            d=d,
        )
        datasets.append((n, t, d, root, transitions, transition_history))
    
    return datasets

def save_puzzles_and_solutions(datasets, base_dir, start_id=0):
    """
    Save puzzles and solutions with customizable starting ID
    
    Args:
        datasets: List of generated puzzle data
        base_dir: Base directory to save files
        start_id: Starting puzzle ID (default: 0)
    """
    puzzles_dir = base_dir / "puzzles"
    solutions_dir = base_dir / "solutions"
    puzzles_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (n, t, d, initial_string, transitions, transition_history) in enumerate(datasets, start=start_id):
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

def process_llm_response(self, response_text: str, puzzle: Dict) -> Dict:
    """Process LLM response and validate solution"""
    error_log = self.study_dir / "validation_errors.log"
    
    try:
        # Parse response using fallback parser
        parsed_response = parse_solution_with_fallback(response_text)
        
        # Additional validation for empty solution list
        if not parsed_response.solutions:
            raise ValueError(f"Empty solution list for puzzle {puzzle['problem_id']}")
            
        # Validate solution sequence    
        is_valid = validate_solution_sequence(
            puzzle['initial_string'],
            puzzle['transitions'],
            parsed_response.solutions[0].solution
        )
        
        return {
            'puzzle_id': puzzle['problem_id'],
            'is_valid': int(is_valid),  # Convert bool to 0/1
            'pattern_score': puzzle['pattern_score']
        }
        
    except Exception as e:
        # Log error details
        error_msg = (f"Validation error for puzzle {puzzle['problem_id']}\n"
                    f"Error: {str(e)}\n"
                    f"Response text: {response_text}\n"
                    "----------------------------------------\n")
        with open(error_log, "a") as f:
            f.write(error_msg)
            
        # Return invalid result on any error
        return {
            'puzzle_id': puzzle['problem_id'],
            'is_valid': 0,
            'pattern_score': puzzle['pattern_score']
        }

if __name__ == "__main__":
    base_dir = Path("MIX_3_3_5_SED_10")
    start_puzzle_id = 41
    datasets = generate_dataset(10)
    results = save_puzzles_and_solutions(datasets, base_dir, start_id=start_puzzle_id)
    save_results_to_csv(results, base_dir / "exploration_results.csv")
