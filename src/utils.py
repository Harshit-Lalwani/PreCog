import numpy as np
import pickle
import random
import networkx as nx
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import json
from solver.solver_utils import *
from schema import Solution, Solutions
import os
import csv
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Callable, Tuple

M = 10

# ! handle M else where

def sample_gaussian_n(mean=None, std_dev=None, M=M):
    min_n = 1
    max_n = M
    if(mean==None):
        mean = max_n/2
    if (std_dev == None):
        std_dev = max_n/4
    n = int(np.clip(np.random.normal(mean, std_dev), min_n, max_n))
    return n

# Function to load the flat dataset
def load_dataset(file_path='data/NIS/NISdb_flat.pkl'):
    with open(file_path, 'rb') as f:
        flat_dataset = pickle.load(f)
    return flat_dataset

# Function to Sample a random string of size n from NISdb_flat
def sample_random_string(n = sample_gaussian_n, file_path='data/NIS/NISdb_flat.pkl'):
    flat_dataset = load_dataset(file_path)
    
    if n not in flat_dataset or not flat_dataset[n]:
        raise ValueError(f"No strings of length {n} found in the dataset.")
    
    return random.choice(flat_dataset[n])

# Function to create transitions
def create_transition(n: int, initial_string: str, file_path='data/NIS/NISdb_flat.pkl'):
    """Create a transition using characters from initial string's ASCII range
    
    Args:
        n: Integer representing size parameter
        initial_string: String to base transitions on
        file_path: Path to dataset (unused)
        
    Returns:
        Dict with 'src' and 'tgt' strings within ASCII range
    """
    Max_size = min(2*n, M)
    
    # Get ASCII range from initial string
    ascii_vals = [ord(c) for c in initial_string]
    min_ascii = min(ascii_vals)
    max_ascii = max(ascii_vals)
    
    # Create extended range for target string (up to +2)
    extended_range = list(range(min_ascii, max_ascii + 3))
    
    while True:
        # Generate random lengths using exponential distribution
        p = int(np.clip(np.random.exponential(scale=n), 1, Max_size))
        q = int(np.clip(np.random.exponential(scale=n), 1, Max_size))
        
        # Generate source string using original ASCII range
        s1_chars = [chr(random.randint(min_ascii, max_ascii)) for _ in range(p)]
        s1 = ''.join(s1_chars)
        
        # Generate target string using extended ASCII range
        s2_chars = [chr(random.choice(extended_range)) for _ in range(q)]
        s2 = ''.join(s2_chars)
        
        # Validate transition
        if s1 != s2: 
            break
            
    return {"src": s1, "tgt": s2}

# Function to create an array of transitions of size t for a given n
def create_transitions_array(n: int, t: int, initial_string: str, file_path='data/NIS/NISdb_flat.pkl'):
    transitions = []
    for _ in range(t):
        transitions.append(create_transition(n, initial_string, file_path))
    return transitions

# To apply a transition [s1, s2] to a string s, we find the first occurrence of s1 in s and replace it with s2:

# Function to apply a transition [s1, s2] to a string s
def apply_transition(s, transition):
    s1, s2 = transition["src"], transition["tgt"]
    if s1 not in s:
        return "-1"
    return s.replace(s1, s2, 1)

# Function to plot the graph
def plot_graph(G):
    pos = nx.shell_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'string'), node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.show()
    
    
def generate_graph(n=sample_gaussian_n(), t=3, d=3, file_path='data/NIS/NISdb_flat.pkl'):
    root = sample_random_string(n, file_path)
    transitions = create_transitions_array(n, t, root, file_path)  # Pass root string
    
    G = nx.DiGraph()
    G.add_node((-1,), string=root)
    current_level = [((-1,), root)]
    
    for depth in range(d):
        next_level = []
        for node, node_string in current_level:
            for i, transition in enumerate(transitions):
                new_node_string = apply_transition(node_string, transition)
                if new_node_string != "-1":
                    new_node = tuple(list(node) + [i])
                    G.add_node(new_node, string=new_node_string)
                    G.add_edge(node, new_node, label=f"{transition['src']} -> {transition['tgt']}")
                    next_level.append((new_node, new_node_string))
        current_level = next_level
    
    return G

def log_conversation(engine, prompt, response):
    logging.info(f"Engine: {engine}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Response: {response}")
    
def user_prompt_generator(train_dir: Path, test_dir: Path, give_explanation_flag=0) -> str:
    """Generate prompts of the form:
    Training:
    Puzzle:{problem_id, initial_string, transitions}
    Solution: {problem_id, solution}
    Test:
    Puzzle:{problem_id, initial_string, transitions}
   
    Args:
        train_path (Path): has subfolders puzzles and solutions with json files <id>.json
        test_path (Path): has subfolder puzzles with json files <id>.json
        give_explanation_flag (int): whether to include explanations in training examples
    """
    
    train_path = Path(train_dir)
    test_path = Path(test_dir)

    train_puzzles_dir = train_path / "puzzles"
    train_solutions_dir = train_path / "solutions"
    test_puzzles_dir = test_path / "puzzles"

    training_data = []
    test_data = []

    # Collect training puzzles and solutions
    if train_puzzles_dir.exists() and train_solutions_dir.exists():
        train_puzzle_files = sorted(train_puzzles_dir.glob("*.json"))
        for puzzle_file in train_puzzle_files:
            problem_id = puzzle_file.stem
            solution_file = train_solutions_dir / f"{problem_id}.json"
            if solution_file.exists():
                with open(puzzle_file, "r") as pf, open(solution_file, "r") as sf:
                    puzzle_content = json.load(pf)
                    solution_content = json.load(sf)
                    
                    # Remove metrics from puzzle content
                    filtered_puzzle = {
                        'problem_id': puzzle_content['problem_id'],
                        'initial_string': puzzle_content['initial_string'],
                        'transitions': puzzle_content['transitions']
                    }
                                     
                    if give_explanation_flag and 'explanation' in solution_content:
                        training_example = {
                        "Puzzle": filtered_puzzle,
                        "Solution": {
                            'problem_id': solution_content['problem_id'],
                            'solution': solution_content['solution'],
                            'explanation': solution_content['explanation']
                        }
                    }
                    else:
                        training_example = {
                        "Puzzle": filtered_puzzle,
                        "Solution": {
                            'problem_id': solution_content['problem_id'],
                            'solution': solution_content['solution']
                        }
                    }

                    
                    training_data.append(training_example)

    # Collect test puzzles
    if test_puzzles_dir.exists():
        test_puzzle_files = sorted(test_puzzles_dir.glob("*.json"))
        for puzzle_file in test_puzzle_files:
            with open(puzzle_file, "r") as pf:
                puzzle_content = json.load(pf)
                # Remove metrics from puzzle content
                filtered_puzzle = {
                    'problem_id': puzzle_content['problem_id'],
                    'initial_string': puzzle_content['initial_string'],
                    'transitions': puzzle_content['transitions']
                }
                test_data.append({"Puzzle": filtered_puzzle})

    # Format the prompt
    
    user_prompt = (
        "Training:\n"
        f"{training_data}\n"
        "```\n"
        "Test:\n"
        f"{test_data}\n"
        "```"
    )
    
    return user_prompt

def validate_solution_sequence(initial_string: str, transitions: list, solution_steps: list) -> int:
    """
    Apply transitions in given order to initial string. 
    Return 1 if empty string reached, 0 otherwise.
    """
    current = initial_string
    
    for step in solution_steps:
        if step >= len(transitions):
            return 0
        
        result = apply_transition(current, transitions[step])
        if result == "-1":  # transition couldn't be applied
            return 0
            
        current = result
        
    return 1 if current == '' else 0

def store_output(
    output_id: str,
    engine: str,
    system_prompt: str,
    train_path: str,
    test_path: str,
    solutions_obj: Solutions
):
    """
    Saves a dictionary into /root/PreCog/data/Outputs/output_<output_id>.txt with fields:
      engine, system_prompt, train_set, test_set, solutions
    Where 'solutions' is itself a dictionary of the format:
      {
        output_id, system_prompt, model, train_set, test_set, score, accuracy, solutions
      }
    and the solutions list is of dictionaries {id, solution, is_valid}.

    Also appends a line to /root/PreCog/data/Outputs/../../Results.csv with columns:
      output_id, engine, system_prompt, train_set, test_set, t, d, score, accuracy, remarks (empty).
    Note: The first and second numbers in the name of the training dataset folder represent t and d.
    """

    validated_solutions = []
    test_puzzles_dir = Path(test_path) / "puzzles"
    
    for sol in solutions_obj.solutions:
        try:
            # Read puzzle file
            puzzle_file = test_puzzles_dir / f"{sol.problem_id}.json"
            with open(puzzle_file, "r") as pf:
                puzzle_data = json.load(pf)
                
            # Validate using our new function
            is_valid = validate_solution_sequence(
                initial_string=puzzle_data["initial_string"],
                transitions=[{"src": t["src"], "tgt": t["tgt"]} for t in puzzle_data["transitions"]],
                solution_steps=sol.solution
            )
        except FileNotFoundError:
            logging.warning(f"Puzzle file not found: {puzzle_file}. Marking solution as invalid.")
            is_valid = 0
        except Exception as e:
            logging.warning(f"Error processing puzzle {sol.problem_id}: {str(e)}. Marking solution as invalid.")
            is_valid = 0
            
        validated_solutions.append({
            "problem_id": sol.problem_id,
            "solution": sol.solution,
            "is_valid": is_valid
        })
    

    total = len(validated_solutions)
    correct = sum( vs["is_valid"] for vs in validated_solutions )
    accuracy = correct / total if total > 0 else 0

    train_folder = os.path.basename(os.path.normpath(train_path))
    test_folder = os.path.basename(os.path.normpath(test_path))

    parts = train_folder.split('_')
    t_val, d_val = "0", "0"
    if len(parts) >= 3:
        if parts[1].isdigit():
            t_val = parts[1]
        if parts[2].isdigit():
            d_val = parts[2]

    data_to_store = {
        "engine": engine,
        "system_prompt": system_prompt,
        "train_set": train_path,
        "test_set": test_path,
        "solutions": {
            "output_id": output_id,
            "system_prompt": system_prompt,
            "model": engine,
            "train_set": train_path,
            "test_set": test_path,
            "score": 0,
            "accuracy": accuracy,
            "solutions": validated_solutions
        }
    }

    output_dir = Path("/root/PreCog/data/Outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"output_{output_id}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_to_store, f, indent=2)

    print(f"Solved correctly: {correct}/{total} (Accuracy: {accuracy:.2f})")

    results_file = output_dir / "../../Results.csv"
    csv_headers = ["output_id", "engine", "system_prompt", "train_set", "test_set", "t", "d", "score", "accuracy", "remarks"]
    write_header = not results_file.exists()

    with open(results_file, "a", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_headers)
        if write_header:
            writer.writeheader()

        writer.writerow({
            "output_id": output_id,
            "engine": engine,
            "system_prompt": system_prompt,
            "train_set": train_folder,
            "test_set": test_folder,
            "t": t_val,
            "d": d_val,
            "score": correct,
            "accuracy": f"{accuracy:.4f}",
            "remarks": ""
        })

def save_results_to_csv(df: pd.DataFrame, filename: str = None, base_dir: str = "/root/PreCog/Results") -> str:
    """Save results DataFrame to CSV with user-provided filename"""
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    while True:
        if not filename:
            print("\nSaving results...")
            filename = input("Enter filename for results (with .csv extension): ")
            filename = f"{filename}.csv" if not filename.endswith('.csv') else filename
        
        if not filename.endswith('.csv'):
            print("Filename must end with .csv")
            filename = None
            continue
            
        filepath = Path(base_dir) / filename
        if not filepath.exists():
            break
            
        print(f"File {filepath} already exists.")
        filename = None  # Reset to prompt again
    
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return str(filepath)

def SED_user_prompt_generator(data_dir: str | Path, 
                            train_test_split: Tuple[List[int], List[int]], 
                            give_explanation_flag: int = 0) -> str:
    """Generate prompts using train/test problem IDs and data directory
    
    Args:
        data_dir: Path to dataset containing puzzles/ and solutions/
        train_test_split: Tuple of (train_ids, test_ids) lists 
        give_explanation_flag: Whether to include explanations
    """
    data_dir = Path(data_dir)
    puzzles_dir = data_dir / "puzzles"
    solutions_dir = data_dir / "solutions"
    train_ids, test_ids = train_test_split
    
    training_data = []
    test_data = []
    
    # Process training examples
    for pid in train_ids:
        puzzle_file = puzzles_dir / f"{pid:03d}.json"
        solution_file = solutions_dir / f"{pid:03d}.json"
        
        with open(puzzle_file) as pf, open(solution_file) as sf:
            puzzle = json.load(pf)
            solution = json.load(sf)
            
            # Remove metrics from puzzle
            filtered_puzzle = {
                'problem_id': puzzle['problem_id'],
                'initial_string': puzzle['initial_string'],
                'transitions': puzzle['transitions']
            }
            
            if give_explanation_flag and 'explanation' in solution:
                training_example = {
                    "Puzzle": filtered_puzzle,
                    "Solution": {
                        'problem_id': solution['problem_id'],
                        'solution': solution['solution'],
                        'explanation': solution['explanation']
                    }
                }
            else:
                training_example = {
                    "Puzzle": filtered_puzzle,
                    "Solution": {
                        'problem_id': solution['problem_id'],
                        'solution': solution['solution']
                    }
                }
            training_data.append(training_example)

    # Process test puzzles
    for pid in test_ids:
        puzzle_file = puzzles_dir / f"{pid:03d}.json"
        with open(puzzle_file) as pf:
            puzzle = json.load(pf)
            filtered_puzzle = {
                'problem_id': puzzle['problem_id'],
                'initial_string': puzzle['initial_string'],
                'transitions': puzzle['transitions']
            }
            test_data.append({"Puzzle": filtered_puzzle})

    # Format the prompt
    user_prompt = (
        "Training:\n"
        f"{training_data}\n"
        "```\n"
        "Test:\n"
        f"{test_data}\n"
        "```"
    )
    
    return user_prompt