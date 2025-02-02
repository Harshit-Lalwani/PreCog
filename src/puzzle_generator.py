from utils import *
import json
import os

def generate_single_path(n = sample_gaussian_n(), t=3, d=3, file_path='data/NIS/NISdb_flat.pkl'):        
    root = sample_random_string(n, file_path)
    transitions = create_transitions_array(n, t, file_path)
    
    G = nx.DiGraph()
    G.add_node((-1,), string=root)
    current_node = (-1,)
    current_string = root
    
    transition_history = []
    
    for i in range(d):
        j = random.choice(range(t))
        transition = transitions[j]        
        new_string = apply_transition(current_string, transition)
        if new_string is not None:
            transition_history.append(j)
            new_node = tuple(list(current_node) + [i])
            G.add_node(new_node, string=new_string)
            # Fix: Use transition["src"] and transition["tgt"] 
            G.add_edge(current_node, new_node, label=f"{transition['src']} -> {transition['tgt']}")
            current_node = new_node
            current_string = new_string
    
    # Fix: Append dictionary format transition
    transitions.append({"src": current_string, "tgt": ""})
    transition_history.append(t)
    
    return G, root, transitions, transition_history

def puzzle_generator(count=10, t=3, d=3):
    base_dir = f"data/OneShot_{t}_{d}"
    i = 0
    while os.path.exists(f"{base_dir}_{i}"):
        i += 1
    output_dir = f"{base_dir}_{i}"
    
    puzzles_dir = os.path.join(output_dir, 'puzzles')
    solutions_dir = os.path.join(output_dir, 'solutions')
    
    os.makedirs(puzzles_dir)
    os.makedirs(solutions_dir)
    
    for i in range(count):
        G, root, transitions, transition_history = generate_single_path(t=t, d=d)
        
        puzzle = {
            "problem_id": f"{i:03d}",
            "initial_string": root,
            # Remove list comprehension as transitions are already in correct format
            "transitions": transitions
        }
        
        solution = {
            "problem_id": f"{i:03d}",
            "solution": transition_history
        }
        
        with open(os.path.join(puzzles_dir, f"{i:03d}.json"), 'w') as f:
            json.dump(puzzle, f, indent=4)
        
        with open(os.path.join(solutions_dir, f"{i:03d}.json"), 'w') as f:
            json.dump(solution, f, indent=4)  
                           
# Example usage
if __name__ == "__main__":
    for i in range(10):
        puzzle_generator(count=1)