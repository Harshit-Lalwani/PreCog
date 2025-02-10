from utils import *
import json
import os
from global_var import *

base_path = "data/OneShot/t_variation"

def generate_single_path(n: int, t: int, d: int) -> Tuple[nx.DiGraph, str, List[Dict], List[int]]:
    """Generate single puzzle instance with exactly d transitions"""
    while True:  # Keep trying until we get a path of length d
        # Sample initial string
        root = sample_random_string(n)
        
        # Create graph
        G = nx.DiGraph()
        G.add_node((0,), string=root)
        
        transitions = []
        transition_history = []
        
        # Generate transitions using ASCII range constraints
        for i in range(t):
            transition = create_transition(n, root)
            transitions.append(transition)
        
        current_node = (0,)
        current_string = root
        
        # Try to make exactly d transitions
        success = True
        for i in range(d):
            # Try all transitions if necessary to find a valid one
            valid_transitions = list(range(t))
            random.shuffle(valid_transitions)
            
            found_valid = False
            for j in valid_transitions:
                transition = transitions[j]
                new_string = apply_transition(current_string, transition)
                if new_string != "-1":
                    transition_history.append(j)
                    new_node = tuple(list(current_node) + [i])
                    G.add_node(new_node, string=new_string)
                    G.add_edge(current_node, new_node, 
                             label=f"{transition['src']} -> {transition['tgt']}")
                    current_node = new_node
                    current_string = new_string
                    found_valid = True
                    break
            
            if not found_valid:
                success = False
                break
        
        # Only return if we got exactly d transitions
        if success and len(transition_history) == d:
            transitions.append({"src": current_string, "tgt": ""})
            transition_history.append(t)
            return G, root, transitions, transition_history

def get_next_E_number(base_path: str) -> int:
    """Find smallest unused E<q> number"""
    i = 1
    while True:
        if not os.path.exists(os.path.join(base_path, f"E{i}")):
            return i
        i += 1

def puzzle_generator(count=10, t=3, d=3, E_num=1):
    # base_path = "data/OneShot/t_variation"
    E_dir = f"E{E_num}"
    base_dir = os.path.join(base_path, E_dir, f"OneShot_{t}_{d}")
    
    j = get_problem_id()
    output_dir = f"{base_dir}_{j}"
    
    puzzles_dir = os.path.join(output_dir, 'puzzles')
    solutions_dir = os.path.join(output_dir, 'solutions')
    
    os.makedirs(puzzles_dir, exist_ok=True)
    os.makedirs(solutions_dir, exist_ok=True)
    
    for i in range(count):
        G, root, transitions, transition_history = generate_single_path(t=t, d=d)
        
        puzzle = {
            "problem_id": f"{j:03d}",
            "initial_string": root,
            "transitions": transitions
        }
        
        solution = {
            "problem_id": f"{j:03d}",
            "solution": transition_history
        }
        
        with open(os.path.join(puzzles_dir, f"{j:03d}.json"), 'w') as f:
            json.dump(puzzle, f, indent=4)
        
        with open(os.path.join(solutions_dir, f"{j:03d}.json"), 'w') as f:
            json.dump(solution, f, indent=4)  
            
def puzzle_set_generator(set_count=10, puzzles_per_set=10, t=3, d=3):
    # base_path = "data/OneShot/t_variation"
    E_num = get_next_E_number(base_path)
    
    for i in range(set_count):
        puzzle_generator(count=puzzles_per_set, t=t, d=d, E_num=E_num)
                           
# Example usage
if __name__ == "__main__":
    puzzle_set_generator(set_count=10, puzzles_per_set=1, t=6, d=3)
    # print(get_next_E_number("data/OneShot/t_variation"))