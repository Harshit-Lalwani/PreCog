"""
For each puzzle in given dataset, add explanation to solutions json files."""
import json
from pathlib import Path
from pipeline_utils import calculate_markov_entropy, apply_transition

def add_explanation_to_solutions(dataset_dir: Path):
    solutions_dir = dataset_dir / "solutions"
    puzzles_dir = dataset_dir / "puzzles"
    for solution_file in solutions_dir.glob("*.json"):
        with open(solution_file, "r") as f:
            solution = json.load(f)
        with open(puzzles_dir / f"{solution['problem_id']}.json", "r") as f:
            puzzle = json.load(f)
        
        transition_history = solution['solution']
        initial_string = puzzle['initial_string']
        transitions = puzzle['transitions']
        
        # Generate step by step explanation
        current_string = initial_string
        explanation_steps = []
        for step_num, trans_idx in enumerate(transition_history, 1):
            if trans_idx == len(transitions):  # Final empty transition
                explanation_steps.append(f"STEP{step_num}: \"{current_string}\" occurs in \"{current_string}\"\n"
                                         f"applying final transition (\"{current_string}\"->\"\") gives \"\"")
                break
            
            transition = transitions[trans_idx]
            next_string = apply_transition(current_string, transition)
            explanation_steps.append(f"STEP{step_num}: \"{transition['src']}\" occurs in \"{current_string}\"\n"
                                     f"applying transition {trans_idx} (\"{transition['src']}\"->\"{transition['tgt']}\") gives \"{next_string}\"")
            current_string = next_string
        
        solution['explanation'] = "\n".join(explanation_steps)
        
        with open(solution_file, "w") as f:
            json.dump(solution, f, indent=4)

# Example usage
if __name__ == "__main__":
    dataset_dir = Path("/root/PreCog/SED_100/data")
    add_explanation_to_solutions(dataset_dir)