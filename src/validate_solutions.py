"""
given data_dir. verify solutions in data_dir/solutions against puzzles in data_dir/puzzles
"""
import json
from pathlib import Path
import logging


def validate_solutions(data_dir: Path):
    count = 0
    puzzles_dir = data_dir / "puzzles"
    solutions_dir = data_dir / "solutions"
    
    for solution_file in solutions_dir.glob("*.json"):
        with open(solution_file, "r") as f:
            solution = json.load(f)
        
        problem_id = solution['problem_id']
        puzzle_file = puzzles_dir / f"{problem_id}.json"
        
        if not puzzle_file.exists():
            logging.warning(f"Puzzle file for problem_id {problem_id} does not exist, skipping...")
            continue
        
        with open(puzzle_file, "r") as f:
            puzzle = json.load(f)
        
        transitions = puzzle['transitions']
        current_string = puzzle['initial_string']
        transition_history = solution['solution']
        
        for step in transition_history:
            if step >= len(transitions):
                logging.warning(f"Invalid step number {step} found in solution for problem_id {problem_id}, skipping...")
                break
            transition = transitions[step]
            current_string = current_string.replace(transition['src'], transition['tgt'], 1)
        
        if current_string != '':
            logging.warning(f"Problem {problem_id} has an invalid solution! Final string: {current_string}")
            count += 1
    return count

# Example usage
if __name__ == "__main__":
    data_dir = Path("/root/PreCog/SED_100")
    print(validate_solutions(data_dir))