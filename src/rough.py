from utils import *



for i in range(10):
    train_path = f"data/OneShot/E1/OneShot_3_3_{i}"
    train_path = Path(train_path)

    train_puzzles_dir = train_path / "puzzles"
    train_solutions_dir = train_path / "solutions"

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
                    training_data.append({
                        "Puzzle": puzzle_content,
                        "Solution": solution_content
                    })

    # Format the prompt
    user_prompt = (
        f"{training_data}\n"
    )
    
    print(user_prompt)