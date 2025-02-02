import os
from pathlib import Path

# Constants
DATA_DIR = Path("/root/PreCog/globals")
OUTPUT_ID_FILE = DATA_DIR / "output_id.txt"
PROBLEM_ID_FILE = DATA_DIR / "problem_id.txt"

def _read_or_create_counter(filepath: Path, default: int = 0) -> int:
    """Read counter from file or create with default value"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not filepath.exists():
        with open(filepath, 'w') as f:
            f.write(str(default))
        return default
        
    with open(filepath, 'r') as f:
        return int(f.read().strip())

def _increment_and_save(filepath: Path) -> int:
    """Increment counter and save to file"""
    current = _read_or_create_counter(filepath)
    next_val = current + 1
    
    with open(filepath, 'w') as f:
        f.write(str(next_val))
    
    return next_val

def get_output_id():
    return _increment_and_save(OUTPUT_ID_FILE)

def get_problem_id():
    return _increment_and_save(PROBLEM_ID_FILE)