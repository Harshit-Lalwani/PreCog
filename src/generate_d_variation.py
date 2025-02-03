from puzzle_generator import *
from pathlib import Path

def generate_d_variation_datasets():
    # Setup base directory
    base_dir = Path("/root/PreCog/data/OneShot/d_variation")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    t = 5  # fixed t value
    
    # Generate datasets for d=1 to 10
    for d in range(1, 11):
        E_dir = f"E{d}"  # E number matches d value
        puzzle_generator(
            count=1,  # one puzzle per set
            t=t, 
            d=d,
        )