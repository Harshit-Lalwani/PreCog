import pickle
import random

def load_flat_dataset(file_path):
    with open(file_path, 'rb') as f:
        flat_dataset = pickle.load(f)
    return flat_dataset

def sample_random_string(n, file_path='NISdb_flat.pkl'):
    flat_dataset = load_flat_dataset(file_path)
    
    if n not in flat_dataset or not flat_dataset[n]:
        raise ValueError(f"No strings of length {n} found in the dataset.")
    
    return random.choice(flat_dataset[n])

# # Example usage
# if __name__ == "__main__":
#     n = 7
#     try:
#         random_string = sample_random_string(n)
#         print(f"Random string of length {n}: {random_string}")
#     except ValueError as e:
#         print(e)