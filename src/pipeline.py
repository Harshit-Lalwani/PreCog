#!/root/PreCog/venv/bin/python
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

"""
Input:
train_count[] - list of possible values for "count" variable for train set
train_t[] - list of possible values for "t" variable for train set
train_d[] - list of possible values for "d" variable for train set
test_count[] - list of possible values for "count" variable for test set
test_t[] - list of possible values for "t" variable for test set
test_d[] - list of possible values for "d" variable for test set
sys_prompt[]- list of possible values for "sys_prompt" variable
tranistion_sampler[] - list of possible values for "tranistion_sampler" function
string_sampler[] - list of possible values for "string_sampler" function
num_runs - number of runs to perform for each experiment

Output:
Study/
    Experiments/
        Experiment<i>/ (each digit of i represents the index of the variable in the list of possible values, include starting 0s) (e.g. if it chooses it chooses first value for first variable (train_count), and second value for all other variables, then i = 0111111111)
            Run<j>/ (j= 0 to num_runs - 1)
                Data/
                    Train/
                        puzzles/
                            <puzzleid>.json
                            ...
                        solutions/
                            <puzzleid>.json
                            ...
                    Test/
                        puzzles/
                            <puzzleid>.json
                            ...
                        solutions/
                            <puzzleid>.json
                            ...
                        Predictions/
                            <puzzleid>.json
                            ...
                Results_<run_id>.csv (contains puzzle_id, validity of predicted solution, 1 row per puzzle)
            Results_<experiment_id>.csv (contains run_id, all the chosen parameters, represent functions with their names, accuracy of each run 1 row per run)
    Results/
        Results.csv (contains experiment_id,  all the chosen parameters, represent functions with their names, accuracy of each experiment (accuracy of an experiment is the average accuracy of all the runs), and the self consistency score of each experimetn. 1 row per experiment)
"""

from pathlib import Path
import itertools
from typing import List, Dict, Any, Callable
from pipeline_utils import *
from utils import *
from parse_utils import *
from puzzle_generator import *
from openai import OpenAI
from datetime import datetime
from puzzle_generator import create_transitions_array 

def get_next_study_number(base_dir: Path) -> int:
    """Find smallest unused Study<i> number"""
    i = 1
    while (base_dir / f"Study{i}").exists():
        i += 1
    return i

class Pipeline:
    def __init__(self, base_dir: Path = Path("/root/PreCog/studies")):
        self.base_dir = base_dir
        self.client = OpenAI()
        
        # Create study directory with next available number
        study_num = get_next_study_number(base_dir)
        self.study_dir = base_dir / f"Study{study_num}"
        self.study_dir.mkdir(parents=True, exist_ok=True)
        
        # Study-specific log file
        self.log_file = self.study_dir / "llm_responses.txt"
        
        # Add data directory
        self.data_dir = self.study_dir / "Data"
        self.data_dir.mkdir(exist_ok=True)
        
    def log_llm_response(self, experiment_id: str, run_id: int, prompt: str, response: str):
        """Log raw LLM interaction"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"""
Timestamp: {timestamp}
Experiment: {experiment_id}
Run: {run_id}
Prompt:
{prompt}
Response:
{response}
----------------------------------------
"""
        with open(self.log_file, "a") as f:
                f.write(log_entry)
            
    def get_experiment_dir(self, experiment_id: str) -> Path:
        """Get experiment directory path"""
        return self.study_dir / "Experiments" / f"Experiment{experiment_id}"

    def generate_dataset(self, count: int, t: int, d: int, M: int, 
                        string_sampler: Callable, transition_array_maker: Callable,  # Changed parameter name
                        output_dir: Path):
        """Generate puzzles with solutions"""
        puzzles_dir = output_dir / "puzzles" 
        solutions_dir = output_dir / "solutions"
        puzzles_dir.mkdir(parents=True, exist_ok=True)
        solutions_dir.mkdir(parents=True, exist_ok=True)
        
        puzzles = []
        for i in range(count):
            G, root, transitions, transition_history = generate_single_path(
                n=sample_gaussian_n(M=M), 
                t=t, 
                d=d,
                string_sampler=string_sampler,
                transition_array_maker=transition_array_maker  # Changed parameter name
            )
            
            puzzle = {
                "problem_id": f"{i:03d}",
                "initial_string": root,
                "transitions": transitions
            }
            
            solution = {
                "problem_id": f"{i:03d}",
                "solution": transition_history  
            }
            
            # Save puzzle and solution
            with open(puzzles_dir / f"{i:03d}.json", "w") as f:
                json.dump(puzzle, f, indent=4)
            with open(solutions_dir / f"{i:03d}.json", "w") as f:
                json.dump(solution, f, indent=4)
                
            puzzles.append(puzzle)
            
        return puzzles

    def generate_data_id(self, data_indices: tuple, num_params: int) -> str:
        """Generate dataset ID from parameter indices"""
        return ''.join(str(i) for i in data_indices).zfill(num_params)

    def create_dataset(self, dataset_id: str, params: Dict, num_runs: int):
        """Create a dataset with multiple runs"""
        dataset_dir = self.data_dir / f"Data_{dataset_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        for run_id in range(num_runs):
            run_dir = dataset_dir / f"Run{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            data_run_dir = run_dir / f"{dataset_id}_{run_id}"
            
            # Generate train data
            train_dir = data_run_dir / "train"
            self.generate_dataset(
                count=params['train_count'],
                t=params['train_t'],
                d=params['train_d'],
                M=params['M'],
                string_sampler=params['string_sampler'],
                transition_array_maker=params['transition_array_maker'],  # Fixed parameter name
                output_dir=train_dir
            )
            
            # Generate test data
            test_dir = data_run_dir / "test"
            self.generate_dataset(
                count=params['test_count'],
                t=params['test_t'],
                d=params['test_d'],
                M=params['M'],
                string_sampler=params['string_sampler'],
                transition_array_maker=params['transition_array_maker'],  # Fixed parameter name
                output_dir=test_dir
            )

    def run_study(self,
                  # Parameters for data generation
                  for_data = {
                      'train_count': List[int],
                      'train_t': List[int],
                      'train_d': List[int],
                      'test_count': List[int],
                      'test_t': List[int],
                      'test_d': List[int],
                      'M': List[int],
                      'string_samplers': List[Callable],
                      'transition_array_makers': List[Callable]  # Changed from transition_samplers
                  },
                  # Parameters for testing
                  for_test = {
                      'sys_prompts': List[str],
                      'models': List[str]
                  },
                  num_runs: int = 1):
        
        # Create parameter combinations
        data_param_lists = [
            for_data['train_count'], 
            for_data['train_t'], 
            for_data['train_d'],
            for_data['test_count'], 
            for_data['test_t'], 
            for_data['test_d'],
            for_data['M'],
            for_data['string_samplers'],
            for_data['transition_array_makers']  # Changed from transition_samplers
        ]
        
        # Generate all datasets first
        for data_indices in itertools.product(*[range(len(x)) for x in data_param_lists]):
            dataset_id = self.generate_data_id(data_indices, len(data_param_lists))
            params = {
                'train_count': for_data['train_count'][data_indices[0]],
                'train_t': for_data['train_t'][data_indices[1]],
                'train_d': for_data['train_d'][data_indices[2]],
                'test_count': for_data['test_count'][data_indices[3]],
                'test_t': for_data['test_t'][data_indices[4]],
                'test_d': for_data['test_d'][data_indices[5]],
                'M': for_data['M'][data_indices[6]],
                'string_sampler': for_data['string_samplers'][data_indices[7]],
                'transition_array_maker': for_data['transition_array_makers'][data_indices[8]]  # Changed from transition_sampler
            }
            self.create_dataset(dataset_id, params, num_runs)

        # Run experiments
        test_param_lists = [for_test['sys_prompts'], for_test['models']]
        all_experiments = []
        
        for data_indices in itertools.product(*[range(len(x)) for x in data_param_lists]):
            dataset_id = self.generate_data_id(data_indices, len(data_param_lists))
            
            for test_indices in itertools.product(*[range(len(x)) for x in test_param_lists]):
                experiment_id = self.generate_data_id(
                    data_indices + test_indices,
                    len(data_param_lists) + len(test_param_lists)
                )
                
                params = {
                    **{k: for_data[k][i] for k, i in zip(
                        ['train_count', 'train_t', 'train_d', 'test_count', 'test_t', 'test_d', 'M'],
                        data_indices
                    )},
                    'sys_prompt': for_test['sys_prompts'][test_indices[0]],
                    'model': for_test['models'][test_indices[1]],
                    'dataset_id': dataset_id
                }
                
                experiment_results = self.run_experiment(experiment_id, dataset_id, params, num_runs)
                all_experiments.append(experiment_results)
        
        save_study_results(self.study_dir, all_experiments)

    def run_experiment(self, experiment_id: str, dataset_id: str, params: Dict[str, Any], num_runs: int) -> Dict:
        experiment_dir = self.get_experiment_dir(experiment_id)
        dataset_dir = self.data_dir / f"Data_{dataset_id}"
        all_runs = []
        
        for run_id in range(num_runs):
            run_dir = experiment_dir / f"Run{run_id}"
            output_dir = run_dir / "Output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write dataset description
            with open(run_dir / "description.txt", "w") as f:
                f.write(f"Dataset: Data_{dataset_id}/Run{run_id}/{dataset_id}_{run_id}\n")
                f.write(f"Parameters: {json.dumps(params, indent=2)}")
            
            # Get data paths
            data_run_dir = dataset_dir / f"Run{run_id}" / f"{dataset_id}_{run_id}"
            train_dir = data_run_dir / "train"
            test_dir = data_run_dir / "test"
            
            # Load test puzzles
            test_puzzles = []
            for puzzle_file in (test_dir / "puzzles").glob("*.json"):
                with open(puzzle_file) as f:
                    test_puzzles.append(json.load(f))
            
            # Get predictions using OpenAI API
            predictions = []
            for test_puzzle in test_puzzles:
                user_prompt = user_prompt_generator(train_dir, test_dir)
                
                # do not change to create. parse is a beta function, but necessary to use response_format
                response = self.client.beta.chat.completions.parse( 
                    model=params['model'],
                    messages=[
                        {"role": "system", "content": params['sys_prompt']},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=Solutions
                )
                
                # Log raw response before parsing
                self.log_llm_response(
                    experiment_id,
                    run_id,
                    user_prompt,
                    response.choices[0].message.content
                )
                
                # Parse response
                response_text = response.choices[0].message.content
                parsed_response = parse_solution_with_fallback(response_text)
                
                prediction = {
                    'puzzle_id': test_puzzle['problem_id'],
                    'predicted_solution': str(parsed_response.solutions[0].solution),
                    'is_valid': validate_solution_sequence(
                        test_puzzle['initial_string'],
                        test_puzzle['transitions'],
                        parsed_response.solutions[0].solution
                    )
                }
                predictions.append(prediction)
                
                # Save prediction to Output directory
                prediction_file = output_dir / f"{test_puzzle['problem_id']}.json"
                with open(prediction_file, 'w') as f:
                    json.dump(prediction, f, indent=4)
            
            # Save results
            run_params = {**params, 'run_id': run_id}
            save_run_results(run_dir / f"Results_{run_id}.csv", predictions, run_params)
            
            run_results = {
                'run_id': run_id,
                'valid_predictions': sum(p['is_valid'] for p in predictions),
                'total_predictions': len(predictions),
                **params
            }
            all_runs.append(run_results)
        
        # Save experiment results directly to experiment directory
        exp_params = {**params, 'experiment_id': experiment_id}
        results_file = experiment_dir / "results.csv"  # Save directly to experiment dir
        save_experiment_results(results_file, all_runs, exp_params)
        
        return {
        'experiment_id': experiment_id,
        'dataset_id': dataset_id,
        'accuracy': sum(r['valid_predictions'] for r in all_runs) / 
                   sum(r['total_predictions'] for r in all_runs),
        **params
    }

# Update the example usage
if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run_study(
        for_data={
            'train_count': [1],
            'train_t': [1,2],
            'train_d': [1],
            'test_count': [1],
            'test_t': [1],
            'test_d': [1],
            'M': [10],
            'string_samplers': [get_string_sampler],
            'transition_array_makers': [create_transitions_array]  # Changed from transition_samplers
        },
        for_test={
            'sys_prompts': ["Solve the puzzle"],
            'models': ["gpt-4o"]
        },
        num_runs=1
    )