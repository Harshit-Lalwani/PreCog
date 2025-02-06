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

    def generate_dataset(self, count: int, t: int, d: int, M: int, output_dir: Path):
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
                d=d
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

    def run_study(self,
                  train_count: List[int],
                  train_t: List[int], 
                  train_d: List[int],
                  test_count: List[int],
                  test_t: List[int],
                  test_d: List[int],
                  sys_prompts: List[str],
                  transition_samplers: List[Callable],
                  string_samplers: List[Callable],
                  models: List[str],
                  M_values: List[int],  # Add M parameter list
                  num_runs: int):
        
        # Create parameter combinations
        param_lists = [train_count, train_t, train_d, 
                      test_count, test_t, test_d,
                      sys_prompts, models, M_values]  # Add M_values
        
        all_experiments = []
        
        for params_indices in itertools.product(*[range(len(x)) for x in param_lists]):
            experiment_id = generate_experiment_id(params_indices, len(param_lists))
            params = {
                'train_count': train_count[params_indices[0]],
                'train_t': train_t[params_indices[1]],
                'train_d': train_d[params_indices[2]],
                'test_count': test_count[params_indices[3]],
                'test_t': test_t[params_indices[4]],
                'test_d': test_d[params_indices[5]],
                'sys_prompt': sys_prompts[params_indices[6]],
                'model': models[params_indices[7]],
                'M': M_values[params_indices[8]]  # Add M
            }
            
            experiment_results = self.run_experiment(experiment_id, params, num_runs)
            all_experiments.append(experiment_results)
            
        save_study_results(self.study_dir, all_experiments)

    def run_experiment(self, experiment_id: str, params: Dict[str, Any], num_runs: int) -> Dict:
        experiment_dir = self.get_experiment_dir(experiment_id)
        all_runs = []
        
        for run_id in range(num_runs):
            run_dir = experiment_dir / f"Run{run_id}"
            data_dir = run_dir / "Data"
            
            # Create directories
            run_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate datasets with solutions
            train_puzzles = self.generate_dataset(
                count=params['train_count'],
                t=params['train_t'],
                d=params['train_d'], 
                M=params['M'],
                output_dir=data_dir / "Train"
            )
            
            test_puzzles = self.generate_dataset(
                count=params['test_count'],
                t=params['test_t'],
                d=params['test_d'],
                M=params['M'], 
                output_dir=data_dir / "Test"
            )
            
            # Get predictions using OpenAI API
            predictions = []
            for test_puzzle in test_puzzles:
                user_prompt = user_prompt_generator(
                    data_dir / "Train",
                    data_dir / "Test"
                )
                
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
                
                # Save prediction with .json extension
                prediction_file = data_dir / "Test" / "Predictions" / f"{test_puzzle['problem_id']}.json"
                prediction_file.parent.mkdir(parents=True, exist_ok=True)
                with open(prediction_file, 'w') as f:
                    json.dump(prediction, f, indent=4)
            
            # Save results with proper path
            run_params = {**params, 'run_id': run_id}
            save_run_results(run_dir, predictions, run_params)
            
            run_results = {
                'run_id': run_id,
                'valid_predictions': sum(p['is_valid'] for p in predictions),
                'total_predictions': len(predictions),
                **params
            }
            all_runs.append(run_results)
        
        # Save experiment results
        exp_params = {**params, 'experiment_id': experiment_id}
        save_experiment_results(experiment_dir, all_runs, exp_params)
        
        return {
            'experiment_id': experiment_id,
            'accuracy': sum(r['valid_predictions'] for r in all_runs) / 
                       sum(r['total_predictions'] for r in all_runs),
            **params
        }

if __name__ == "__main__":
    # Example usage
    pipeline = Pipeline()
    pipeline.run_study(
        train_count=[1,2],
        train_t=[1,2],
        train_d=[1],
        test_count=[1],
        test_t=[1], 
        test_d=[1],
        sys_prompts=["Solve the puzzle by providing step sequence"],
        transition_samplers=[create_transition],
        string_samplers=[sample_random_string],
        models=["gpt-4o"],
        M_values=[3],  # Add M values
        num_runs=2
    )