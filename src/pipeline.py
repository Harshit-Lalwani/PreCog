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
from prompt_manager import PromptManager

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
        self.prompt_manager = PromptManager()
        
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
                        string_sampler_type: int,
                        transition_array_sampler_type: int,
                        output_dir: Path):
        """Generate puzzles with solutions"""
        puzzles_dir = output_dir / "puzzles" 
        solutions_dir = output_dir / "solutions"
        puzzles_dir.mkdir(parents=True, exist_ok=True)
        solutions_dir.mkdir(parents=True, exist_ok=True)
        
        puzzles = []
        for i in range(count):
            n = sample_gaussian_n(M=M)
            
            # Get samplers based on specified types
            string_sampler = get_string_sampler(n, string_sampler_type)
            transition_array_sampler = get_transition_array_sampler(n, transition_array_sampler_type)
            
            # Pass both samplers to generate_single_path_for_pipeline
            G, root, transitions, transition_history = generate_single_path_for_pipeline(
                n=n,
                t=t,
                d=d,
                string_sampler=string_sampler,
                transition_array_sampler=transition_array_sampler
            )
            
            # Generate step by step explanation
            current_string = root
            explanation_steps = []
            for step_num, trans_idx in enumerate(transition_history, 1):
                if (trans_idx == t):  # Final empty transition
                    explanation_steps.append(f"STEP{step_num}: \"{current_string}\" occurs in \"{current_string}\"\n"
                                          f"applying final transition (\"{current_string}\"->\"\") gives \"\"")
                    break
                    
                transition = transitions[trans_idx]
                next_string = apply_transition(current_string, transition)
                explanation_steps.append(f"STEP{step_num}: \"{transition['src']}\" occurs in \"{current_string}\"\n"
                                      f"applying transition {trans_idx} (\"{transition['src']}\"->\"{transition['tgt']}\") gives \"{next_string}\"")
                current_string = next_string
                
            puzzle = {
                "problem_id": f"{i:03d}",
                "initial_string": root,
                "transitions": transitions
            }
            
            solution = {
                "problem_id": f"{i:03d}",
                "solution": transition_history,
                "explanation": "\n".join(explanation_steps)
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
                string_sampler_type=params['string_sampler_type'],
                transition_array_sampler_type=params['transition_array_sampler_type'],  # Fixed name
                output_dir=train_dir
            )
            
            # Generate test data
            test_dir = data_run_dir / "test"
            self.generate_dataset(
                count=params['test_count'],
                t=params['test_t'],
                d=params['test_d'],
                M=params['M'],
                string_sampler_type=params['string_sampler_type'],
                transition_array_sampler_type=params['transition_array_sampler_type'],  # Fixed name
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
                      'string_sampler_types': List[int],    # Added
                      'transition_array_sampler_types': List[int] # Renamed
                  },
                  # Parameters for testing
                  for_test = {
                      'prompt_titles': List[str],  # List of prompt titles from prompts.json
                      'give_explanation_flags': [0, 1],  # Whether to include explanation in prompts
                      'ask_explanation_flags': [0, 1],   # Whether to expect explanation in response
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
            for_data['string_sampler_types'],
            for_data['transition_array_sampler_types']
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
                'string_sampler_type': for_data['string_sampler_types'][data_indices[7]],
                'transition_array_sampler_type': for_data['transition_array_sampler_types'][data_indices[8]]  # Fixed name
            }
            self.create_dataset(dataset_id, params, num_runs)

        # Run experiments
        test_param_lists = [
            for_test['prompt_titles'],
            for_test['give_explanation_flags'], 
            for_test['ask_explanation_flags'],
            for_test['models']
        ]
        all_experiments = []
        
        # Calculate total number of combinations
        data_combinations = len(list(itertools.product(*[range(len(x)) for x in data_param_lists])))
        test_combinations = len(list(itertools.product(*[range(len(x)) for x in test_param_lists])))
        total_combinations = data_combinations * test_combinations
        current = 0
        
        print(f"Starting study with {total_combinations} total experiments...")
        
        for data_indices in itertools.product(*[range(len(x)) for x in data_param_lists]):
            dataset_id = self.generate_data_id(data_indices, len(data_param_lists))
            
            for test_indices in itertools.product(*[range(len(x)) for x in test_param_lists]):
                experiment_id = self.generate_data_id(
                    data_indices + test_indices,
                    len(data_param_lists) + len(test_param_lists)
                )
                
                params = {
                    **{k: for_data[k][i] for k, i in zip(
                        ['train_count', 'train_t', 'train_d', 'test_count', 'test_t', 'test_d', 'M', 
                         'string_sampler_types', 'transition_array_sampler_types'],  # Fixed name
                        data_indices
                    )},
                    'prompt_title': for_test['prompt_titles'][test_indices[0]],  # Get title
                    'sys_prompt': self.prompt_manager.get_prompt(for_test['prompt_titles'][test_indices[0]]),  # Get content
                    'give_explanation_flag': for_test['give_explanation_flags'][test_indices[1]],  # Store flag
                    'ask_explanation_flag': for_test['ask_explanation_flags'][test_indices[2]],  # Store flag
                    'model': for_test['models'][test_indices[3]],
                    'dataset_id': dataset_id
                }
                
                experiment_results = self.run_experiment(experiment_id, dataset_id, params, num_runs)
                all_experiments.append(experiment_results)
                
                # Update and display progress
                current += 1
                progress = (current / total_combinations) * 100
                print(f"Progress: {progress:.1f}% ({current}/{total_combinations} experiments)", end='\r')
        
        print("\nStudy completed!") 
        save_study_results(self.study_dir, all_experiments)
        
        # Convert enum values to names for study description
        string_sampler_names = [StringSamplerType(t).name for t in for_data['string_sampler_types']]
        transition_array_names = [TransitionArraySamplerType(t).name for t in for_data['transition_array_sampler_types']]
        
        # Create study description
        description = f"""# Study Description

## Parameters
### Data Generation Parameters
- Train Count: {for_data['train_count']}
- Train t: {for_data['train_t']}
- Train d: {for_data['train_d']}
- Test Count: {for_data['test_count']}
- Test t: {for_data['test_t']}
- Test d: {for_data['test_d']}
- M: {for_data['M']}
- String Sampler Types: {string_sampler_names}
- Transition Array Types: {transition_array_names}

### Testing Parameters
- System Prompts: {len(for_test['prompt_titles'])} prompts
- Models: {for_test['models']}
- Number of Runs: {num_runs}

## Results Summary
Total Experiments: {total_combinations}

### Model-wise Accuracies
"""
        
        # Calculate and add model-wise accuracies
        model_results = {}
        for exp in all_experiments:
            model = exp['model']
            if model not in model_results:
                model_results[model] = {'total': 0, 'count': 0}
            model_results[model]['total'] += exp['accuracy']
            model_results[model]['count'] += 1
        
        for model, results in model_results.items():
            avg_accuracy = results['total'] / results['count']
            description += f"- {model}: {avg_accuracy:.2%}\n"
        
        description += "\n## Remarks\n"
        description += "<!-- Add your remarks here -->"
        
        # Save description
        with open(self.study_dir / "description.md", "w") as f:
            f.write(description)

    def run_experiment(self, experiment_id: str, dataset_id: str, params: Dict[str, Any], num_runs: int) -> Dict:
        experiment_dir = self.get_experiment_dir(experiment_id)
        dataset_dir = self.data_dir / f"Data_{dataset_id}"
        all_runs = []
        
        # Add error log file
        error_log = experiment_dir / "validation_errors.log"
        
        # Convert enum values to names
        string_sampler_name = StringSamplerType(params['string_sampler_types']).name
        transition_array_name = TransitionArraySamplerType(params['transition_array_sampler_types']).name
        
        # Create experiment description at the start
        exp_description = f"""# Experiment {experiment_id}

## Parameters
- Train Count: {params['train_count']}
- Train t: {params['train_t']}
- Train d: {params['train_d']}
- Test Count: {params['test_count']}
- Test t: {params['test_t']}
- Test d: {params['test_d']}
- M: {params['M']}
- String Sampler Type: {string_sampler_name}
- Transition Array Type: {transition_array_name}
- System Prompt: {params['sys_prompt']}
- Model: {params['model']}

## Results
"""
        
        for run_id in range(num_runs):
            run_dir = experiment_dir / f"Run{run_id}"
            output_dir = run_dir / "Output"  # Add this line
            output_dir.mkdir(parents=True, exist_ok=True)  # Add this line
            
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
                try:
                    user_prompt = user_prompt_generator(train_dir, test_dir)
                    
                    response = self.client.beta.chat.completions.parse(
                        model=params['model'],
                        messages=[
                            {"role": "system", "content": params['sys_prompt']},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format=Solutions
                    )
                    
                    # Log raw response
                    self.log_llm_response(
                        experiment_id,
                        run_id,
                        user_prompt,
                        response.choices[0].message.content
                    )
                    
                    try:
                        # Parse response
                        response_text = response.choices[0].message.content
                        parsed_response = parse_solution_with_fallback(response_text)
                        
                        # Validate solution with error handling
                        try:
                            is_valid = validate_solution_sequence(
                                test_puzzle['initial_string'],
                                test_puzzle['transitions'],
                                parsed_response.solutions[0].solution
                            )
                        except Exception as e:
                            # Log validation error
                            error_msg = (f"Validation error for puzzle {test_puzzle['problem_id']}\n"
                                       f"Run: {run_id}, Experiment: {experiment_id}\n"
                                       f"Error: {str(e)}\n"
                                       f"Solution: {parsed_response.solutions[0].solution}\n"
                                       f"Initial string: {test_puzzle['initial_string']}\n"
                                       f"Transitions: {test_puzzle['transitions']}\n"
                                       "----------------------------------------\n")
                            with open(error_log, "a") as f:
                                f.write(error_msg)
                            is_valid = False

                        prediction = {
                            'puzzle_id': test_puzzle['problem_id'],
                            'predicted_solution': str(parsed_response.solutions[0].solution),
                            'is_valid': is_valid,
                            'error': None if is_valid else "Validation failed"
                        }

                    except Exception as e:
                        # Handle parsing errors
                        error_msg = (f"Parsing error for puzzle {test_puzzle['problem_id']}\n"
                                   f"Run: {run_id}, Experiment: {experiment_id}\n"
                                   f"Error: {str(e)}\n"
                                   f"Response: {response_text}\n"
                                   "----------------------------------------\n")
                        with open(error_log, "a") as f:
                            f.write(error_msg)
                            
                        prediction = {
                            'puzzle_id': test_puzzle['problem_id'],
                            'predicted_solution': "[]",
                            'is_valid': False,
                            'error': f"Parsing failed: {str(e)}"
                        }

                except Exception as e:
                    # Handle API errors
                    error_msg = (f"API error for puzzle {test_puzzle['problem_id']}\n"
                               f"Run: {run_id}, Experiment: {experiment_id}\n"
                               f"Error: {str(e)}\n"
                               "----------------------------------------\n")
                    with open(error_log, "a") as f:
                        f.write(error_msg)
                        
                    prediction = {
                        'puzzle_id': test_puzzle['problem_id'],
                        'predicted_solution': "[]",
                        'is_valid': False,
                        'error': f"API error: {str(e)}"
                    }

                predictions.append(prediction)
                
                # Save prediction with error information
                prediction_file = output_dir / f"{test_puzzle['problem_id']}.json"
                with open(prediction_file, 'w') as f:
                    json.dump(prediction, f, indent=4)

        # Save results
        run_params = {**params, 'run_id': run_id}
        save_run_results(run_dir, predictions, run_params)
        
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
        
        # After all runs are complete, add run accuracies to description
        total_accuracy = 0
        for run_result in all_runs:
            accuracy = run_result['valid_predictions'] / run_result['total_predictions']
            total_accuracy += accuracy
            exp_description += f"Run {run_result['run_id']}: {accuracy:.2%} accuracy\n"
        
        # Add average accuracy
        avg_accuracy = total_accuracy / len(all_runs)
        exp_description += f"\nAverage Accuracy: {avg_accuracy:.2%}"
        
        # Save experiment description
        with open(experiment_dir / "description.md", "w") as f:
            f.write(exp_description)
        
        return {
        'experiment_id': experiment_id,
        'dataset_id': dataset_id,
        'accuracy': avg_accuracy,
        'num_runs': num_runs,  # Add this line
        **params
    }

# Example usage
if __name__ == "__main__":
    pipeline = Pipeline()

    pipeline.run_study(
        for_data={
            'train_count': [1],
            'train_t': [3],
            'train_d': [3],
            'test_count': [1],
            'test_t': [3],
            'test_d': [3],
            'M': [7],
            'string_sampler_types': [0],
            'transition_array_sampler_types': [1]
        },
        for_test={

            'prompt_titles': [ 'go_step_by_step'],  # Add this line            
            'give_explanation_flags': [0, 1],  # Whether to include explanation in prompts
            'ask_explanation_flags': [0, 1],   # Whether to expect explanation in response
            'models': ["gpt-4o"]
        },
        num_runs=1
    )