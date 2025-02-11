from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import json
import itertools
from openai import OpenAI
from datetime import datetime
from prompt_manager import PromptManager
from pipeline_utils import *
from utils import *


def get_next_SED_study_number(base_dir: Path) -> int:
    """Find smallest unused SED_Study<i> number"""
    i = 1
    while (base_dir / f"SED_Study{i}").exists():
        i += 1
    return i

class SEDPipeline:
    def __init__(self, dataset_path: str, base_dir: Path = Path("/root/PreCog/SED_studies")):
        self.dataset_path = Path(dataset_path)
        self.base_dir = base_dir
        self.client = OpenAI()
        self.prompt_manager = PromptManager()
        
        # Verify dataset structure
        self.puzzles_dir = self.dataset_path / "puzzles"
        self.solutions_dir = self.dataset_path / "solutions"
        if not (self.puzzles_dir.exists() and self.solutions_dir.exists()):
            raise ValueError("Dataset must contain 'puzzles' and 'solutions' directories")
            
        # Create study directory
        study_num = get_next_SED_study_number(base_dir)
        self.study_dir = base_dir / f"SED_Study{study_num}"
        self.study_dir.mkdir(parents=True, exist_ok=True)
        
        # Study-specific log file
        self.log_file = self.study_dir / "llm_responses.txt"

    def log_llm_response(self, experiment_id: str, run_id: int, system_prompt,  prompt: str, response: str):
        """Log raw LLM interaction"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"""
Timestamp: {timestamp}
Experiment: {experiment_id}
Run: {run_id}
System Prompt: {system_prompt}
Prompt:
{prompt}
Response:
{response}
----------------------------------------
"""
        with open(self.log_file, "a") as f:
            f.write(log_entry)

    def get_puzzle_solution(self, problem_id: int) -> Tuple[Dict, Dict]:
        """Load puzzle and solution files for given problem_id"""
        puzzle_file = self.puzzles_dir / f"{problem_id:03d}.json"
        solution_file = self.solutions_dir / f"{problem_id:03d}.json"
        
        with open(puzzle_file) as pf, open(solution_file) as sf:
            puzzle = json.load(pf)
            solution = json.load(sf)
        return puzzle, solution

    def run_study(self, train_test_splits: List[Tuple[List[int], List[int]]], for_test: Dict[str, List[Any]]):
        """Run study with given splits and test parameters"""
        # Save study description
        description = f"""Study Description
----------------
Dataset Path: {self.dataset_path}
Study Directory: {self.study_dir}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Parameters
----------
Prompt Titles: {', '.join(for_test['prompt_titles'])}
Give Explanation Flags: {', '.join(map(str, for_test['give_explanation_flags']))}
Ask Explanation Flags: {', '.join(map(str, for_test['ask_explanation_flags']))}
Models: {', '.join(for_test['models'])}

Splits Information
-----------------
Total Splits: {len(train_test_splits)}
Example Split:
  Train IDs: {train_test_splits[0][0]}
  Test IDs: {train_test_splits[0][1]}
"""
    
        with open(self.study_dir / "description.txt", "w") as f:
            f.write(description)
        
        # Rest of the study execution
        param_lists = [
            for_test['prompt_titles'],
            for_test['give_explanation_flags'],
            for_test['ask_explanation_flags'],
            for_test['models']
        ]
        
        experiments_results = []
        
        # For each parameter combination
        for params_indices in itertools.product(*[range(len(x)) for x in param_lists]):
            experiment_id = generate_experiment_id(params_indices, len(param_lists))
            exp_dir = self.study_dir / "Experiments" / f"Experiment{experiment_id}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Create parameter dictionary
            params = {
                'prompt_title': for_test['prompt_titles'][params_indices[0]],
                'give_explanation': for_test['give_explanation_flags'][params_indices[1]],
                'ask_explanation': for_test['ask_explanation_flags'][params_indices[2]],
                'model': for_test['models'][params_indices[3]]
            }
            
            run_results = []
            
            # For each train/test split
            for run_id, (train_ids, test_ids) in enumerate(train_test_splits):
                results = self.run_experiment(experiment_id, run_id, train_ids, test_ids, params)
                run_results.append(results)
            
            # Save experiment results
            self.save_experiment_results(exp_dir, run_results, params)
            experiments_results.append({
                'experiment_id': experiment_id,
                **params,
                'accuracy': sum(r['run_accuracy'] for r in run_results) / len(run_results)
            })
        
        # Save study results
        self.save_study_results(experiments_results)

    def run_run(self, experiment_id: str, run_id: int, 
                train_ids: List[int], test_ids: List[int], 
                params: Dict[str, Any]) -> Tuple[List[Dict], Path]:
        """Run a single run with given train/test split and save results"""
        # Create run directory
        run_dir = self.study_dir / "Experiments" / f"Experiment{experiment_id}" / f"Run{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training data
        train_data = [self.get_puzzle_solution(pid) for pid in train_ids]
        test_puzzles = [self.get_puzzle_solution(pid)[0] for pid in test_ids]
        
        # Get system prompt from prompt manager
        system_prompt = self.prompt_manager.get_prompt(params['prompt_title'])
        
        # Process each test puzzle
        predictions = []
        for puzzle in test_puzzles:
            # Generate prompt using prompt manager
            prompt = SED_user_prompt_generator(
                data_dir=self.dataset_path,
                train_test_split=(train_ids, test_ids),
                give_explanation_flag=params['give_explanation']
            )
            try:
                # Get LLM response with proper system prompt
                response = self.client.beta.chat.completions.parse(
                            model=params['model'],
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            response_format=Solutions
                        )
                
                # Log interaction
                self.log_llm_response(experiment_id, run_id, system_prompt, prompt, response.choices[0].message.content)
                
                # Process prediction
                prediction = self.process_llm_response(response.choices[0].message.content, puzzle)
            except Exception as e:
                self.log_llm_response(experiment_id, run_id, system_prompt, prompt, f"Error: {str(e)}")
            prediction = {
                'puzzle_id': puzzle['problem_id'],
                'is_valid': 0,
                'pattern_score': puzzle['pattern_score']
            }
            predictions.append(prediction)
        
        # Save run results CSV
        run_results_df = pd.DataFrame([{
            'puzzle_id': p['puzzle_id'],
            'is_valid': p['is_valid']
        } for p in predictions])
        run_results_df.to_csv(run_dir / "results.csv", index=False)
        
        return predictions, run_dir

    def run_experiment(self, experiment_id: str, run_id: int, 
                      train_ids: List[int], test_ids: List[int], 
                      params: Dict[str, Any]) -> Dict[str, Any]:
        """Run single experiment and calculate metrics"""
        # Run the core LLM interaction and get predictions
        predictions, _ = self.run_run(experiment_id, run_id, train_ids, test_ids, params)
        
        # Calculate run metrics for experiment tracking
        valid_count = sum(p['is_valid'] for p in predictions)
        pattern_weighted_sum = sum(p['is_valid'] * p['pattern_score'] for p in predictions)
        total_pattern_score = sum(p['pattern_score'] for p in predictions)
        
        return {
            'run_id': run_id,
            'run_accuracy': valid_count / len(predictions),
            'pattern_weighted_accuracy': pattern_weighted_sum / total_pattern_score if total_pattern_score > 0 else 0
        }
    
    def save_experiment_results(self, exp_dir: Path, run_results: List[Dict], params: Dict):
        """Save experiment results to CSV"""
        results_df = pd.DataFrame(run_results)
        results_df = pd.concat([
            results_df,
            pd.DataFrame([params] * len(run_results))
        ], axis=1)
        results_df.to_csv(exp_dir / "results.csv", index=False)

    def save_study_results(self, experiments_results: List[Dict]):
        """Save study-level results"""
        results_dir = self.study_dir / "Results"
        results_dir.mkdir(exist_ok=True)
        
        df = pd.DataFrame(experiments_results)
        df.to_csv(results_dir / "Results.csv", index=False)

    def process_llm_response(self, response_text: str, puzzle: Dict) -> Dict:
        """Process LLM response and validate solution"""
        error_log = self.study_dir / "validation_errors.log"
        
        try:
            # Parse response using fallback parser
            parsed_response = parse_solution_with_fallback(response_text)
            
            # Validate solution sequence
            is_valid = validate_solution_sequence(
                puzzle['initial_string'],
                puzzle['transitions'],
                parsed_response.solutions[0].solution
            )
            
            return {
                'puzzle_id': puzzle['problem_id'],
                'is_valid': int(is_valid),  # Convert bool to 0/1
                'pattern_score': puzzle['pattern_score']
            }
            
        except Exception as e:
            # Log error details
            error_msg = (f"Validation error for puzzle {puzzle['problem_id']}\n"
                        f"Error: {str(e)}\n"
                        f"Response text: {response_text}\n"
                        "----------------------------------------\n")
            with open(error_log, "a") as f:
                f.write(error_msg)
                
            # Return invalid result on any error
            return {
                'puzzle_id': puzzle['problem_id'],
                'is_valid': 0,
                'pattern_score': puzzle['pattern_score']
            }
        
def generate_zero_shot_splits(count: int = None) -> List[Tuple[List[int], List[int]]]:
    """Generate zero-shot splits where each test set has one puzzle and training is empty
    
    Args:
        count: Number of problems to use (uses all if None)
    """
    # Read puzzle IDs from dataset
    df = pd.read_csv(Path(data_dir) /"exploration_results.csv")
    puzzle_ids = df["puzzle_id"].tolist()
    
    # Use only first count problems if specified
    if count is not None:
        puzzle_ids = puzzle_ids[:count]
    
    # Create splits: empty train set, single puzzle test set
    zero_shot_splits = []
    for puzzle_id in puzzle_ids:
        train_ids = []  # Empty training set
        test_ids = [puzzle_id]  # Single puzzle test set
        zero_shot_splits.append((train_ids, test_ids))
    
    return zero_shot_splits

def generate_few_shot_splits(group_size: int = 5, count: int = None) -> List[Tuple[List[int], List[int]]]:
    """Generate few-shot splits with reciprocal train/test groups
    
    Args:
        group_size: Number of puzzles in each training/testing group
        count: Number of problems to use (uses all if None)
        
    Returns:
        List of (train_ids, test_ids) tuples for each split
    """
    # Read puzzle IDs
    df = pd.read_csv(Path(data_dir) /"exploration_results.csv")
    puzzle_ids = df["puzzle_id"].tolist()
    
    # Use only first count problems if specified
    if count is not None:
        puzzle_ids = puzzle_ids[:count]
    
    # Shuffle puzzle IDs
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(puzzle_ids)
    
    # Divide into groups of group_size
    groups = [puzzle_ids[i:i + group_size] for i in range(0, len(puzzle_ids), group_size)]
    
    # Generate reciprocal splits
    few_shot_splits = []
    for p in range(len(groups)//2):
        i = 2*p
        j = 2*p + 1
        few_shot_splits.append((groups[i], groups[j]))
        few_shot_splits.append((groups[j], groups[i]))
    
    return few_shot_splits

# Example usage
if __name__ == "__main__":
    data_dir = Path("MIX_2_3_5_SED_20")
    pipeline = SEDPipeline(data_dir)

    # Use only first 3 problems
    # splits = generate_zero_shot_splits(count=4)
    splits = generate_few_shot_splits(group_size=2, count=12)
    
    print(len(splits))
    
    test_params = {
        'prompt_titles': ['go_step_by_step'],
        'give_explanation_flags': [1],
        'ask_explanation_flags': [1],
        'models': ['gpt-4o']
    }
    
    pipeline.run_study(splits, test_params)