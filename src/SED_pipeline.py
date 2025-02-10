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

    def get_puzzle_solution(self, problem_id: int) -> Tuple[Dict, Dict]:
        """Load puzzle and solution files for given problem_id"""
        puzzle_file = self.puzzles_dir / f"{problem_id:03d}.json"
        solution_file = self.solutions_dir / f"{problem_id:03d}.json"
        
        with open(puzzle_file) as pf, open(solution_file) as sf:
            puzzle = json.load(pf)
            solution = json.load(sf)
        return puzzle, solution

    def run_study(self,
                  train_test_splits: List[Tuple[List[int], List[int]]],
                  for_test: Dict[str, List[Any]]):
        """Run study with given splits and test parameters"""
        # Generate parameter combinations for experiments
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

    def run_experiment(self, experiment_id: str, run_id: int, 
                      train_ids: List[int], test_ids: List[int], 
                      params: Dict[str, Any]) -> Dict[str, Any]:
        """Run single experiment with given parameters"""
        run_dir = self.study_dir / "Experiments" / f"Experiment{experiment_id}" / f"Run{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training data
        train_data = [self.get_puzzle_solution(pid) for pid in train_ids]
        test_puzzles = [self.get_puzzle_solution(pid)[0] for pid in test_ids]
        
        # Process each test puzzle
        predictions = []
        for puzzle in test_puzzles:
            # Generate prompt using prompt manager
            prompt = self.prompt_manager.generate_user_prompt(
                puzzle, 
                train_data
            )
            
            # Get LLM response
            response = self.client.chat.completions.create(
                model=params['model'],
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Log interaction
            self.log_llm_response(experiment_id, run_id, prompt, response.choices[0].message.content)
            
            # Process prediction
            prediction = self.process_llm_response(response.choices[0].message.content, puzzle)
            predictions.append(prediction)
        
        # Save run results
        results_df = pd.DataFrame(predictions)
        results_df.to_csv(run_dir / "predictions.csv", index=False)
        
        # Calculate run metrics
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
            # Return invalid result on any error
            return {
                'puzzle_id': puzzle['problem_id'],
                'is_valid': 0,
                'pattern_score': puzzle['pattern_score']
            }
        

        
# Example usage
if __name__ == "__main__":
    pipeline = SEDPipeline("SED_100/data")
    
    splits = [
        ([6, 16, 17], [35, 44, 50]),
        ([35, 44, 50], [6, 16, 17])
    ]
    
    # Example test parameters
    test_params = {
        'prompt_titles': ['baseline', 'go_step_by_step'],
        'give_explanation_flags': [0, 1],
        'ask_explanation_flags': [0],
        'models': ['gpt-4']
    }
    
    pipeline.run_study(splits, test_params)