from openai import OpenAI
from pydantic import BaseModel
import logging
from utils import *
from global_var import *
from parse_utils import *
from schema import *
from pathlib import Path
import glob
import pandas as pd
import os
from datetime import datetime

client = OpenAI()
OpenAI.api_key = "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"

# Set up logging
logging.basicConfig(filename='Openai_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Example conversation
system_prompt = "Provide solutions for test puzzles."
engine = "gpt-4o"

def run_combinations(data_dir):    
    oneshot_dirs = sorted(glob.glob(str(data_dir / "OneShot*")))
    current_run_ids = []  # Track IDs from this run
    
    for train_dir in oneshot_dirs:
        for test_dir in oneshot_dirs:
            if train_dir == test_dir:
                continue
                
            print(f"Training on {train_dir}, Testing on {test_dir}")
            
            user_prompt = user_prompt_generator(train_dir, test_dir)
            
            completion = client.beta.chat.completions.parse(
                model=engine,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=Solutions,
            )

            response_text = str(completion.choices[0].message.parsed)
            parsed_response = parse_solutions_string(response_text)
            log_conversation(engine, user_prompt, parsed_response)

            output_id = get_output_id()
            current_run_ids.append(output_id)  # Store ID
            
            store_output(
                output_id=output_id,
                engine=engine,
                system_prompt=system_prompt,
                train_path=train_dir,
                test_path=test_dir,
                solutions_obj=parsed_response
            )
    
    return current_run_ids

def analyze_results(current_run_ids):
    results_file = Path("/root/PreCog/NewResults.csv")
    if not results_file.exists():
        print("No results found")
        return None, None
        
    # Read results into DataFrame
    df = pd.read_csv(results_file)
    
    # Filter for current run only
    df = df[df['output_id'].isin(current_run_ids)]
    
    # Skip where train=test
    df = df[df['train_set'] != df['test_set']]
    
    # Clean directory names for display
    df['train_set'] = df['train_set'].apply(lambda x: os.path.basename(x))
    df['test_set'] = df['test_set'].apply(lambda x: os.path.basename(x))
    
    # Training set performance
    train_pivot = pd.pivot_table(
        df,
        values='accuracy',
        index='train_set',
        aggfunc=['mean', 'count']
    ).round(4)
    
    train_pivot.columns = ['Average Accuracy', 'Number of Tests']
    
    # Test set performance
    test_pivot = pd.pivot_table(
        df,
        values='accuracy', 
        index='test_set',
        aggfunc=['mean', 'count']
    ).round(4)
    
    test_pivot.columns = ['Average Accuracy', 'Number of Tests']
    
    print("\nPerformance as Training Set:")
    print(train_pivot)
    print("\nPerformance as Test Set:")
    print(test_pivot)
    
    return df, train_pivot, test_pivot

if __name__ == "__main__":
    data_dir = Path("/root/PreCog/data/OneShot/t_variation/E7")
    current_run_ids = run_combinations(data_dir)
    results_df, train_results, test_results = analyze_results(current_run_ids)
    
    if results_df is not None:
        # Save main results
        save_results_to_csv(results_df)