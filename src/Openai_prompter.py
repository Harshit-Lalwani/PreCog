from openai import OpenAI
from pydantic import BaseModel
import logging
from utils import *
from global_var import *
from parse_utils import *
from schema import *

client = OpenAI()
OpenAI.api_key = "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"

# Set up logging
logging.basicConfig(filename='Openai_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Example conversation
system_prompt = """
Provide solutions for test puzzles
"""
train_path = "data/single_3_3_0"
test_path = "data/single_3_3_0"

user_prompt = user_prompt_generator(train_path, test_path)

engine = "gpt-4o"

completion = client.beta.chat.completions.parse(
    model=engine,
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": user_prompt},
    ],
    response_format=Solutions,
)

# Print and log the response
response_text = str(completion.choices[0].message.parsed)
parsed_response = parse_solutions_string(response_text)
# print(parsed_response)
log_conversation(engine, user_prompt, parsed_response)

# # Apply store_output
store_output(
    output_id= get_output_id(),  # Provide a suitable output_id
    engine=engine,
    system_prompt=system_prompt,
    train_path=train_path,
    test_path=test_path,
    solutions_obj=parsed_response
)