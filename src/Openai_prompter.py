from openai import OpenAI
import logging
from utils import *

client = OpenAI()

# # Set up logging
# logging.basicConfig(filename='Openai_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Example conversation
prompt = "Hello world!"
engine = "gpt-4o-mini"

# Set up logging
logging.basicConfig(filename='Openai_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')


completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Say this is a test"
        }
    ]
)

# Print and log the response
response_text = completion.choices[0].message
print(response_text)
log_conversation(engine, prompt, response_text)