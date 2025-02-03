import google.generativeai as genai
import logging
import typing_extensions as typing

genai.configure(api_key="AIzaSyDXGeYv-7dIXg_k3bJWtswkI94rlqi-IUE")

logging.basicConfig(filename='Gemini_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# def log_conversation(engine, prompt, response):
#     logging.info(f"Engine: {engine}")
#     logging.info(f"Prompt: {prompt}")
#     logging.info(f"Response: {response}")

# Define the Recipe schema
class Recipe(typing.TypedDict):
    recipe_name: str
    ingredients: list[str]

# Example conversation
prompt = "List a few popular cookie recipes."
engine = "gemini-1.5-pro-latest"
model = genai.GenerativeModel(engine)
result = model.generate_content(
    prompt,
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json", response_schema=list[Recipe]
    ),
)

print(result)
print(type(result))


# print(response.text)

# log_conversation(engine, prompt, response.text)