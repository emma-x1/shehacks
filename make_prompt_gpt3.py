from pydantic import BaseModel
from openai import OpenAI
import json

config_data = json.load(open('config.json'))
OPENAI_KEY = config_data['OPENAI_KEY']

client = OpenAI(
  api_key=OPENAI_KEY
)

class Prompt(BaseModel):
    output: str

class Review(BaseModel):
    score: int
    feedback: str

def make_prompt(setting:str, level:int):
    prompt = f"Ask me a conversational question appropriate for a {setting} setting. This should be a level {level} question, where level 1 indicates a straightforward, typical question, and higher levels indicate more complex questions."
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "user", "content": prompt}
        ], 
        response_format=Prompt,
    )
    return completion.choices[0].message.content

def evaluate_response(question:str, response:str):
    prompt = f"Based on this question: {question}, how would you rate this response: {response}? Give it a score out of 10, where 1 is a very poor response and 10 is an excellent response. Then, give specific feedback on what you liked or didn't like about the response, as well as ways in which it could be improved."
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "user", "content": prompt}
        ], 
        response_format=Review,
    )
    return completion.choices[0].message.content

print(make_prompt("professional", 3))
print(evaluate_response("What is your favorite color?", "My favorite color is red"))