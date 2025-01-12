#functions:
#make_prompt: take in level (difficulty), setting (professional, friendly, romantic), produce user prompt
#rate_response: take in setting, user response, produce score and feedback

import json
from llamaapi import LlamaAPI

"""
Recommended Flow
Send query and function definitions to the model
Model returns JSON adhering to function schema (if it chooses to call one)
Parse the JSON
Validate it
Call the function
Send function result back to model to summarize for user
"""

# Initialize the SDK
llama = LlamaAPI("LA-f425ee57f11749b18d6b142476f159350e1531c9d17a4494bb518853aa19d8ea")

# Build the API request
api_request_json = {
    "model": "llama3.1-70b",
    "messages": [
        {"role": "user", "content": f"Generate a text message prompt for a user to answer. This text message prompt should be a typical question asked in a specific type of setting. The user should be asked to provide a response to the question. If the indicated level is low, the question should be a typical, easy-to-answer question, while if the level is high, the question should require more thought, be more unexpected, and be more creative."},
    ],
    "functions": [
        {
            "name": "generate_text_prompt",
            "description": "Make a text prompt for the user to answer to test their communication skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "setting": {
                        "type": "string",
                        "description": "The setting or context under which this text would be sent, ie. professional, romantic, friendly.",
                    },
                    "level": {
                        "type": "number",
                        "description": "How difficult the prompt shoul be, ie. 1, 2, 3. For a higher level, the prompt should be more creative and provoke more thought in the user (ie. ask harder, deeper questions).",
                    },
                },
            },
            "max_token": 500,
            "temperature": 0.1,
            "required": ["setting", "level"],
        }
    ],
    "stream": False,
    "function_call": "make_query",
}

# Execute the Request
response = llama.run(api_request_json)
print(json.dumps(response.json(), indent=2))

"""

def make_query(level:int, setting:str -> query:str):
    #level: 1, 2, 3
    #setting: professional, friendly, romantic
    #call API

#extract data from JSON model response

def extract_query_data(query: str)

"""