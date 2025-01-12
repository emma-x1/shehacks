import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
messages = [
    {"role": "user", "content": "Generate a text message prompt for a user to answer. This text message prompt should be a typical question asked in a specific type of setting. The user should be asked to provide a response to the question. If the indicated level is low, the question should be a typical, easy-to-answer question, while if the level is high, the question should require more thought, be more unexpected, and be more creative."},
]

pipe = pipeline(
    "text-generation", 
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    )

outputs = pipe(
    messages,
    max_new_tokens=256,)

print(outputs[0]['generated_text'])[-1]
