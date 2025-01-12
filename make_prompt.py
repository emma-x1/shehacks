import json
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline)

config_data = json.load(open('config.json'))
HF_TOKEN = config_data['HF_TOKEN']

#config data
model_id = "meta-llama/Llama-3.2-1B"

#quantisation config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

#tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

#model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    quantization_config=bnb_config,
    token=HF_TOKEN,
    )

#pipeline
messages = [
    {"role": "user", "content": "Generate a text message prompt for a user to answer. This text message prompt should be a typical question asked in a specific type of setting. The user should be asked to provide a response to the question. If the indicated level is low, the question should be a typical, easy-to-answer question, while if the level is high, the question should require more thought, be more unexpected, and be more creative."},
]

pipe = pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    )

outputs = pipe(
    messages,
    max_new_tokens=256,)

print(outputs[0]['generated_text'])[-1]
