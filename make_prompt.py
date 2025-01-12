import json
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, AutoConfig)
from accelerate import Accelerator

accelerator = Accelerator()

torch.cuda.empty_cache()

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

model_config = AutoConfig.from_pretrained(model_id, token=HF_TOKEN)
model_config.rope_scaling = {
    "type": "linear",
    "factor": 32.0
}


#model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    config=model_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    #quantization_config=bnb_config,
    token=HF_TOKEN,
    )

model = accelerator.prepare(model)

#pipeline

text_generator = pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.7
    )

def generate_text_prompt(setting:str, level:int):
    message = f"Ask a conversational question appropriate for a {setting} setting." #The question should be easy for level {level}."
    result = text_generator(message)
    return result[0]['generated_text']

print(generate_text_prompt("professional", 1))
