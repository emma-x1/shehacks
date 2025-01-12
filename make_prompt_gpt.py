from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

def make_prompt(setting:str, level:int):
    #prompt = f"Ask a conversational question appropriate for a {setting} setting. This should be a typical, everyday question of level {level}, where a higher level corresponds to a harder question."
    prompt = f"Ask me a question appropriate for a {setting} setting."

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    attention_mask = torch.ones(inputs['input_ids'].shape, dtype=torch.long)

    gen_tokens = model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,
        do_sample=True,
        temperature=0.9,
        max_length=100,
        pad_token_id=model.config.pad_token_id,
    )

    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    return(gen_text)

print(make_prompt("professional", 3))