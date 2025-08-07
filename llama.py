from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="mps")

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)