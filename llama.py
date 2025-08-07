from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)

messages = [
    {"role": "user", "content": "Who are you?"},
]

response = pipe(messages, max_new_tokens=100)
print(response[0]["generated_text"])