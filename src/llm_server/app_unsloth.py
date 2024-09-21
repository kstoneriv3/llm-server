# Triton needs compute capability >= 7.0

from flask import Flask, request
import torch
from unsloth import FastLanguageModel


# Create a custom LLM class
class LocalLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=100, use_cache=True)
        generated_text = self.tokenizer.batch_decode(outputs)
        return generated_text
        # alpaca_prompt = Copied from above


# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",  # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",  # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",  # Phi-3 2x faster!d
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
    "unsloth/gemma-2-2b-bnb-4bit",  # New small Gemma model!
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-2b",
    max_seq_length=1024,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

local_llm = LocalLLM(model, tokenizer)

# Create a Flask app (optional)
app = Flask(__name__)
app.config["HOST"] = "localhost"
app.config["PORT"] = 3000


@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json["prompt"]
    response = local_llm.generate(prompt)
    return {"response": response}


def main():
    app.run(debug=True)


# Example usage
if __name__ == "__main__":
    response = local_llm.generate("Tell me a joke")
    print(response)
