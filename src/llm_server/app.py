from flask import Flask, request
from functools import cache
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Create a custom LLM class
class LocalLLM:
    def __init__(self, model_name):
        self.model_name = model_name

    @property
    @cache
    def model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to("cuda")

    @property
    @cache
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=1024, use_cache=True)
        generated_text = self.tokenizer.batch_decode(outputs)
        return generated_text


local_llm = LocalLLM("google/gemma-2-2b-it")

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
