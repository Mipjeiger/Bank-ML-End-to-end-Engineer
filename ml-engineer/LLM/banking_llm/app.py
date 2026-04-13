from transformers import pipeline
from prompt_engine import build_prompt

class BankingLLM:
    def __init__(self):
        self.generator = None
        self.load()

    def load(self):
        self.generator = pipeline(
            "text-generation",
            model="distilgpt2"
        )

    def predict(self, X, names=None):
        question = X[0]
        context = X[0][1]
        prompt = build_prompt(question, context)
        result = self.generator(
            prompt,
            max_length=150,
            do_sample=True,
            temperature=0.7,
        )

        return [result[0]["generated_text"]]