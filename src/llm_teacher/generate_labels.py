import json
import openai
from prompt import build_prompt

openai.api_key = "YOUR_API_KEY"

def query_llm(text):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": build_prompt(text)}],
        temperature=0.0
    )
    return json.loads(response.choices[0].message.content)

def generate():
    data = json.load(open("data/processed/train.json"))
    for sample in data:
        sample["soft_labels"] = query_llm(sample["text"])

    json.dump(
        data,
        open("data/processed/train_llm_soft.json", "w"),
        indent=2
    )

if __name__ == "__main__":
    generate()
