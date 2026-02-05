import time
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = torch.jit.load("student_intent.pt")
model.eval()

text = "turn on the headlights"
enc = tokenizer(text, return_tensors="pt")

runs = 1000
start = time.time()
for _ in range(runs):
    model(enc["input_ids"])
latency = (time.time() - start) / runs * 1000

print(f"Average latency: {latency:.2f} ms")
