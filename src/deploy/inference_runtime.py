import torch
from transformers import BertTokenizer

# Load tokenizer (could also be replaced with custom vocab)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load TorchScript model
model = torch.jit.load("student_intent.pt")
model.eval()

INTENTS = [
    "LIGHT_ON",
    "LIGHT_OFF",
    "AC_ON",
    "AC_OFF",
    "PLAY_MUSIC",
    "STOP_MUSIC"
]

def predict(text):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    logits = model(enc["input_ids"])
    intent_id = logits.argmax(dim=1).item()
    return INTENTS[intent_id]

if __name__ == "__main__":
    while True:
        cmd = input("Command: ")
        print("→ Intent:", predict(cmd))
