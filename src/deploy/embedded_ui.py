import streamlit as st
import torch
from transformers import BertTokenizer

# Load model
model = torch.jit.load("student_intent.pt")
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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
    with torch.no_grad():
        logits = model(enc["input_ids"])
    return INTENTS[logits.argmax(dim=1).item()]

# UI
st.title("Embedded Intent Classification Demo")
st.caption("On-device inference using distilled model (TorchScript)")

user_input = st.text_input("Enter command", "turn on the headlights")

if st.button("Run Inference"):
    intent = predict(user_input)
    st.success(f"Predicted Intent: {intent}")
