import torch
from models.student import StudentIntentModel
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = StudentIntentModel(tokenizer.vocab_size, 6)
model.load_state_dict(torch.load("student.pt"))
model.eval()

example = torch.randint(0, tokenizer.vocab_size, (1, 32))
traced = torch.jit.trace(model, example)
traced.save("student_intent.pt")
