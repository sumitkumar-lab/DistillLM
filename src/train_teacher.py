import torch, json
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data.dataset import IntentDataset
from models.teacher import TeacherIntentModel

labels = ["LIGHT_ON","LIGHT_OFF","AC_ON","AC_OFF","PLAY_MUSIC","STOP_MUSIC"]
label2id = {l:i for i,l in enumerate(labels)}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = IntentDataset("data/processed/train.json", tokenizer, label2id)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = TeacherIntentModel(len(labels))
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(3):
    for ids, mask, y in loader:
        logits = model(ids, mask)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

torch.save(model.state_dict(), "teacher.pt")
