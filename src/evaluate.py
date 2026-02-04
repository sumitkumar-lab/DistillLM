import time, torch
from transformers import BertTokenizer
from data.dataset import IntentDataset
from models.student import StudentIntentModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
labels = ["LIGHT_ON","LIGHT_OFF","AC_ON","AC_OFF","PLAY_MUSIC","STOP_MUSIC"]
label2id = {l:i for i,l in enumerate(labels)}

dataset = IntentDataset("data/processed/test.json", tokenizer, label2id)
model = StudentIntentModel(tokenizer.vocab_size, len(labels))
model.load_state_dict(torch.load("student.pt"))
model.eval()

correct = 0
for ids, mask, y in dataset:
    with torch.no_grad():
        pred = model(ids.unsqueeze(0)).argmax(1)
    correct += int(pred == y)

print("Accuracy:", correct / len(dataset))

# latency
x = dataset[0][0].unsqueeze(0)
start = time.time()
for _ in range(100):
    model(x)
print("Latency(ms):", (time.time()-start)/100*1000)
