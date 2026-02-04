import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data.dataset import IntentDataset
from models.teacher import TeacherIntentModel
from models.student import StudentIntentModel

labels = ["LIGHT_ON","LIGHT_OFF","AC_ON","AC_OFF","PLAY_MUSIC","STOP_MUSIC"]
label2id = {l:i for i,l in enumerate(labels)}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = IntentDataset("data/processed/train.json", tokenizer, label2id)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

teacher = TeacherIntentModel(len(labels))
teacher.load_state_dict(torch.load("teacher.pt"))
teacher.eval()

student = StudentIntentModel(tokenizer.vocab_size, len(labels))
opt = torch.optim.Adam(student.parameters(), lr=3e-4)

def kd_loss(s, t, y, T=4.0, alpha=0.6):
    ce = F.cross_entropy(s, y)
    kd = F.kl_div(
        F.log_softmax(s/T, dim=1),
        F.softmax(t/T, dim=1),
        reduction="batchmean"
    )
    return alpha*T*T*kd + (1-alpha)*ce

for epoch in range(10):
    for ids, mask, y in loader:
        with torch.no_grad():
            t_logits = teacher(ids, mask)
        s_logits = student(ids)
        loss = kd_loss(s_logits, t_logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

torch.save(student.state_dict(), "student.pt")


labels = ["LIGHT_ON","LIGHT_OFF","AC_ON","AC_OFF","PLAY_MUSIC","STOP_MUSIC"]
label2id = {l:i for i,l in enumerate(labels)}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = IntentDataset(
    "data/processed/train_llm_soft.json",
    tokenizer,
    label2id,
    use_soft=True
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

student = StudentIntentModel(tokenizer.vocab_size, len(labels))
opt = torch.optim.Adam(student.parameters(), lr=3e-4)

def llm_kd_loss(student_logits, soft_targets, T=2.0):
    student_logp = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(student_logp, soft_targets, reduction="batchmean") * T * T

for epoch in range(10):
    for ids, soft_targets in loader:
        logits = student(ids)
        loss = llm_kd_loss(logits, soft_targets)

        opt.zero_grad()
        loss.backward()
        opt.step()

torch.save(student.state_dict(), "student_llm.pt")
