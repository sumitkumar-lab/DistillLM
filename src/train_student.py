import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data.dataset import IntentDataset
from models.student import StudentIntentModel


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
