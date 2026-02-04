import torch.nn as nn
from transformers import BertModel

class TeacherIntentModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls)
