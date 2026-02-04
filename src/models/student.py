import torch.nn as nn

class StudentIntentModel(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256
            ),
            num_layers=2
        )
        self.fc = nn.Linear(128, num_labels)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = x.transpose(0,1)
        x = self.encoder(x)
        return self.fc(x[0])
