import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1])
