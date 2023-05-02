import torch
from torch import nn

in_sz = 28 * 28


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.ln0 = nn.Linear(in_sz, 256)
        self.ln1 = nn.Linear(256, 256)
        self.ln2 = nn.Linear(256, 128)
        self.ln3 = nn.Linear(128, 32)
        self.ln4 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor):
        out0 = self.relu(self.ln0(x))
        out0 = self.relu(self.ln1(out0))
        out0 = self.relu(self.ln2(out0))
        out0 = self.relu(self.ln3(out0))
        out0 = self.softmax(self.ln4(out0))
        return out0
