import torch
import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.mse(logits, targets)
