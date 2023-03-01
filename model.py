import os
import torch
import torch.nn as nn
import torchvision.models as models
import unittest
from copy import deepcopy


# TODO être assez générique pour pouvoir prendre n'importe quel backbone et heads (selon config)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        self.backbone.classifier = nn.Sequential(
            self.backbone.classifier[0],
            self.backbone.classifier[1],
            nn.ReLU(),
            nn.Linear(
                1000,
                100,
            ),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.backbone(x)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, name: str):
        model = cls()
        if not name.endswith(".pt"):
            name = f"{name}.pt"
        model.load_state_dict(torch.load(os.path.join(ckpt_path, name)))
        return model

    def save_checkpoint(self, ckpt_path: str, name: str):
        os.makedirs(ckpt_path, exist_ok=True)
        if not name.endswith(".pt"):
            name = f"{name}.pt"
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt = deepcopy(self.state_dict())
        torch.save(ckpt, os.path.join(ckpt_path, name))
