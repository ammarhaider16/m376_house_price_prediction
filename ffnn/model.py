import torch
from torch import nn

class FFNN(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(FFNN, self).__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x