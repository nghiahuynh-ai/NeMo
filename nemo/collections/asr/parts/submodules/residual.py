import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super(Residual, self).__init__()
        self.module = module
        
    def forward(self, x):
        residual = x
        x = self.module(x)
        return residual + x
        