import torch
from torch import nn

## https://discuss.pytorch.org/t/rmsle-loss-function/67281
class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, actual):
        return self.mse(torch.log(pred+1), torch.log(actual+1))