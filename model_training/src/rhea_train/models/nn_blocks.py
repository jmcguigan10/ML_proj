import torch.nn as nn
class DenseBNReLUDrop(nn.Module):
    def __init__(self, in_dim, out_dim, p=0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.bn  = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop= nn.Dropout(p) if p>0 else nn.Identity()
    def forward(self, x): return self.drop(self.act(self.bn(self.lin(x))))

