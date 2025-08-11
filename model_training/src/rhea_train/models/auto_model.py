import torch.nn as nn
from .registry import register
from .nn_blocks import DenseBNReLUDrop

@register("auto_model")
class AutoModel(nn.Module):
    """
    Your BinaryClassifier from full_nn.py:
    Linear -> BN -> ReLU -> Dropout (num_layers times), then Linear -> 1
    """
    def __init__(self, input_size: int, num_layers: int, hidden_size: int, dropout: float, **_):
        super().__init__()
        layers = [DenseBNReLUDrop(input_size, hidden_size, p=dropout)]
        for _ in range(num_layers - 1):
            layers.append(DenseBNReLUDrop(hidden_size, hidden_size, p=dropout))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_size, 1)
    def forward(self, x): return self.out(self.net(x))
