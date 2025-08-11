import torch

@torch.no_grad()
def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean()
