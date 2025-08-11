import torch.nn.functional as F

def create_loss(name: str = "cross_entropy", **kw):
    name = name.lower()
    if name in ("ce", "cross_entropy"): 
        return lambda logits, y: F.cross_entropy(logits, y)
    raise ValueError(f"Unknown loss {name}")
