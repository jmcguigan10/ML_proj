import torch, math, numpy as np
from ..registry import register_generator

_GENERATORS = {}
def register_generator(name):
    def deco(fn):
        _GENERATORS[name] = fn
        return register_generator
    return deco

def create_generator(name): return _GENERATORS[name]

@register_generator("gauss_blobs")
def gauss_blob(idx: int, n_classes: int = 3, dim: int = 2, std: float = 0.5, seed: int = 1337):
    rng = np.random.RandomState(seed + idx)
    label = rng.randint(0, n_classes)
    angle = 2 * math.pi * label / n_classes
    center = np.array([math.cos(angle), math.sin(angle)] + [0]*(dim-2), dtype=np.float32)
    x = center + rng.randn(dim).astype(np.float32) * std
    return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)
