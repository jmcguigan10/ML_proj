import os, random, numpy as np, torch

def set_deterministic(flag=True):
    torch.backends.cudnn.deterministic = bool(flag)
    torch.backends.cudnn.benchmark = not bool(flag)

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)

