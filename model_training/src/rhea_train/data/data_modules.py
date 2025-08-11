import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from .npz_dataset import load_concat_npz
from .transforms import fit_standard_scaler, apply_standard_scaler

def build_loaders(data_cfg, seed:int, batch_size:int, random_multiplier:int,
                  num_workers:int=None, pin_memory:bool=None):
    num_workers = data_cfg.get("num_workers", 4) if num_workers is None else num_workers
    pin_memory  = data_cfg.get("pin_memory", True) if pin_memory is None else pin_memory

    X_full, y_full, random_slice = load_concat_npz(data_cfg)

    weights = np.ones(len(X_full), dtype=np.float32)
    weights[random_slice] = float(random_multiplier)

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X_full, y_full, weights, test_size=data_cfg.get("test_size",0.2), random_state=seed)

    scaler, X_tr = fit_standard_scaler(X_tr)
    X_te = apply_standard_scaler(scaler, X_te)

    # torch tensors (y shape Nx1 like your code)
    X_tr = torch.from_numpy(X_tr.astype(np.float32))
    y_tr = torch.from_numpy(y_tr.astype(np.float32)).view(-1,1)
    w_tr = torch.from_numpy(w_tr.astype(np.float32)).view(-1,1)

    X_te = torch.from_numpy(X_te.astype(np.float32))
    y_te = torch.from_numpy(y_te.astype(np.float32)).view(-1,1)
    w_te = torch.from_numpy(w_te.astype(np.float32)).view(-1,1)

    g = torch.Generator().manual_seed(seed)
    train_ds = TensorDataset(X_tr, y_tr, w_tr)
    test_ds  = TensorDataset(X_te, y_te, w_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=g, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader

