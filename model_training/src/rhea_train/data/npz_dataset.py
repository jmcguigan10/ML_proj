import os, numpy as np

def load_concat_npz(data_cfg):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    root = data_cfg.get("root_dir",".")
    root = root if os.path.isabs(root) else os.path.join(base, root)
    npz_paths = {k: os.path.join(root, v) for k,v in data_cfg["npz_files"].items()}
    kx, ky = data_cfg["keys"]["X"], data_cfg["keys"]["y"]
    n = int(data_cfg.get("n_samples", 200000))

    d1 = np.load(npz_paths["zerofluxfac"]); X1, y1 = d1[kx["zerofluxfac"]][:n], d1[ky["zerofluxfac"]][:n]
    d2 = np.load(npz_paths["oneflavor"]);   X2, y2 = d2[kx["oneflavor"]][:n],   d2[ky["oneflavor"]][:n]
    d3 = np.load(npz_paths["random"]);      X3, y3 = d3[kx["random"]][:n],      d3[ky["random"]][:n]
    d4 = np.load(npz_paths["NSM"]);         X4, y4 = d4[kx["NSM"]][:n],         d4[ky["NSM"]][:n]

    X_full = np.concatenate([X1, X2, X3, X4]).astype(np.float32)
    y_full = np.concatenate([y1, y2, y3, y4]).astype(np.float32)

    off_random = len(X1) + len(X2)
    len_random = len(X3)
    random_slice = slice(off_random, off_random + len_random)
    return X_full, y_full, random_slice

