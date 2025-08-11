import os, torch

class Callback:
    def on_step_end(self, **kw): pass
    def on_epoch_end(self, **kw): pass

class CheckpointCallback(Callback):
    def __init__(self, dir: str, save_top_k: int = 3, monitor: str = "val/loss", mode: str = "min"):
        self.dir = dir; os.makedirs(dir, exist_ok=True)
        self.save_top_k = save_top_k; self.monitor = monitor; self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.kept = []  # (score, path)

    def maybe_save(self, model, epoch, logs):
        score = logs.get(self.monitor)
        if score is None: return
        better = (score < self.best) if self.mode == "min" else (score > self.best)
        path = os.path.join(self.dir, f"epoch_{epoch:03d}.pt")
        torch.save(model.state_dict(), path)
        if better:
            self.best = score
            best_path = os.path.join(self.dir, "best.pt")
            torch.save(model.state_dict(), best_path)

class EarlyStopping(Callback):
    def __init__(self, patience: int = 10, monitor: str = "val/loss", mode: str = "min"):
        self.patience = patience; self.monitor = monitor; self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.bad = 0
    def on_epoch_end(self, logs=None, **_):
        if logs is None or self.monitor not in logs: return False
        val = logs[self.monitor]
        improved = (val < self.best) if self.mode == "min" else (val > self.best)
        if improved:
            self.best = val; self.bad = 0
        else:
            self.bad += 1
        return self.bad >= self.patience

def build_callbacks(cfg):
    cbs = []
    ck = cfg.get("ckpt", None)
    if ck: cbs.append(CheckpointCallback(dir=ck["dir"], save_top_k=ck.get("save_top_k", 3),
                                         monitor=cfg.get("early_stop", {}).get("monitor", "val/loss"),
                                         mode="min"))
    return cbs, EarlyStopping(patience=cfg.get("early_stop", {}).get("patience", 10),
                              monitor=cfg.get("early_stop", {}).get("monitor", "val/loss"),
                              mode="min")
