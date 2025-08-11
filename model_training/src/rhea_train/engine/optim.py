import time, numpy as np, torch, torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

def _metrics_at_point(logits, y):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == y).sum().item()
    tp = ((preds==1) & (y==1)).sum().item()
    fp = ((preds==1) & (y==0)).sum().item()
    fn = ((preds==0) & (y==1)).sum().item()
    acc  = correct / y.numel()
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return acc, prec, rec, f1

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tloss = tsamp = 0
    tcorr = tp = fp = fn = 0
    for Xb, yb, wb in loader:
        Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
        logits = model(Xb)
        loss   = (criterion(logits, yb)*wb).mean()/wb.mean()
        bs = Xb.size(0)
        tloss += float(loss.item()) * bs
        tsamp += bs
        # metrics @ 0.5
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        tcorr += (preds==yb).sum().item()
        tp    += ((preds==1)&(yb==1)).sum().item()
        fp    += ((preds==1)&(yb==0)).sum().item()
        fn    += ((preds==0)&(yb==1)).sum().item()
    loss = tloss / max(1, tsamp)
    acc  = tcorr / max(1, tsamp)
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return loss, acc, prec, rec, f1

def f1_sweep(model, loader, device):
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for Xb, yb, _ in loader:
            all_probs.append(torch.sigmoid(model(Xb.to(device))).cpu().numpy())
            all_y.append(yb.numpy())
    probs = np.concatenate(all_probs).ravel()
    y_true= np.concatenate(all_y).ravel()
    best = dict(thr=-1, prec=0.0, rec=0.0, f1=-1.0)
    for thr in np.linspace(0.01, 0.99, 99):
        preds = (probs >= thr).astype(np.int8)
        tp = np.sum((preds==1)&(y_true==1))
        fp = np.sum((preds==1)&(y_true==0))
        fn = np.sum((preds==0)&(y_true==1))
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec  = tp/(tp+fn) if tp+fn else 0.0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        if f1 > best["f1"]:
            best.update(thr=float(thr), prec=float(prec), rec=float(rec), f1=float(f1))
    return best

def train_once(
    model, train_loader, test_loader, *,
    lr: float, weight_decay: float,
    epochs: int, patience: int,
    warmup_epochs: int,
    plate_factor: float, plate_patience: int, plate_cooldown: int,
    device: str = "cuda"
):
    device = torch.device(device)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # schedulers
    warmup = LambdaLR(optim, lr_lambda=lambda ep: (ep+1)/warmup_epochs if ep < warmup_epochs else 1.0)
    plateau= ReduceLROnPlateau(optim, mode="min", factor=plate_factor, patience=plate_patience, cooldown=plate_cooldown)

    history = []
    best = float("inf"); bad = 0

    for ep in range(epochs):
        model.train()
        tloss_sum = tsamp = 0
        for Xb, yb, wb in train_loader:
            Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss   = (criterion(logits,yb)*wb).mean()/wb.mean()
            loss.backward()
            optim.step()
            bs = Xb.size(0)
            tloss_sum += float(loss.item()) * bs
            tsamp     += bs

        if ep < warmup_epochs:
            warmup.step()

        train_loss = tloss_sum / max(1, tsamp)
        test_loss, acc, prec, rec, f1 = evaluate(model, test_loader, criterion, device)
        history.append([ep, train_loss, test_loss, acc, prec, rec, f1])

        if test_loss < best: best, bad = test_loss, 0
        else:
            bad += 1
            if bad >= patience: break

        if ep >= warmup_epochs:
            plateau.step(test_loss)

    # F1 sweep
    best_sweep = f1_sweep(model, test_loader, device)
    return np.array(history, dtype=np.float32), best_sweep

