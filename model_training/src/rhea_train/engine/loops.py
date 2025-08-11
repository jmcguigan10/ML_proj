import torch
from ..metrics.classification import accuracy

def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, amp=False, grad_clip=None, logger=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    for step, batch in enumerate(loader):
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()
        if scheduler: scheduler.step()
        if logger and step % 25 == 0:
            acc = float(accuracy(logits, y).item())
            logger.log(step=step, train_loss=float(loss.item()), train_acc=acc)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, logger=None, epoch=None):
    model.eval()
    n, loss_sum, acc_sum = 0, 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(loss_fn(logits, y).item())
        acc_sum += float(accuracy(logits, y).item())
        n += 1
    logs = {"val/loss": loss_sum/max(n,1), "val/acc": acc_sum/max(n,1)}
    if logger: logger.log(epoch=epoch, **logs)
    return logs

