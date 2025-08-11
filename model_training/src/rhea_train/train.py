import argparse, os, numpy as np, torch
from rhea_train.utils.config import load_config
from rhea_train.utils.randomness import seed_all, set_deterministic
from rhea_train.utils.logging import get_logger
from rhea_train.data.datamodules import build_loaders
from rhea_train.models.registry import create_model
from rhea_train.engine.trainer_bce import train_once

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    # overrides (like your grid)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--num_layers", type=int, default=None)
    ap.add_argument("--hidden_size", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--random_multiplier", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_all(args.seed); set_deterministic(cfg.train.get("deterministic", True))
    logger = get_logger(cfg.logging)

    # loaders
    bs = args.bs or cfg.data.batch_size
    tr_loader, te_loader = build_loaders(cfg.data.__dict__, seed=args.seed,
                                         batch_size=bs, random_multiplier=args.random_multiplier)

    # model
    mcfg = cfg.model.__dict__.copy()
    if args.num_layers is not None: mcfg["num_layers"] = args.num_layers
    if args.hidden_size is not None: mcfg["hidden_size"] = args.hidden_size
    if args.dropout is not None: mcfg["dropout"] = args.dropout
    model = create_model(name=mcfg.pop("name"), **mcfg)

    # train
    lr = args.lr or cfg.train["optimizer"]["lr"]
    wd = args.wd or cfg.train["optimizer"]["weight_decay"]
    hist, sweep = train_once(
        model, tr_loader, te_loader,
        lr=lr, weight_decay=wd,
        epochs=cfg.train["epochs"], patience=cfg.train["patience"],
        warmup_epochs=cfg.train["warmup_epochs"],
        plate_factor=cfg.train["plateau"]["factor"],
        plate_patience=cfg.train["plateau"]["patience"],
        plate_cooldown=cfg.train["plateau"]["cooldown"],
        device=("cuda" if torch.cuda.is_available() else "cpu")
    )

    # save checkpoint like your tag
    tag = f"S{args.seed}_L{mcfg['num_layers']}_HS{mcfg['hidden_size']}_DR{mcfg['dropout']}_WD{wd}_BS{bs}_RM{args.random_multiplier}"
    os.makedirs("artifacts/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"artifacts/checkpoints/{tag}.pt")

    # print final metrics like your script
    final = hist[-1]
    print(f"seed={args.seed} rm={args.random_multiplier} layers={mcfg['num_layers']} hs={mcfg['hidden_size']} "
          f"dr={mcfg['dropout']} wd={wd} lr={lr} bs={bs} | "
          f"Loss {final[2]:.4f} Acc {final[3]:.4f} Prec0.5 {final[4]:.4f} Rec0.5 {final[5]:.4f} F1@0.5 {final[6]:.4f} || "
          f"BEST thr={sweep['thr']:.2f} Prec {sweep['prec']:.4f} Rec {sweep['rec']:.4f} F1 {sweep['f1']:.4f} | "
          f"Epochs {int(final[0])} Time N/A")

    # save history if you want
    os.makedirs("artifacts/results", exist_ok=True)
    np.save("artifacts/results/last_run.npy", hist)

if __name__ == "__main__":
    main()

