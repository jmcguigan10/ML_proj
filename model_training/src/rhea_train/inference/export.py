import argparse, torch
from rhea_train.utils.config import load_config
from rhea_train.models.registry import create_model

def export_torchscript(ckpt, cfg_path, out_path, example_shape):
    cfg = load_config(cfg_path)
    model = create_model(**cfg.model.__dict__)
    state = torch.load(ckpt, map_location="cpu")
    if "model" in state: state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    ex = torch.randn(*example_shape)
    with torch.no_grad():
        ts = torch.jit.trace(model, ex)
    ts.save(out_path)
    print(f"Saved TorchScript to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="../../example_model.pt")  # repo root default
    ap.add_argument("--example", default="1,128")               # e.g., "1,128"
    a = ap.parse_args()
    shape = tuple(int(x) for x in a.example.split(","))
    export_torchscript(a.ckpt, a.config, a.out, shape)
