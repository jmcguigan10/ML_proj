import os, yaml
from types import SimpleNamespace

def _ns(d):
    if isinstance(d, dict): return SimpleNamespace(**{k: _ns(v) for k,v in d.items()})
    if isinstance(d, list): return [_ns(x) for x in d]
    return d

def load_config(path):
    with open(path, "r") as f: top = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    def load(rel):
        p = rel if os.path.isabs(rel) else os.path.join(base, rel)
        with open(p, "r") as f: return yaml.safe_load(f)
    cfg = {
        "seed": top.get("seed", 43),
        "project_name": top.get("project_name", "rhea_train"),
        "data": load(top["data_cfg"]),
        "model": load(top["model_cfg"]),
        "train": load(top["train_cfg"]),
        "logging": top.get("logging", {"framework": "stdout"}),
    }
    return _ns(cfg)

