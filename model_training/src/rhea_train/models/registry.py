_MODELS = {}
def register(name):
    def deco(cls): _MODELS[name]=cls; return cls
    return deco
def create_model(name, **kw):
    if name not in _MODELS: raise KeyError(f"Unknown model: {name}")
    return _MODELS[name](**kw)

