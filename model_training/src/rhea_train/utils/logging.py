class StdoutLogger:
    def log(self, **kw):
        safe = {k: (float(v) if hasattr(v,"item") else v) for k,v in kw.items()}
        print(safe)
def get_logger(_cfg): return StdoutLogger()
