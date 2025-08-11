# placeholder for EarlyStopping, Checkpoint, LR schedulers

# src/train/callbacks.py
class EarlyStopping:
    def __init__(self, patience=10, mode="max"):
        self.patience = int(patience)
        self.mode = mode
        self.best = None
        self.wait = 0

    def step(self, metric_value: float) -> bool:
        score = metric_value if self.mode == "max" else -metric_value
        if self.best is None or score > self.best:
            self.best = score
            self.wait = 0
            return True  # mejoró
        self.wait += 1
        return False

    def should_stop(self) -> bool:
        return self.wait >= self.patience
