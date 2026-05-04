class EarlyStopping:
    """
    Stops training when validation loss has not improved by `min_delta`
    for `patience` consecutive epochs.
    """
    def __init__(self, patience: int = 3, min_delta: float = 1e-3):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0
        self.stopped   = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stopped = True
        return self.stopped