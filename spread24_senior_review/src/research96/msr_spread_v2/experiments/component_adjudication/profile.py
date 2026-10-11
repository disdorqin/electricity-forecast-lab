"""Frozen V2_GPU_SCIENTIFIC_V1; floors apply to selected refit checkpoints too."""
from dataclasses import replace
from ...config import Config

NAME = "V2_GPU_SCIENTIFIC_V1"
BOUNDS = (("A", 15, 40), ("B", 5, 12), ("C", 10, 30))
ARMS = ("FULL_V2", "DIRECT96", "NO_DYNAMIC_DEP", "FIXED_BUSINESS_GATES",
        "NO_LOW_TO_HIGH_CONTEXT", "NO_BRIER0")

def scientific_config(batch, arm="FULL_V2"):
    if arm not in ARMS or batch not in (64, 32):
        raise ValueError("unqualified arm/batch")
    return replace(Config(), epochs_a=40, epochs_b=12, epochs_c=30,
                   patience=7, validation_days=28, batch_days=batch)

class FloorSelector:
    """No pre-floor checkpoint, no pre-floor accumulated patience."""
    def __init__(self, floor, patience=7):
        self.floor, self.patience = floor, patience
        self.best, self.epoch, self.bad = float("inf"), 0, 0
    def consider(self, epoch, score, guard=True):
        if epoch < self.floor:
            return False
        if score < self.best and guard:
            self.best, self.epoch, self.bad = score, epoch, 0
            return True
        self.bad += 1
        return False
    @property
    def stop(self):
        return self.epoch >= self.floor and self.bad >= self.patience
