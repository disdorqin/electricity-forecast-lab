"""Frozen docs/84 architecture; docs/82 dates. No scientific sweep."""
from dataclasses import dataclass, asdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
LAB_ROOT = PACKAGE_ROOT.parents[2]
OUTPUT_ROOT = LAB_ROOT / "outputs/experiments/msr_spread_v2"
SOURCE_PATH = LAB_ROOT / "data/r96_wave2_stage_a_20261004/canonical_v2/r96store_v2_3d083991b4fcd963e2ab/EPF_PMOS_96_FULL.parquet"
SOURCE_SHA256 = "49fade43b0f3505a437e8704377bf5ef07f780429f9052caaca74aa3525e95a2"
Q_KNOTS = (.01, .05, .10, .25, .50, .75, .90, .95, .99)
FAMILIES = ("PRICE", "DEMAND", "RENEWABLE", "SUPPLY_SYSTEM", "MARKET_TIGHTNESS")
BYPASS = "CALENDAR_MASK"
T24_LAMBDA = 215.32246474604284
T24_SHA256 = "7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7"
FAST24 = (
    "2026-01-05","2026-01-15","2026-01-25",
    "2026-02-05","2026-02-15","2026-02-25",
    "2026-03-05","2026-03-15","2026-03-31",
    "2026-04-01","2026-04-15","2026-04-25",
    "2026-05-05","2026-05-15","2026-05-25",
    "2026-06-05","2026-06-15","2026-06-25",
    "2026-07-05","2026-07-15","2026-07-25",
    "2026-08-03","2026-08-08","2026-08-13",
)

@dataclass(frozen=True)
class Config:
    seed: int = 20261009
    family_width: int = 8
    low_width: int = 48
    high_width: int = 64
    tab_width: int = 64
    tab_k: int = 4
    dropout: float = .05
    m_train: int = 32
    m_eval: int = 256
    k_dep: int = 6
    tau0: float = .10
    rank_bandwidth: float = .08
    epochs_a: int = 30
    epochs_b: int = 8
    epochs_c: int = 20
    batch_days: int = 32
    validation_days: int = 28
    patience: int = 5
    lr_a: float = 1e-3
    lr_b: float = 5e-4
    lr_c: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    def to_dict(self):
        return asdict(self)
