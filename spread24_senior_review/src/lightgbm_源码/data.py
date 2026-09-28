from pathlib import Path
import pandas as pd

REQUIRED = {"target_day", "period", "target_spread", "target_direction"}

def load_frozen(root):
    p = Path(root) / "data" / "frozen_repro" / "slot_table.parquet"
    df = pd.read_parquet(p)
    df["target_day"] = pd.to_datetime(df["target_day"]).dt.date
    missing = REQUIRED - set(df.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    return df

def load_raw(root):
    p = Path(root) / "data" / "raw_reference" / "shandong_pmos_hourly.csv"
    return pd.read_csv(p)
