from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from datetime import date
from src.leakage import training_last_day,audit

def test_cutoff(): assert training_last_day(date(2026,1,10))==date(2026,1,8)
def test_features(): assert audit([date(2026,1,10)],14,["spread_lag2d","ctx_spread_mean14"])[0]["training_boundary_ok"]
