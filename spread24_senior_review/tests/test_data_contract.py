from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from src.data import load_frozen

def test_contract():
 d=load_frozen(Path(__file__).parents[1]); assert len(d)==40670; assert set(d.period.unique())=={"1_8","9_16","17_24"}; assert (d.loc[d.target_spread.notna(),"target_direction"].astype(bool)==(d.loc[d.target_spread.notna(),"target_spread"]>0)).all()



