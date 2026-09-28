from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from src.pipeline import run

def test_smoke():
 out=run(Path(__file__).parents[1], smoke=True); assert len(out)==48; assert out.target_day.nunique()==2
