from pathlib import Path
import argparse
from src.pipeline import run

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--rebuild-from-raw",action="store_true"); ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args(); root=Path(__file__).resolve().parent
    if a.rebuild_from_raw: print("raw_reference is supplied for inspection; fixed reproduction uses frozen_repro")
    run(root, smoke=a.smoke)
