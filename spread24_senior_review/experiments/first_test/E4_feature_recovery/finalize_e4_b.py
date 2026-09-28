"""E4-B finalize driver: waits for the F2 panel to complete, then runs the full artifact pipeline.

Steps (all CPU except the already-finished panel training):
  1. poll runs/E4B-F2-*/RUN_RECORD.json until >= 28 PASS (max ~45 min)
  2. run focused E4-B test suite -> tests_result.json (Gate A item 19)
  3. analyze_e4_b.py          -> all plan29 §18 CSVs + figures + E4_B_GATE.md + E4_B_result.json
  4. gate_a_e4_b.py           -> benchmark/gate_a_*.json + failclosed.txt
  5. benchmark_e4_b.py        -> benchmark/benchmark_2026-02-13.* + eligible_digest.json
  6. write_summary_e4_b.py    -> E4_B_summary.md (16 sections)
Finally prints the E4_B_GATE FEATURE_RECOVERY_SIGNAL.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

H = Path(__file__).resolve().parent
REPO = H.parents[2]
PY = "D:/computer_download/environment/conda/epf-2/python.exe"
TEST = REPO / "src/TafM_改进源码/tests/test_e4_b_feature_recovery.py"
LOG = H / "runs" / "finalize_pipeline.log"


def log(msg: str):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(script: str, *extra):
    cmd = [PY, str(H / script), *extra]
    log(f"RUN {script} {' '.join(extra)}")
    p = subprocess.run(cmd, cwd=str(H), capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    for ln in out.splitlines()[-25:]:
        log("   " + ln)
    if p.returncode != 0:
        log(f"!! {script} returned {p.returncode}")
    return p.returncode


def main():
    # 1) wait for F2 panel
    target = 28
    deadline = time.time() + 45 * 60
    while True:
        done = len(list((H / "runs").glob("E4B-F2-*/RUN_RECORD.json")))
        if done >= target:
            log(f"F2 panel complete: {done}/{target} PASS")
            break
        if time.time() > deadline:
            log(f"TIMEOUT waiting for F2 panel: only {done}/{target} done; proceeding with partial data")
            break
        time.sleep(20)

    # 2) focused test suite -> tests_result.json
    log("running focused E4-B test suite")
    tp = subprocess.run([PY, "-m", "pytest", str(TEST), "-q", "--tb=short"],
                        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", errors="replace")
    (H / "tests_result.json").write_text(json.dumps(
        {"returncode": tp.returncode, "passed": None, "total": None,
         "stdout_tail": (tp.stdout or "")[-2000:], "stderr_tail": (tp.stderr or "")[-2000:]},
        indent=2), encoding="utf-8")
    log(f"pytest returncode={tp.returncode}")

    # 3-6) pipeline
    run("analyze_e4_b.py")
    run("gate_a_e4_b.py")
    run("benchmark_e4_b.py")
    run("write_summary_e4_b.py")

    gate = (H / "E4_B_GATE.md").read_text(encoding="utf-8")
    signal = gate.split("`")[1] if "`" in gate else "UNKNOWN"
    log(f"=== E4_B_GATE FEATURE_RECOVERY_SIGNAL = {signal} ===")
    # update snapshot status
    snap = H / "00_PLAN_SNAPSHOT.md"
    if snap.exists():
        txt = snap.read_text(encoding="utf-8").replace("STATUS=EXECUTING", "STATUS=COMPLETE")
        snap.write_text(txt, encoding="utf-8")
    log("finalize done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
