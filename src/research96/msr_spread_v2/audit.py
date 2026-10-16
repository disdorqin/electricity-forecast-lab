"""Read-only hash isolation audit and fail-closed write/feature gates."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
from .config import LAB_ROOT,PACKAGE_ROOT,OUTPUT_ROOT
from .contracts import assert_write_path,ContractError

BASELINE_PATH = OUTPUT_ROOT/"implementation_20261009/BASELINE.json"

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _walk(root,skip_transient=False):
    root = Path(root).resolve()
    text = str(root)
    if os.name=="nt":
        text = "\\\\?\\"+text
    for directory,subdirs,files in os.walk(text):
        subdirs[:] = [s for s in subdirs if s!=".git" and not (
            skip_transient and (
                s.startswith((".tmp", ".pytest")) or
                s in ("__pycache__",)
            )
        )]
        for name in files:
            p = Path(directory)/name
            relative = os.path.relpath(p,text).replace("\\","/")
            yield relative,p

def package_hashes(extensions=(".py",".md",".npy")):
    return {relative:file_hash(p) for relative,p in _walk(PACKAGE_ROOT,skip_transient=True)
            if p.suffix in extensions}


def tree_hash(root):
    return {relative:file_hash(p) for relative,p in _walk(root)}

def difference(before,after):
    return dict(changed=sorted(k for k in before.keys()&after.keys() if before[k]!=after[k]),
                added=sorted(after.keys()-before.keys()),missing=sorted(before.keys()-after.keys()))

def frozen_check(baseline_path=BASELINE_PATH,production=False):
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    report = {}
    for key,root in (("v1",LAB_ROOT/"src/research96/msr_spread_v1"),
                     ("tafm",LAB_ROOT/"src/TafM_改进源码")):
        report[key] = difference(baseline[key],tree_hash(root))
    if production:
        # Code SHA + metadata only. Never materialize market/model/lockbox file contents.
        root = LAB_ROOT.parent.parent/"electricity_forecast_model2.5"
        files = list(_walk(root))
        sha = {relative:file_hash(p) for relative,p in files if p.suffix==".py"}
        metadata = {relative:[p.stat().st_size,p.stat().st_mtime_ns] for relative,p in files}
        report["production_source"] = difference(baseline["production_source"],sha)
        report["production_metadata"] = difference(baseline["production_metadata"],metadata)
    report["zero_drift"] = all(not any(d.values()) for d in report.values())
    return report

def code_audit():
    violations = []
    for p in PACKAGE_ROOT.glob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                mod = node.module or ""
                if "msr_spread_v1" in mod or "TafM" in mod or "electricity_forecast_model2" in mod:
                    violations.append(str(p)+": frozen runtime import")
            if isinstance(node,ast.Import):
                for alias in node.names:
                    if any(t in alias.name for t in ("pymysql","sqlalchemy","psycopg","msr_spread_v1","TafM")):
                        violations.append(str(p)+": forbidden dependency")
    return dict(status="PASS" if not violations else "FAIL",violations=violations,
                output_guard="contracts.assert_write_path before each artifact write",
                database_dependencies=False,production_runtime_dependency=False)

def scientific_gate(receipt_path):
    if not receipt_path:
        raise ContractError("scientific execution requires current independent-review receipt")
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt.get("status")!="PASS" or receipt.get("independent_review")!="PASS":
        raise ContractError("independent literal architecture review not passed")
    current = package_hashes((".py",))
    if receipt.get("code_hashes")!=current:
        raise ContractError("scientific review receipt does not bind current V2 code")
    return receipt

def git_outside_drift(baseline_path=BASELINE_PATH):
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    status = subprocess.check_output(["git","status","--porcelain","-uall"],cwd=LAB_ROOT,text=True)
    def outside(text):
        return sorted(line for line in text.splitlines() if "msr_spread_v2" not in line)
    return difference(dict.fromkeys(outside(baseline["git_status"]),1),dict.fromkeys(outside(status),1))
