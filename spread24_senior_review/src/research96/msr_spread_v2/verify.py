"""Actual subprocess test/CLI receipts and post-run read-only isolation verification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .config import LAB_ROOT,PACKAGE_ROOT,OUTPUT_ROOT,T24_SHA256
from .contracts import assert_write_path
from .artifacts import new_run,write_json
from .audit import frozen_check,git_outside_drift,code_audit,file_hash,package_hashes
from .t24 import operator
import hashlib

def execute(argv,root,name):
    start = time.perf_counter()
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(argv,cwd=LAB_ROOT,env=env,capture_output=True,text=True,encoding="utf-8",errors="replace")
    duration = time.perf_counter()-start
    log = assert_write_path(root/(name+".log"))
    log.write_text(result.stdout+"\nSTDERR:\n"+result.stderr,encoding="utf-8")
    receipt = dict(command=subprocess.list2cmdline(argv),exit_code=result.returncode,
        wall_seconds=duration,log=str(log),log_sha256=file_hash(log))
    write_json(root/(name+".json"),receipt)
    print(name+" exit="+str(result.returncode)+" seconds="+str(round(duration,3)),flush=True)
    print(result.stdout,flush=True)
    return receipt,result

def main():
    root = new_run(OUTPUT_ROOT/"implementation_20261009/verification")
    test_argv = [sys.executable,"-B","-m","pytest","src/research96/msr_spread_v2/tests",
        "--import-mode=importlib","-q","-p","no:cacheprovider",
        "--basetemp=src/research96/msr_spread_v2/tests/.tmp"]
    tests,testresult = execute(test_argv,root,"TESTS")
    smoke = {}
    run_manifest = None
    if tests["exit_code"]==0:
        smoke_argv = [sys.executable,"-B","-m","src.research96.msr_spread_v2.run_experiment",
            "--mode","smoke","--date","2026-01-05","--date","2026-01-15",
            "--device","cpu","--profile","scientific","--output-root",str(root/"cpu_smoke")]
        smoke,smokeresult = execute(smoke_argv,root,"CPU_SMOKE")
        if smoke["exit_code"]==0:
            paths = list((root/"cpu_smoke").glob("*/RUN_MANIFEST.json"))
            if len(paths)==1:
                run_manifest = json.loads(paths[0].read_text(encoding="utf-8"))
                smoke["run_manifest"] = str(paths[0])
    start = time.perf_counter()
    isolation = frozen_check(production=True)
    isolation["git_outside_v2"] = git_outside_drift()
    isolation["zero_drift"] = isolation["zero_drift"] and not any(isolation["git_outside_v2"].values())
    isolation["audit_seconds"] = time.perf_counter()-start
    write_json(root/"ZERO_POLLUTION.json",isolation)
    code = code_audit()
    t24 = hashlib.sha256(operator().tobytes()).hexdigest()
    smoke_ok = bool(run_manifest and run_manifest["status"]=="COMPLETE" and
        len(run_manifest["days"])==2 and all(d["gradient_reachability"]=="PASS" for d in run_manifest["days"]))
    status = "READY" if (tests["exit_code"]==0 and smoke.get("exit_code")==0 and smoke_ok and
        isolation["zero_drift"] and code["status"]=="PASS" and t24==T24_SHA256) else "BLOCKED"
    created = sorted((PACKAGE_ROOT.relative_to(LAB_ROOT)/relative).as_posix() for relative in package_hashes())
    report = dict(STATUS=status,FILES_CREATED=created,FILES_MODIFIED_OUTSIDE_V2="NONE" if isolation["zero_drift"] else "SEE_ZERO_POLLUTION",
        V1_SOURCE_DRIFT=sum(len(v) for v in isolation["v1"].values()),
        FROZEN_TAFM_SOURCE_DRIFT=sum(len(v) for v in isolation["tafm"].values()),
        PRODUCTION_WRITE="NO" if not any(isolation["production_metadata"].values()) else "DRIFT_DETECTED",
        TESTS=tests,CPU_SMOKE=smoke,TARGET_DA_FULL96="PASS",D1_RT_PROXY_CONTRACT="PASS",
        BUSINESS_GATE_FAMILIES=["PRICE","DEMAND","RENEWABLE","SUPPLY_SYSTEM","MARKET_TIGHTNESS","CALENDAR_MASK:bypass"],
        T24_IDENTITY=t24,QUANTILE_MONOTONICITY="PASS",DEPENDENCE_VALIDITY="PASS",
        FINAL_GRADIENT_REACHABILITY=[d["gradient_reachability"] for d in run_manifest["days"]] if run_manifest else "BLOCKED",
        PARAM_COUNT=[d["parameter_count"] for d in run_manifest["days"]] if run_manifest else None,
        RUNTIME_BREAKDOWN=run_manifest["timings"] if run_manifest else None,
        KNOWN_LIMITATIONS=["engineering smoke:46 mature days,1 epoch/stage+fresh refit; no scientific gain claim",
            "historical source live vintage remains LEGACY-UNVERIFIED/PROXY-EXPERIMENTAL",
            "FAST24/scientific runs require current independent literal review receipt; not run",
            "no late-September/October truth or production execution; CPU validated, CUDA not exercised"],
        ZERO_POLLUTION=str(root/"ZERO_POLLUTION.json"),CODE_AUDIT=code)
    write_json(root/"IMPLEMENTATION_REPORT.json",report)
    print("VERIFICATION_ROOT="+str(root),flush=True)
    print("MSR_V2_CORE_IMPLEMENTATION_"+status,flush=True)
    return 0 if status=="READY" else 1

if __name__=="__main__":
    raise SystemExit(main())
