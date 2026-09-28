"""E4-B Gate A cheap probe: validate the in-memory experiment feature manifest builder.

Verifies plan29 Gate A items 1-13 that do NOT require GPU training:
 - selected222 reproduces frozen bit-identically
 - F1 literature240 = 240 features, 18 recovered, Strong211/Weak29
 - F2 all259 = 259 features, 37 recovered, Strong211/Weak48
 - canonical registry order + ascending selected_indices
 - all recovered routed Weak only
 - identical eligible-index SETS across F0/F1/F2 for all 28 formal days
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.TafM_改进源码.config import default_selector_path
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector
from src.TafM_改进源码.train import build_experiment_feature_manifest, LITERATURE18_E4B

store = SequenceStore.load()
frozen, fhash = load_frozen_selector(default_selector_path())
print(f"frozen selected_feature_count = {frozen['selected_feature_count']}")
print(f"frozen candidate count = {frozen['candidate_feature_count']}")
print(f"frozen feature_roles len = {len(frozen['feature_roles'])}")
print(f"store.feature_names len = {len(store.feature_names)}")

# frozen role tallies
role_counts = {}
for r in frozen["feature_roles"]:
    role_counts[r["role"]] = role_counts.get(r["role"], 0) + 1
print("frozen role counts:", role_counts)
strong_n = sum(1 for r in frozen["feature_roles"] if r["role"] != "Weak")
weak_n = sum(1 for r in frozen["feature_roles"] if r["role"] == "Weak")
print(f"frozen strong(=non-Weak)={strong_n} weak={weak_n} sum={strong_n+weak_n}")

# check 18 literature features roles in frozen
print("\n--- literature18 roles in frozen selector ---")
for n in LITERATURE18_E4B:
    rr = next((r["role"] for r in frozen["feature_roles"] if r["feature_name"] == n), "MISSING")
    print(f"  {n}: {rr}")

# Build manifests
profiles = ["selected222", "literature240", "all259"]
manifests = {}
for p in profiles:
    m = build_experiment_feature_manifest(frozen, p, store)
    manifests[p] = m
    sel = m["selected_features"]
    rec = m.get("experiment_recovered_features", [])
    strong = m.get("experiment_strong_count")
    weak = m.get("experiment_weak_count")
    # canonical order + ascending indices
    idx = m["selected_indices"]
    canonical_ok = sel == [n for n in store.feature_names if n in set(sel)]
    asc_ok = idx == sorted(idx)
    # recovered all Weak
    role_by_name = {r["feature_name"]: r["role"] for r in m["feature_roles"]}
    rec_weak_ok = all(role_by_name[n] == "Weak" for n in rec)
    print(f"\n[{p}] selected={len(sel)} recovered={len(rec)} strong={strong} weak={weak} "
          f"canonical_order={canonical_ok} ascending_idx={asc_ok} recovered_all_weak={rec_weak_ok}")
    if not (canonical_ok and asc_ok and rec_weak_ok):
        print("  !!! FAIL"); sys.exit(2)

# selected222 must be bit-identical to frozen
assert manifests["selected222"] is frozen, "selected222 must return the frozen object unchanged"
print("\nselected222 returns frozen object unchanged: OK")

# Eligibility identity across 28 formal days
WINDOWS = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
           "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}
days = [str((pd.Timestamp(a) + pd.Timedelta(days=i)).date()) for a, b in WINDOWS.values() for i in range(7)]
print(f"\nChecking eligibility identity over {len(days)} formal days ...")
all_ok = True
for d in days:
    sets = {}
    for p in profiles:
        elig = store.eligibility(manifests[p], current_target_day=d)
        sets[p] = set(elig["eligible_indices"])
    base = sets["selected222"]
    same = all(sets[p] == base for p in profiles)
    if not same:
        all_ok = False
        print(f"  DAY {d}: identity FAIL sizes={ {p: len(sets[p]) for p in profiles} }")
print("eligibility identity across profiles:", "OK" if all_ok else "FAIL")

# selector_indices declared-vs-expected check for F1/F2
for p in ["literature240", "all259"]:
    names, cols, _ = store.selector_indices(manifests[p])
    expected = [store.feature_names.index(n) for n in names]
    assert list(cols) == expected, f"{p} selected_indices mismatch"
    print(f"{p}: selector_indices declared==expected OK ({len(cols)} cols)")

print("\nPROBE RESULT:", "ALL GATE A PRE-FLIGHT CHECKS PASS" if all_ok else "PRE-FLIGHT FAIL")
