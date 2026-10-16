"""One-target fresh-fit production/shadow orchestration over the frozen core."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np

from ..contracts import ContractError, assert_mature, label_cutoff
from ..dataset import build_sample, training_panel
from ..direction_decision import DirectionDecisionConfig, direction_decision
from ..feature_registry import registry
from ..formal_full_v2 import fit_formal_full_v2_day
from ..frontier_f import FrontierFConfig, decide_frontier
from ..frontier_f_history import context_candidates, scored_history
from .api import ProductionAPI
from .bundle_identity import sha256_file
from .qualification import QualificationError


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def _write_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False, default=_jsonable), encoding="utf-8")
    os.replace(tmp, path)


def _write_npz(path: Path, arrays):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _file_map(root: Path):
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _to_numpy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: _to_numpy(v) for k, v in value.items()}
    return value


def _access_guard(store, target, start):
    d = __import__("datetime").date.fromisoformat(target)
    cutoff = d - timedelta(days=2)
    for accessed_day, field, prefix in store.access_log[start:]:
        day = __import__("datetime").date.fromisoformat(str(accessed_day)[:10])
        if field == "rt" and not (day <= cutoff or (day == d - timedelta(days=1) and prefix == 56)):
            raise ContractError("illegal RT information access: " + str((accessed_day, prefix)))


def _require_complete_training_panel(samples, rejected, target_day):
    training_days = [s.target_day for s in samples]
    cutoff = label_cutoff(target_day)
    start_day = __import__("datetime").date(2023, 1, 1)
    expected_days = [(start_day + timedelta(days=i)).isoformat()
                     for i in range((cutoff - start_day).days + 1)]
    assert_mature(training_days, target_day)
    if rejected or training_days != expected_days:
        raise ContractError("production training history must equal every day from 2023-01-01 through D-2")
    return training_days


def _verify_existing_target(final: Path, api: ProductionAPI, bundle, target_day: str):
    manifest_path = final / "ARTIFACT_MANIFEST.json"
    if not manifest_path.is_file():
        raise ContractError("existing target output is not a complete sealed artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("truth_scoring_performed") is not False or not isinstance(manifest.get("files"), dict):
        raise ContractError("existing target artifact manifest is invalid")
    expected_files = manifest["files"]
    actual_files = {p.relative_to(final).as_posix() for p in final.rglob("*") if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"}
    if actual_files != set(expected_files):
        raise ContractError("existing target artifact inventory drift")
    for rel, expected in expected_files.items():
        if sha256_file(final / rel) != expected:
            raise ContractError("existing target artifact hash drift: " + rel)
    seals = {}
    for stage, stem in (("FULL", "FULL"), ("DIRECTION", "DIRECTION"), ("F", "F")):
        path = final / stage / f"{stem}_SEAL.json"
        seal = json.loads(path.read_text(encoding="utf-8"))
        if seal.get("target_day") != target_day or seal.get("target_truth_opened") is not False:
            raise ContractError("existing target stage seal mismatch: " + stage)
        if seal.get("arrays_sha256") != sha256_file(final / stage / f"{stem}.npz"):
            raise ContractError("existing target stage array hash mismatch: " + stage)
        identity = seal.get("identity", {})
        if identity.get("bundle_identity") != bundle.receipt or identity.get("bundle_manifest_sha256") != api.bundle_identity["manifest_sha256"]:
            raise ContractError("existing target input/bundle identity drift: " + stage)
        seals[stage] = (seal, sha256_file(path))
    if seals["DIRECTION"][0].get("identity", {}).get("stage_inputs", {}).get("full_seal_sha256") != seals["FULL"][1]:
        raise ContractError("existing DIRECTION upstream FULL seal mismatch")
    f_inputs = seals["F"][0].get("identity", {}).get("stage_inputs", {})
    if f_inputs.get("full_seal_sha256") != seals["FULL"][1] or f_inputs.get("direction_seal_sha256") != seals["DIRECTION"][1]:
        raise ContractError("existing F upstream seal mismatch")
    return {"target_day": target_day, "status": "REUSED", "path": str(final),
            "full": seals["FULL"][0], "direction": seals["DIRECTION"][0], "f": seals["F"][0],
            "target_truth_opened": False, "new_full_fit": False, "reused": True}


class FStateReader:
    """Read only existing, hash-sealed historical F artifacts; never trains missing days."""
    def __init__(self, root: Path | None):
        self.root = None if root is None else Path(root).resolve()

    def read(self, target: str):
        if self.root is None:
            return None
        folder = self.root / target / "F"
        npz, seal_path = folder / "F.npz", folder / "F_SEAL.json"
        if not npz.exists() and not seal_path.exists():
            return None
        if not npz.is_file() or not seal_path.is_file():
            raise ContractError("partial historical F artifact")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if (seal.get("target_day") != target or seal.get("target_truth_opened") is not False or
                seal.get("arrays_sha256") != sha256_file(npz)):
            raise ContractError("historical F artifact seal/hash mismatch")
        inputs = seal.get("identity", {}).get("stage_inputs", {})
        for stage, stem, expected_key in (("FULL", "FULL", "full_seal_sha256"),
                                           ("DIRECTION", "DIRECTION", "direction_seal_sha256")):
            base = self.root / target / stage
            base_npz, base_seal_path = base / f"{stem}.npz", base / f"{stem}_SEAL.json"
            if not base_npz.is_file() or not base_seal_path.is_file():
                raise ContractError(f"historical F state missing paired {stage} seal")
            base_seal = json.loads(base_seal_path.read_text(encoding="utf-8"))
            if (base_seal.get("target_day") != target or base_seal.get("target_truth_opened") is not False or
                    base_seal.get("arrays_sha256") != sha256_file(base_npz) or
                    inputs.get(expected_key) != sha256_file(base_seal_path)):
                raise ContractError(f"historical F {stage} provenance/seal mismatch")
        with np.load(npz, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        required = {"candidate_pool", "features", "positive_candidate", "negative_candidate", "post_positive"}
        if not required.issubset(arrays):
            raise ContractError("historical F artifact missing required candidate state")
        return arrays, {"arrays_sha256": seal["arrays_sha256"], "seal_sha256": sha256_file(seal_path)}


def run_target(api: ProductionAPI, *, target_day: str, manifest_path: Path, values_path: Path,
               history_path: Path, output_root: Path, mode: str = "promotion",
               i1_ledger_path: Path | None = None, f_history_root: Path | None = None,
               f_history_start: str = "2026-01-01"):
    """Fresh FULL fit followed by frozen DDV1/F; never opens target truth."""
    prepared = api.prepare_target(target_day=target_day, manifest_path=manifest_path,
                                  values_path=values_path, history_path=history_path,
                                  mode=mode, i1_ledger_path=i1_ledger_path)
    if mode == "synthetic_test":
        raise QualificationError("synthetic_test mode is prepare-only; model fitting is disabled")
    if mode == "promotion" and not prepared["production_eligible"]:
        raise QualificationError("promotion input is not I1/bundle-qualified")
    bundle, sample = prepared["bundle"], prepared["sample"]
    store = bundle.store
    output_root = Path(output_root).resolve()
    if any(part.casefold() == "electricity_forecast_model2.5" for part in output_root.parts):
        raise ContractError("production2.5 output paths are forbidden")
    final = output_root / target_day
    if final.exists():
        return _verify_existing_target(final, api, bundle, target_day)
    output_root.mkdir(parents=True, exist_ok=True)
    attempt = output_root / (".tmp_" + target_day + "_" + uuid.uuid4().hex)
    attempt.mkdir()
    try:
        start_access = len(store.access_log)
        samples, labels, rejected = training_panel(store, target_day)
        _require_complete_training_panel(samples, rejected, target_day)
        model, pre, training = fit_formal_full_v2_day(samples, labels, target_day, batch=64)
        _access_guard(store, target_day, start_access)
        import torch
        model.eval()
        x = torch.as_tensor(pre.transform(sample.x[None]), device=model.component_center.device)
        da = torch.as_tensor(sample.da[None], dtype=torch.float64, device=model.component_center.device)
        with torch.no_grad():
            out = _to_numpy(model(x, da, scenarios=256))
        if not np.array_equal(out["spread_scenarios"], out["low_scenarios"] + out["high_scenarios"]):
            raise ContractError("frozen FULL scenario identity mismatch")
        full_keys = ("low_q_raw", "high_q_raw", "low_scenarios", "high_scenarios", "spread_scenarios",
                     "spread_quantiles", "spread_median", "rt_quantiles", "rt_median", "p_positive",
                     "fusion_low", "fusion_high", "clamp_fraction")
        full_arrays = {k: out[k] for k in full_keys}
        full_arrays["target_da"] = sample.da
        full_arrays.update(out["dependence"])
        if any(not np.isfinite(v).all() for v in full_arrays.values()):
            raise ContractError("nonfinite FULL numeric output")
        full_dir = attempt / "FULL"
        full_dir.mkdir()
        _write_npz(full_dir / "FULL.npz", full_arrays)
        model_path = full_dir / "MODEL.pt"
        torch.save(model.state_dict(), model_path)
        _write_json(full_dir / "TRAINING.json", training)
        _write_json(full_dir / "PREPROCESSOR.json", pre.manifest())
        _write_json(full_dir / "CONFIG.json", model.cfg.to_dict())
        _write_json(full_dir / "DATA_AUDIT.json", {"rejected_training_days": rejected, "sample_provenance": sample.provenance})
        full_identity = _stage_identity(api, bundle, target_day, "FULL", training)
        full_seal = _seal(full_dir, "FULL", target_day, full_identity)

        dd_start = len(store.access_log)
        dd = direction_decision(np.asarray(full_arrays["p_positive"]).reshape(96),
                                np.asarray(full_arrays["spread_median"]).reshape(96), target_day, store)
        _access_guard(store, target_day, dd_start)
        dd_dir = attempt / "DIRECTION"
        dd_dir.mkdir()
        dd_arrays = dd.arrays()
        _write_npz(dd_dir / "DIRECTION.npz", dd_arrays)
        dd_identity = _stage_identity(api, bundle, target_day, "DIRECTION", {"full_seal_sha256": full_seal["seal_sha256"]})
        _write_json(dd_dir / "TRACE.json", dd.trace)
        dd_seal = _seal(dd_dir, "DIRECTION", target_day, dd_identity)

        f_start = len(store.access_log)
        candidates = context_candidates(store, target_day, full_arrays, dd_arrays)
        reader = FStateReader(f_history_root)
        f_history, dense = scored_history(store, target_day, f_history_start, reader.read)
        result = decide_frontier(candidates, f_history, f_history_start, dense)
        _access_guard(store, target_day, f_start)
        f_dir = attempt / "F"
        f_dir.mkdir()
        f_arrays = result.arrays()
        _write_npz(f_dir / "F.npz", f_arrays)
        _write_json(f_dir / "TRACE.json", result.trace)
        f_identity = _stage_identity(api, bundle, target_day, "F", {
            "full_seal_sha256": full_seal["seal_sha256"],
            "direction_seal_sha256": dd_seal["seal_sha256"],
            "history_artifact_hashes": result.trace["history_artifact_hashes"],
            "f_history_start": f_history_start,
        })
        f_seal = _seal(f_dir, "F", target_day, f_identity)
        if any(s.get("target_truth_opened") is not False for s in (full_seal, dd_seal, f_seal)):
            raise ContractError("truth-before-seal boundary violation")
        _write_json(attempt / "ARTIFACT_MANIFEST.json", {"files": _file_map(attempt), "truth_scoring_performed": False})
        if final.exists():
            raise ContractError("target output appeared during execution")
        os.replace(attempt, final)
        return {"target_day": target_day, "status": "SEALED", "path": str(final),
                "full": full_seal, "direction": dd_seal, "f": f_seal,
                "target_truth_opened": False, "new_full_fit": True}
    except Exception:
        failed = output_root / ("FAILED_" + target_day + "_" + attempt.name[5:])
        if attempt.exists() and not failed.exists():
            os.replace(attempt, failed)
        raise


def _stage_identity(api, bundle, target, stage, stage_inputs):
    from ..direction_decision import DirectionDecisionConfig
    from ..frontier_f import FrontierFConfig
    source_root = Path(__file__).resolve().parent.parent
    def policy_sha(names):
        rows = [f"{name}\0{sha256_file(source_root / name)}\n" for name in sorted(names)]
        return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    return {
        "stage": stage, "target_day": target, "label_cutoff": bundle.receipt["max_label_day"],
        "origin": "D-1 14:00 Asia/Shanghai", "bundle_identity": bundle.receipt,
        "bundle_manifest_sha256": api.bundle_identity["manifest_sha256"],
        "full_policy_hash": policy_sha(("formal_full_v2.py", "full_v2_profile.py", "full_v2_trainer.py",
                                         "trainer.py", "model.py", "losses.py", "config.py")),
        "ddv1_policy_hash": policy_sha(("direction_decision.py", "direction_history.py")) + ":" +
            DirectionDecisionConfig().config_hash,
        "f_policy_hash": policy_sha(("frontier_f.py", "frontier_f_history.py")) + ":" +
            FrontierFConfig().config_hash, "stage_inputs": stage_inputs,
        "warm_start": False, "rolling_window": False, "target_truth_opened": False,
    }


def _seal(folder: Path, stage: str, target: str, identity: dict):
    stem = stage if stage != "DIRECTION" else "DIRECTION"
    array_path = folder / (stem + ".npz")
    seal = {
        "schema": "spread96-production-stage-seal-v1", "stage": stage,
        "target_day": target, "target_truth_opened": False,
        "arrays_sha256": sha256_file(array_path), "identity": identity,
        "sealed_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    path = folder / (stem + "_SEAL.json")
    _write_json(path, seal)
    seal["seal_sha256"] = sha256_file(path)
    return seal

