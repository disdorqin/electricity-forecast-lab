"""Stable CPU preparation API and explicit production/shadow entry contract."""
from __future__ import annotations

from pathlib import Path

from ..dataset import build_sample
from .bundle_identity import verify_bundle
from .history_store import load_mature_history
from .input_bundle import build_input_bundle
from .qualification import QualificationError


class ProductionAPI:
    def __init__(self, bundle_root: Path):
        self.bundle_root = Path(bundle_root).resolve()
        self.bundle_identity = verify_bundle(self.bundle_root)

    def prepare_target(self, *, target_day: str, manifest_path: Path, values_path: Path,
                       history_path: Path, mode: str = "promotion", i1_ledger_path: Path | None = None):
        history, history_receipt = load_mature_history(history_path, target_day=target_day)
        bundle = build_input_bundle(manifest_path, values_path, history, mode=mode,
                                    i1_ledger_path=i1_ledger_path,
                                    bundle_manifest_sha256=self.bundle_identity["aggregate_sha256"])
        if bundle.target_day != target_day:
            raise QualificationError("requested target differs from exact-origin capture")
        sample = build_sample(bundle.store, target_day)
        if sample.provenance["label_cutoff"] != bundle.receipt["max_label_day"]:
            raise QualificationError("sample label cutoff differs from bundle cutoff")
        return {
            "bundle": bundle, "sample": sample, "history_receipt": history_receipt,
            "bundle_identity": self.bundle_identity, "production_eligible": bundle.receipt["production_eligible"],
        }

