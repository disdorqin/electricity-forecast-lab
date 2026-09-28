import warnings

from src import run_tabm_v2
from src.TafM_改进源码.gate import run_v21_gate


def test_legacy_v2_cli_is_retired_and_fail_closed(capsys):
    assert run_tabm_v2.main(["train", "--target-day", "2026-06-15"]) == 2
    captured = capsys.readouterr()
    assert "V2 CLI is retired" in captured.err
    assert "run_tabm_v21.py" in captured.err
    assert not hasattr(run_tabm_v2, "_gate")
    assert not hasattr(run_tabm_v2, "train_target_day")
    assert not hasattr(run_tabm_v2, "evaluate_range")


def test_v21_gate_is_independent_and_has_no_synthetic_single_bin_warning(tmp_path):
    path = tmp_path / "gate.json"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = run_v21_gate(path)
    assert report["status"] == "PASS"
    assert report["source_gate"] == "PASS"
    assert report["sequence_gate"] == "PASS"
    assert report["gate_ple_bins"]
    assert all(len(edges) >= 3 for edges in report["gate_ple_bins"])
    messages = [str(item.message).lower() for item in caught]
    assert not any("single bin" in message or "two bin edges" in message for message in messages)
    assert path.exists()
