import inspect

from src.TafM_改进源码 import evaluate, train
from src import run_tabm_v21


def test_training_and_evaluation_function_defaults_are_stage_a():
    assert inspect.signature(train.train_target_day).parameters["train_mode"].default == "stage_a"
    assert inspect.signature(evaluate.evaluate_range).parameters["train_mode"].default == "stage_a"
    assert inspect.signature(evaluate.run_ablation).parameters["train_mode"].default == "stage_a"


def test_cli_train_default_and_explicit_stage_ab_route(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run_tabm_v21, "train_target_day", lambda *a, **kw: calls.append(kw["train_mode"]) or
                        {"run_dir": "dummy", "manifest": {}, "metrics": {}})
    assert run_tabm_v21.main(["train", "--target-day", "2026-06-15", "--profile", "smoke"]) == 0
    assert run_tabm_v21.main(["train", "--target-day", "2026-06-15", "--profile", "smoke",
                              "--train-mode", "stage_ab"]) == 0
    capsys.readouterr()
    assert calls == ["stage_a", "stage_ab"]


def test_cli_eval_routes_stage_a_by_default(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run_tabm_v21, "evaluate_range", lambda *a, **kw: calls.append(kw["train_mode"]) or
                        {"output_dir": "dummy", "metrics": {}})
    assert run_tabm_v21.main(["eval", "--start", "2026-06-01", "--end", "2026-06-01"]) == 0
    assert run_tabm_v21.main(["eval", "--start", "2026-06-01", "--end", "2026-06-01",
                              "--train-mode", "stage_ab"]) == 0
    capsys.readouterr()
    assert calls == ["stage_a", "stage_ab"]
