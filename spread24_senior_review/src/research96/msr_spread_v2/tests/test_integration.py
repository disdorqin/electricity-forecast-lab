"""T17/T18: real graph, full fresh selection/refit, sealed evaluation, seeded rerun."""
import json
from datetime import date,timedelta
import numpy as np
import pytest
from ..walkforward import run_walkforward
from ..dataset import true_label
from ..audit import frozen_check,code_audit,scientific_gate
from ..contracts import ContractError
from ..feature_registry import registry
from ..trainer import Preprocessor

@pytest.fixture(scope="module")
def integrations(tmp_path_factory):
    # Construct the consumed/synthetic fixture without broadening any real truth access.
    from .conftest import store as fixture
    store = fixture.__wrapped__()
    dates = ["2023-02-10","2023-02-15"]
    roots = []
    for repeat in range(2):
        root = tmp_path_factory.mktemp("T17_T18_"+str(repeat))
        def reader(d):
            seals = list(root.glob("*/sealed/"+d+"/PREDICTION_SEAL.json"))
            assert len(seals)==1, "target evaluation must follow prediction persistence"
            seal = json.loads(seals[0].read_text())
            assert seal["target_truth_opened"] is False
            return true_label(store,d)
        run,manifest = run_walkforward(dates,store=store,label_reader=reader,
            output_root=root,command="pytest T17_T18 synthetic CPU integration")
        roots.append((run,manifest))
    return roots

def test_T17_seeded_daily_fresh_refit(integrations):
    (a,ma),(b,mb) = integrations
    assert ma["config_hash"]==mb["config_hash"]
    for d in ma["target_dates"]:
        pa = np.load(a/d/"DISTRIBUTIONS.npz")
        pb = np.load(b/d/"DISTRIBUTIONS.npz")
        for key in ("spread_scenarios","spread_median","p_positive"):
            np.testing.assert_array_equal(pa[key],pb[key]) # declared tolerance = 0 on CPU
        ta = json.loads((a/d/"TRAINING.json").read_text())
        tb = json.loads((b/d/"TRAINING.json").read_text())
        assert ta["selected_epochs"]==tb["selected_epochs"]==dict(A=1,B=1,C=1)
        assert ta["fresh_fit"] and not ta["warm_start"]
        assert len(ta["inner_validation_days"])==28
        cutoff = date.fromisoformat(d)-timedelta(days=2)
        assert date.fromisoformat(ta["max_label_day"])<=cutoff
        assert ta["selection_preprocessor"]["fit_days"]==ta["training_days"][:-28]
        assert {r["phase"] for r in ta["stage_audits"]}=={"selection","refit"}
        assert all(r["freeze_pass"] for r in ta["stage_audits"])
        assert ta["gradient_audit"]["reachability"]=="PASS"

def test_T18_cpu_two_day_artifacts(integrations):
    run,manifest = integrations[0]
    assert len(manifest["days"])==2 and manifest["status"]=="COMPLETE"
    assert manifest["smoke"] and not manifest["fresh_confirm_eligible"]
    assert manifest["eligible_for_scientific_claim"] is False
    assert not manifest["production_write"] and not manifest["hard_state"] and not manifest["markov"]
    for entry in manifest["days"]:
        root = run/entry["target_day"]
        required = ["MODEL.pt","PREDICTIONS.csv","DISTRIBUTIONS.npz","TRAINING.json",
            "PREPROCESSOR.json","FEATURE_REGISTRY.json","SPECTRAL_GATES.json",
            "GRADIENT_AUDIT.json","METRICS.json","SLOT_METRICS.csv","TIMINGS.json","DAY_MANIFEST.json"]
        assert all((root/p).is_file() for p in required)
        p = np.load(root/"DISTRIBUTIONS.npz")
        np.testing.assert_array_equal(p["spread_scenarios"],p["low_scenarios"]+p["high_scenarios"])
        np.testing.assert_array_equal(p["rt_median"],p["target_da"][None]+p["spread_median"])
        np.testing.assert_array_equal(p["rt_quantiles"],p["target_da"][None,:,None]+p["spread_quantiles"])
        assert p["spread_scenarios"].shape==(1,256,96)
        metrics = json.loads((root/"METRICS.json").read_text())
        assert abs(metrics["rt_mae"]-metrics["spread_mae"])<1e-10
        assert all(k in metrics for k in ("positive_recall","negative_recall","balanced_accuracy","all_negative_baseline"))
        assert entry["timing"]["gradient_audit"]>0
    assert (run/"DAILY_METRICS.csv").is_file() and (run/"MONTHLY_DIAGNOSTICS.csv").is_file()

def test_T01_frozen_trees_after_integration(integrations):
    audit = frozen_check()
    assert audit["zero_drift"],audit
    assert code_audit()["status"]=="PASS"

def test_scientific_execution_fail_closed():
    with pytest.raises(ContractError,match="independent-review"):
        scientific_gate(None)
