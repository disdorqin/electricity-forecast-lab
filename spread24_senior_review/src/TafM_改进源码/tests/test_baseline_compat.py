from src import data, evaluate, features, leakage, model, pipeline
from src.lightgbm_源码 import data as baseline_data
from src.lightgbm_源码 import evaluate as baseline_evaluate
from src.lightgbm_源码 import features as baseline_features
from src.lightgbm_源码 import leakage as baseline_leakage
from src.lightgbm_源码 import model as baseline_model
from src.lightgbm_源码 import pipeline as baseline_pipeline


def test_thin_shims_reexport_unchanged_baseline():
    assert data.load_frozen is baseline_data.load_frozen
    assert data.load_raw is baseline_data.load_raw
    assert features.feature_columns is baseline_features.feature_columns
    assert features.matrix is baseline_features.matrix
    assert leakage.training_last_day is baseline_leakage.training_last_day
    assert leakage.audit is baseline_leakage.audit
    assert model.fit_predict is baseline_model.fit_predict
    assert evaluate.metrics is baseline_evaluate.metrics
    assert evaluate.summarize is baseline_evaluate.summarize
    assert pipeline.run is baseline_pipeline.run

