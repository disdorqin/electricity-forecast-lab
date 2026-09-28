import numpy as np
from src.TafM_改进源码.target_adapter import source_to_model_target, model_direction


def test_da_rt_to_rt_da_and_exact_zero_is_nonpositive():
    y = np.array([-5.0, 0.0, 5.0])
    model_y = source_to_model_target(y)
    assert model_y.tolist() == [5.0, 0.0, -5.0]
    assert model_direction(model_y).tolist() == [True, False, False]
