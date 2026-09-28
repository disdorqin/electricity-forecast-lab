from src.TafM_改进源码.train import update_checkpoint_state


def _m(raw, magnitude, total=1.0, epoch=1):
    return {"raw_direction_accuracy": raw, "magnitude_mae": magnitude, "L_total": total, "epoch": epoch}


def test_checkpoint_guardrail_high_raw_more_than_two_points_invalidates_old():
    selected = _m(.55, 20.0, epoch=1)
    state = update_checkpoint_state(_m(.58, 100.0, epoch=2), .55, selected, tolerance=.02)
    assert state["raw_anchor"] == .58
    assert state["selected_invalidated"]
    assert state["selected"] == _m(.58, 100.0, epoch=2)


def test_within_guardrail_lower_magnitude_wins_even_at_lower_raw():
    state = update_checkpoint_state(_m(.54, 70.0, epoch=2), .55, _m(.55, 80.0, epoch=1), tolerance=.02)
    assert state["selected"]["epoch"] == 2
    assert not state["raw_anchor_improved"]


def test_anchor_invalidates_old_selected_and_counts_as_progress():
    state = update_checkpoint_state(_m(.58, 90.0, epoch=3), .55, _m(.55, 10.0, epoch=1), tolerance=.02)
    assert state["selected_invalidated"] is True
    assert state["selected"]["epoch"] == 3
    assert state["progress"] is True


def test_magnitude_tie_then_total_loss_breaks_tie():
    state = update_checkpoint_state(_m(.55, 80.0, .9, 2), .55, _m(.55, 80.0, 1.0, 1), tolerance=.02)
    assert state["selected"]["epoch"] == 2


def test_full_tie_keeps_earlier_epoch():
    state = update_checkpoint_state(_m(.55, 80.0, 1.0, 2), .55, _m(.55, 80.0, 1.0, 1), tolerance=.02)
    assert state["selected"]["epoch"] == 1
    assert state["selected_improved"] is False


def test_real_2026_06_15_two_epoch_fixture_selects_epoch_two():
    # Recorded monitor fixture from the 2026-06-15 real Stage-A engineering smoke.
    epoch1 = _m(.5495, 155.06, 2.0, 1)
    state1 = update_checkpoint_state(epoch1, None, None, tolerance=.02)
    epoch2 = _m(.5337, 56.16, 1.0, 2)
    state2 = update_checkpoint_state(epoch2, state1["raw_anchor"], state1["selected"], tolerance=.02)
    assert state2["raw_anchor"] == .5495
    assert state2["selected"]["epoch"] == 2
    assert state2["selected"]["raw_direction_accuracy"] >= state2["raw_anchor"] - .02
