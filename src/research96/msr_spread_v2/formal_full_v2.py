"""Thin docs97 hash-guarded adapter. No training loop or recipe duplication."""
import torch
from .config import PACKAGE_ROOT
from .contracts import ContractError,assert_mature
from .audit import file_hash
from .full_v2_profile import NAME,BOUNDS,scientific_config
from . import full_v2_trainer as frozen_trainer

FROZEN_HASHES = {
    'model.py':'6ef936d80a16267056442976e43e2cb625f613c84313dc44ab97fa381e5ef524',
    'losses.py':'636ebe11703582911a2c74ef0262fb5200cf91dfec977213c531394abfd25a46',
    'trainer.py':'5e0eb225e79fd88f1e9f8b9a693d3628a00592af1df5089121d9575396f4e3e3',
    'config.py':'eb6b2f69fd114f5b39307f83e6dcee40a22a2efc111d93c4358c9a05626c5da6',
    'full_v2_profile.py':'f34819a76862d5d6f0ec7a8b630709548bafd6e0c93e28a37b94b404e75569ea',
    'full_v2_trainer.py':'677766940dc8eb993f6953bc4facf5317018982c7a16c2c2e4184a2357cf863e',
}


def verify_frozen_training_hashes(root=PACKAGE_ROOT):
    actual={name:file_hash(root/name) for name in FROZEN_HASHES}
    if actual!=FROZEN_HASHES:
        raise ContractError('FORMAL_FULL_V2_HASH_DRIFT_BLOCKED: '+str(
            [name for name in actual if actual[name]!=FROZEN_HASHES[name]]))
    return actual


def fit_formal_full_v2_day(samples,labels,target,batch=64):
    verify_frozen_training_hashes()
    if batch!=64:
        raise ContractError('formal FULL_V2 requires batch64; no batch32 substitution')
    if not torch.cuda.is_available():
        raise ContractError('formal FULL_V2 requires CUDA')
    assert_mature([s.target_day for s in samples],target)
    model,pre,training=frozen_trainer.fit_day(samples,labels,target,batch=64,arm='FULL_V2')
    if (training.get('profile')!=NAME or training.get('arm')!='FULL_V2' or
        training.get('fresh_fit') is not True or training.get('warm_start') is not False):
        raise ContractError('frozen FULL_V2 training metadata mismatch')
    assert_mature(training['training_days'],target)
    if training['max_label_day']!=max(training['training_days']):
        raise ContractError('training maturity metadata mismatch')
    if len(training['inner_validation_days'])!=28:
        raise ContractError('formal inner validation must be 28 mature days')
    for stage,floor,maximum in BOUNDS:
        if not floor<=training['selected_epochs'][stage]<=maximum:
            raise ContractError('selected epoch outside docs86 bounds')
    if model.cfg.to_dict()!=scientific_config(64,'FULL_V2').to_dict():
        raise ContractError('returned model recipe drift')
    return model,pre,training
