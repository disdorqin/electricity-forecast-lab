from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.TafM_改进源码.config import load_v21_config
from src.TafM_改进源码.dataset import SequenceStore
from src.TafM_改进源码.train import train_target_day

Q2=ROOT/'experiments/first_test/E2_architecture/horizon_specialized_head/runs'
OUT=ROOT/'experiments/first_test/E4_signal/class_weight'
OUT.mkdir(parents=True,exist_ok=True)
WINDOWS={
'W1':('2026-02-12','2026-02-18'),
'W2':('2026-04-12','2026-04-18'),
'W3':('2026-06-12','2026-06-18'),
'W4':('2026-08-07','2026-08-13')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--window',required=True,choices=WINDOWS);a=ap.parse_args()
    lo,hi=WINDOWS[a.window]
    days=[]
    for p in sorted(Q2.glob('E2E1-Q2-*/RUN_RECORD.json')):
        r=json.loads(p.read_text(encoding='utf-8')); d=r['target_day']
        if lo<=d<=hi: days.append(d)
    if len(days)!=7: raise RuntimeError((a.window,days))
    store=SequenceStore.load();cfg=load_v21_config()
    records=[]
    for mode in ['sqrt_balanced','full_balanced']:
        for d in days:
            rec_path=OUT/f'{a.window}_{mode}_{d}.json'
            if rec_path.exists():
                rec=json.loads(rec_path.read_text(encoding='utf-8'));records.append(rec);print('SKIP',mode,d);continue
            t=time.perf_counter()
            r=train_target_day(d,mode='A2',profile='default',train_mode='stage_a',config=cfg,store=store,
                objective_mode='dir_only',checkpoint_policy='direction_first',gradient_policy='vanilla',
                architecture_mode='full_current',direction_fusion_alpha=.8,direction_tabular_mode='current',
                direction_horizon_gate_mode='current',strong_role_profile='all',numeric_encoding_mode='canonical',
                direction_readout_mode='segment_heads',direction_postprocess_mode='none',
                direction_class_weight_mode=mode,experiment_only=True)
            rec={'window':a.window,'arm':'L1' if mode=='sqrt_balanced' else 'L2','mode':mode,
                 'target_day':d,'run_dir':r['run_dir'],'metrics':r['metrics'],
                 'wall_seconds':time.perf_counter()-t}
            rec_path.write_text(json.dumps(rec,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
            records.append(rec);print('DONE',mode,d,r['metrics']['raw_direction_accuracy'],
                r['metrics']['balanced_accuracy'],r['metrics']['positive_recall'])
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    print('COMPLETE',a.window,len(records))
if __name__=='__main__':main()
