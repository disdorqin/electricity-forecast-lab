"""Official V2.1 research CLI. All write paths resolve under outputs/tabm_v21."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path: sys.path.insert(0,str(ROOT.parent))
from src.TafM_改进源码.config import V2Config, formal_output_dir, load_v21_config
from src.TafM_改进源码.research_reports import transition_report
from src.TafM_改进源码.train import train_target_day
from src.TafM_改进源码.gate import run_v21_gate

def evaluate_range(*args, **kwargs):
    """Lazy proxy keeps non-selector commands usable without SHAP in the environment."""
    from src.TafM_改进源码.evaluate import evaluate_range as implementation
    return implementation(*args, **kwargs)

def main(argv=None):
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    commands=[]
    def add_command(name):
        parser=sub.add_parser(name); parser.add_argument("--config",default=None);commands.append(parser);return parser
    g=add_command("gate")
    s=add_command("selector");s.add_argument("--cutoff",default=None)
    tr=add_command("transition-report");tr.add_argument("--end",required=True)
    t=add_command("train");t.add_argument("--target-day",required=True);t.add_argument("--profile",default="default");t.add_argument("--train-mode",choices=["stage_a","stage_ab","full_retrain"],default="stage_a");t.add_argument("--mode",choices=["A0","A1","A2"],default="A2")
    e=add_command("eval");e.add_argument("--start",required=True);e.add_argument("--end",required=True);e.add_argument("--profile",default="smoke");e.add_argument("--mode",choices=["A0","A1","A2"],default="A2");e.add_argument("--train-mode",choices=["stage_a","stage_ab","full_retrain"],default="stage_a")
    a=add_command("ablate-core");a.add_argument("--start",required=True);a.add_argument("--end",required=True);a.add_argument("--modes",default="A0,A1,A2");a.add_argument("--profile",default="smoke")
    c=add_command("train-strategy");c.add_argument("--start",required=True);c.add_argument("--end",required=True);c.add_argument("--modes",default="stage_a,stage_ab,full_retrain");c.add_argument("--profile",default="smoke")
    d=add_command("direction-benchmark");d.add_argument("--start",required=True);d.add_argument("--end",required=True);d.add_argument("--profile",default="smoke")
    x=add_command("direction-experiment");x.add_argument("--target-day",required=True);x.add_argument("--profile",choices=["smoke","default"],default="smoke");x.add_argument("--mode",choices=["A0","A1","A2"],default="A2")
    x.add_argument("--objective-mode",choices=["joint_v21","dir_only","mag_only"],default="joint_v21")
    x.add_argument("--checkpoint-policy",choices=["v21_guardrail","direction_first","magnitude_first"],default="v21_guardrail")
    x.add_argument("--gradient-policy",choices=["vanilla","direction_protected"],default="vanilla")
    x.add_argument("--architecture-mode",choices=["full_current","tabular_only","temporal_only"],default="full_current")
    x.add_argument("--direction-tabular-mode",choices=["current","strong_only","weak_only"],default="current")
    x.add_argument("--direction-horizon-gate-mode",choices=["current","fixed_08","global_learnable"],default="current")
    x.add_argument("--strong-role-profile",choices=["all","drop_mag","drop_dir"],default="all")
    x.add_argument("--numeric-encoding-mode",choices=["canonical","raw_only","ple_only"],default="canonical")
    x.add_argument("--direction-readout-mode",choices=["shared","segment_bias","segment_heads"],default="shared")
    x.add_argument("--direction-postprocess-mode",choices=["none","segment_logit","regime_logit"],default="none")
    x.add_argument("--direction-class-weight-mode",choices=["unweighted","sqrt_balanced","full_balanced"],default="unweighted")
    x.add_argument("--feature-recovery-profile",choices=["selected222","literature240","all259"],default="selected222")
    x.add_argument("--e4-three-way-split",action="store_true")
    x.add_argument("--stage-a-monitor-days",type=int,default=None)
    x.add_argument("--stage-a-history-window-days",type=int,default=None)
    x.add_argument("--direction-fusion-alpha",type=float,default=None)
    x.add_argument("--direction-bce-min-delta",type=float,default=1e-4);x.add_argument("--magnitude-l1-min-delta",type=float,default=1e-4)
    x.add_argument("--gradient-diagnostics",action="store_true")
    x.add_argument("--gradient-batches-per-epoch",type=int,default=2)
    args=p.parse_args(argv)
    try:
        cfg=load_v21_config(args.config)
        if args.command=="gate":
            gate_path=formal_output_dir()/"gates"/f"phase_a_v21_gate_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
            result={**run_v21_gate(output_path=gate_path),"gate_path":str(gate_path),
                    "config_path":cfg.config_path,"config_sha256":cfg.config_sha256}
        elif args.command=="selector":
            from src.TafM_改进源码.selector import build_selector_manifest
            cutoff=args.cutoff or cfg.selector_cutoff
            if cutoff!=cfg.selector_cutoff: raise ValueError("selector cutoff must match executable YAML config")
            result=build_selector_manifest(cutoff=cutoff,config=cfg)[0]
        elif args.command=="transition-report":
            report,path=transition_report(args.end);result={"path":str(path),"report":report}
        elif args.command=="train":
            r=train_target_day(args.target_day,mode=args.mode,profile=args.profile,train_mode=args.train_mode,config=cfg)
            result={"run_dir":r["run_dir"],"manifest":r["manifest"],"metrics":r["metrics"]}
        elif args.command=="eval":
            r=evaluate_range(args.start,args.end,mode=args.mode,profile=args.profile,train_mode=args.train_mode,config=cfg)
            result={"output_dir":r["output_dir"],"metrics":r["metrics"]}
        elif args.command=="ablate-core":
            from src.TafM_改进源码.pipeline import run_ablate_core
            r=run_ablate_core(args.start,args.end,modes=tuple(x.strip() for x in args.modes.split(",")),profile=args.profile,config=cfg)
            result={"gate_a_dir":r["gate_a"]["output_dir"],"gate_a":r["gate_a"]["comparison"].to_dict(orient="records"),"gate_b_dir":r["gate_b"]["output_dir"],"gate_b":r["gate_b"]["comparison"].to_dict(orient="records")}
        elif args.command=="train-strategy":
            from src.TafM_改进源码.pipeline import run_train_strategy
            r=run_train_strategy(args.start,args.end,modes=tuple(x.strip() for x in args.modes.split(",")),profile=args.profile,config=cfg)
            result={"output_dir":r["output_dir"],"comparison":r["comparison"].to_dict(orient="records")}
        elif args.command=="direction-experiment":
            if args.gradient_policy=="direction_protected" and args.objective_mode!="joint_v21":
                raise ValueError("direction_protected requires --objective-mode joint_v21")
            if args.objective_mode=="mag_only" and args.checkpoint_policy!="magnitude_first":
                raise ValueError("mag_only requires --checkpoint-policy magnitude_first")
            if args.checkpoint_policy=="magnitude_first" and args.objective_mode!="mag_only":
                raise ValueError("magnitude_first is reserved for --objective-mode mag_only")
            r=train_target_day(args.target_day,mode=args.mode,profile=args.profile,train_mode="stage_a",config=cfg,
                objective_mode=args.objective_mode,checkpoint_policy=args.checkpoint_policy,
                direction_bce_min_delta=args.direction_bce_min_delta,magnitude_l1_min_delta=args.magnitude_l1_min_delta,
                gradient_policy=args.gradient_policy,
                gradient_diagnostics=args.gradient_diagnostics,gradient_batches_per_epoch=args.gradient_batches_per_epoch,
                architecture_mode=args.architecture_mode,
                direction_fusion_alpha=args.direction_fusion_alpha,
                direction_tabular_mode=args.direction_tabular_mode,
                direction_horizon_gate_mode=args.direction_horizon_gate_mode,
                strong_role_profile=args.strong_role_profile,
                numeric_encoding_mode=args.numeric_encoding_mode,
                direction_readout_mode=args.direction_readout_mode,
                direction_postprocess_mode=args.direction_postprocess_mode,
                e4_three_way_split=args.e4_three_way_split,
                direction_class_weight_mode=args.direction_class_weight_mode,
                feature_recovery_profile=args.feature_recovery_profile,
                stage_a_monitor_days=args.stage_a_monitor_days,
                stage_a_history_window_days=args.stage_a_history_window_days,
                experiment_only=True)
            result={"run_dir":r["run_dir"],"manifest":r["manifest"],"metrics":r["metrics"]}
        else:
            from src.TafM_改进源码.pipeline import run_direction_benchmark
            r=run_direction_benchmark(args.start,args.end,profile=args.profile,config=cfg)
            result={"output_dir":r["output_dir"],"metrics":r["metrics"].to_dict(orient="records")}
        print(json.dumps(result,indent=2,ensure_ascii=False,default=str)); return 0
    except Exception as e:
        print(f"{type(e).__name__}: {e}",file=sys.stderr);return 2
if __name__=="__main__": raise SystemExit(main())
