"""CLI only: frozen configs and explicit execution gates."""
import argparse
from .config import FAST24
from .walkforward import run_walkforward

def main():
    p = argparse.ArgumentParser(description="Isolated MSR-V2 Native96 research runner")
    p.add_argument("--mode",choices=("smoke","single-day","fast24-lowhigh","fast24-full","fast24-screen"),required=True)
    p.add_argument("--date",action="append",help="repeat for a two-day smoke")
    p.add_argument("--device",choices=("cpu","cuda","auto"),default="cpu")
    p.add_argument("--profile",choices=("scientific","no-refit-fast-dev","fast-screen"),default="scientific")
    p.add_argument("--output-root")
    p.add_argument("--source")
    p.add_argument("--review-receipt")
    args = p.parse_args()
    dates = list(FAST24) if args.mode.startswith("fast24") else (args.date or list(FAST24[:2]))
    if args.mode=="single-day" and len(dates)!=1:
        p.error("single-day requires exactly one --date")
    run,manifest = run_walkforward(dates,mode=args.mode,device=args.device,profile=args.profile,
        output_root=args.output_root,source=args.source,review_receipt=args.review_receipt)
    print("RUN_ROOT="+str(run))
    print("STATUS=CPU_SMOKE_COMPLETE" if args.mode=="smoke" else "STATUS=RETROSPECTIVE_COMPLETE")

if __name__=="__main__":
    main()
