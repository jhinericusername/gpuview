import argparse
from dotenv import load_dotenv
from src.agent.planner import plan as make_plan
from src.agent.validator import validate_plan
from src.exec.engine import execute_plan
from src.utils.logger import panel, info, error
def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--yes', action='store_true')
    ap.add_argument('--report', default='run_reports/demo.md')
    args = ap.parse_args()
    pl = make_plan(args.prompt)
    ok, msg = validate_plan(pl)
    if not ok: error(f"Plan invalid: {msg}"); return
    panel("Plan", str(pl))
    if not args.yes:
        resp = input("Execute this plan? [y/N]: ")
        if resp.strip().lower()!='y': info("Aborted"); return
    execute_plan(pl, report_path=args.report)
if __name__ == '__main__': main()
