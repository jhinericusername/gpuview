import traceback
from ..utils.logger import info, warn, error
from ..tools.local import LocalDocker
from ..tools.wandb_tool import env_injection
from .monitor import wait_with_idle_teardown
from .report import write_report
def execute_plan(plan: dict, report_path: str|None=None):
    idle_m=plan.get("stop_conditions",{}).get("idle_minutes",5)
    max_m =plan.get("stop_conditions",{}).get("max_minutes",45)
    local=LocalDocker(); container=None; runtime_log=[]
    try:
        for step in plan["steps"]:
            tool=step["tool"]; args=step.get("args",{}); on_fail=step.get("on_fail","abort")
            rec={"tool":tool,"args":args}
            try:
                if tool=="create_instance":
                    container=local.create_instance(args.get("image","nvidia/cuda:12.2.0-cudnn9-runtime-ubuntu22.04"))
                elif tool=="git_clone":
                    local.git_clone(args["repo"], args.get("dest","/workspace/app"))
                elif tool=="run_command":
                    rc=local.exec(args["cmd"], env=env_injection())
                    if rc!=0: raise RuntimeError(f"command failed rc={rc}")
                elif tool=="monitor":
                    wait_with_idle_teardown(container, idle_m, max_m, local.terminate); container=None
                elif tool=="terminate_instance":
                    local.terminate(); container=None
                else:
                    warn(f"Unknown tool {tool}, skipping")
                rec["result"]="OK"
            except Exception as e:
                rec["result"]=f"ERROR: {e}"
                if on_fail=="retry":
                    info(f"Retrying {tool} once…")
                    if tool=="run_command":
                        rc=local.exec(args["cmd"], env=env_injection())
                        rec["result"]="OK" if rc==0 else f"ERROR: rc={rc}"
                    else:
                        error("Retry unsupported for this tool in demo")
                elif on_fail=="skip":
                    warn(f"Skipping {tool}")
                else:
                    raise
            finally:
                runtime_log.append(rec)
    finally:
        try:
            if container: warn("Forcing teardown in finally-block"); local.terminate()
        except Exception: error("Teardown failed")
        if report_path: write_report(report_path, plan, runtime_log)
