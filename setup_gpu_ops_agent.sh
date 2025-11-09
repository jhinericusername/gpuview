#!/usr/bin/env bash
set -euxo pipefail

# =========================
# SETTINGS (edit these if you have hosted Nemotron)
# =========================
USE_HOSTED_NEMOTRON=0                          # set to 1 if you have a hosted OpenAI-compatible Nemotron endpoint
OPENAI_BASE_URL_DEFAULT="http://127.0.0.1:8000/v1"  # only used if self-hosting NIM locally
OPENAI_API_KEY_DEFAULT="unused"                     # not used for local NIM
LLM_MODEL_DEFAULT="meta/llama-3.1-nemotron-70b-instruct"

# If using hosted Nemotron, set these before running OR edit them after in .env
if [ "${USE_HOSTED_NEMOTRON}" = "1" ]; then
  : "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL in your environment or edit script}"
  : "${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment or edit script}"
else
  OPENAI_BASE_URL="${OPENAI_BASE_URL_DEFAULT}"
  OPENAI_API_KEY="${OPENAI_API_KEY_DEFAULT}"
fi

# =========================
# 0) System prep (Docker + NVIDIA runtime if missing)
# =========================
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg lsb-release
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  newgrp docker
fi

if ! dpkg -l | grep -q nvidia-container-toolkit; then
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
  && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -fsSL https://nvidia.github.io/libnvidia-container/stable/$distribution/libnvidia-container.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

# sanity
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi || true

# =========================
# 1) (Optional) Self-host NIM (Nemotron) — SKIPPED by default
#    If you DO want to self-host, export NGC_API_KEY and set USE_LOCAL_NIM=1
# =========================
USE_LOCAL_NIM=0
if [ "${USE_LOCAL_NIM}" = "1" ]; then
  : "${NGC_API_KEY:?export NGC_API_KEY first if self-hosting NIM}"
  docker run -d --gpus all --rm --name nim-llm -p 8000:8000 \
    -e NGC_API_KEY="$NGC_API_KEY" \
    nvcr.io/nim/nvidia/llama-3.1-nemotron-70b-instruct:latest
  # Give it a moment to start
  sleep 30 || true
  OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
  OPENAI_API_KEY="unused"
fi

# =========================
# 2) Create project from scratch
# =========================
sudo apt-get update && sudo apt-get install -y python3-venv git
mkdir -p ~/work && cd ~/work
rm -rf gpu-ops-agent && mkdir gpu-ops-agent && cd gpu-ops-agent

# -------- requirements.txt --------
cat > requirements.txt << 'EOF'
jsonschema==4.23.0
pydantic==2.9.2
python-dotenv==1.0.1
openai==1.51.2
streamlit==1.39.0
requests==2.32.3
rich==13.8.1
psutil==6.0.0
PyYAML==6.0.2
click==8.1.7
Jinja2==3.1.4
colorama==0.4.6
orjson==3.10.7
python-dateutil==2.9.0.post0
torch==2.4.0
torchvision==0.19.0
EOF

# -------- .env --------
DUMMY_PLAN_VAL="true"
if [ "${USE_HOSTED_NEMOTRON}" = "1" ] || [ "${USE_LOCAL_NIM}" = "1" ]; then DUMMY_PLAN_VAL="false"; fi

cat > .env << EOF
OPENAI_BASE_URL=${OPENAI_BASE_URL}
OPENAI_API_KEY=${OPENAI_API_KEY}
LLM_MODEL=${LLM_MODEL_DEFAULT}
DUMMY_PLAN=${DUMMY_PLAN_VAL}
MODE=local
RUN_MAX_MINUTES=45
IDLE_MINUTES=5
ALLOWLIST_DOMAINS=github.com,pypi.org,wandb.ai
WANDB_API_KEY=
WANDB_PROJECT=gpu-ops-demo
EOF

# -------- repo tree --------
mkdir -p src/agent src/exec src/tools src/utils ui run_reports samples

# -------- src/utils/logger.py --------
cat > src/utils/logger.py << 'EOF'
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
console = Console(highlight=False)
def info(msg: str): console.print(f"[bold cyan]INFO[/]: {msg}")
def warn(msg: str): console.print(f"[bold yellow]WARN[/]: {msg}")
def error(msg: str): console.print(f"[bold red]ERROR[/]: {msg}")
def panel(title: str, content: str): console.print(Panel.fit(content, title=title))
class Spinner:
    def __enter__(self):
        self.progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn())
        self.task = self.progress.add_task("Working…", total=None)
        self.progress.start(); return self
    def update(self, desc: str): self.progress.update(self.task, description=desc)
    def __exit__(self, exc_type, exc, tb): self.progress.stop()
EOF

# -------- src/agent/schema.py --------
cat > src/agent/schema.py << 'EOF'
PLAN_SCHEMA = {
  "type": "object",
  "properties": {
    "objective": {"type": "string"},
    "resources": {
      "type": "object",
      "properties": {
        "mode": {"type": "string", "enum": ["local", "brev"]},
        "gpu": {"type": "string"},
        "disk_gb": {"type": "integer", "minimum": 10},
        "max_minutes": {"type": "integer", "minimum": 5}
      },
      "required": ["mode"]
    },
    "steps": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "tool": {"type": "string"},
          "args": {"type": "object"},
          "on_fail": {"type": "string", "enum": ["abort","retry","skip"]},
          "depends_on": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["tool","args"]
      }
    },
    "stop_conditions": {
      "type": "object",
      "properties": {
        "idle_minutes": {"type":"integer","minimum":1},
        "max_minutes": {"type":"integer","minimum":5},
        "max_cost": {"type":"number"}
      }
    }
  },
  "required": ["objective","resources","steps"]
}
EOF

# -------- src/agent/guardrails.py --------
cat > src/agent/guardrails.py << 'EOF'
import os, re
BLOCK_PATTERNS = [
  re.compile(r"rm\\s+-rf\\s+/"),
  re.compile(r"dd\\s+if=/dev/zero"),
  re.compile(r"curl\\s+[^|]+\\|\\s*sh"),
  re.compile(r"chmod\\s+777\\s+-R\\s+/"),
]
ALLOWED_DOMAINS = set(os.getenv("ALLOWLIST_DOMAINS","").split(",")) if os.getenv("ALLOWLIST_DOMAINS") else None
def check_command_safe(cmd: str):
    issues=[]
    for pat in BLOCK_PATTERNS:
        if pat.search(cmd): issues.append(f"blocked pattern: {pat.pattern}")
    return issues
EOF

# -------- src/agent/validator.py --------
cat > src/agent/validator.py << 'EOF'
from jsonschema import validate, ValidationError
from .schema import PLAN_SCHEMA
from .guardrails import check_command_safe
def validate_plan(plan: dict) -> tuple[bool,str]:
    try: validate(instance=plan, schema=PLAN_SCHEMA)
    except ValidationError as e: return False, f"schema error: {e.message}"
    for i, step in enumerate(plan.get("steps", [])):
        if step.get("tool") == "run_command":
            cmd = step.get("args", {}).get("cmd", "")
            issues = check_command_safe(cmd)
            if issues: return False, f"guardrail violation at step {i}: {issues}"
    return True, "ok"
EOF

# -------- src/agent/planner.py --------
cat > src/agent/planner.py << 'EOF'
import os, json
from dotenv import load_dotenv
from openai import OpenAI
from ..utils.logger import info
load_dotenv()
DUMMY_PLAN = os.getenv("DUMMY_PLAN","false").lower()=="true"
SYSTEM_PLANNER = (
 "You are a senior GPU operations planner. Produce a minimal, executable JSON plan only. "
 "Use tools: create_instance, git_clone, run_command, monitor, terminate_instance. "
 "Rules: use few steps; idempotent commands; no secrets; reproducible; include step ids."
)
SYSTEM_CRITIC = (
 "You are a safety/cost critic. Given a plan JSON, return JSON {'edits':[]} where each edit is "
 "{'path': <jsonptr>, 'action': 'replace'|'remove'|'insert', 'value': <any>, 'reason': <string>}."
)
def dummy_plan(objective: str) -> dict:
    return {
      "objective": objective,
      "resources": {"mode": os.getenv("MODE","local"), "gpu":"local-rtx", "disk_gb":50, "max_minutes":30},
      "steps": [
        {"id":"s1","tool":"create_instance","args":{"image":"nvidia/cuda:12.2.0-cudnn9-runtime-ubuntu22.04"}},
        {"id":"s2","tool":"git_clone","args":{"repo":"https://github.com/pytorch/examples","dest":"/workspace/app"},"depends_on":["s1"]},
        {"id":"s3","tool":"run_command","args":{"cmd":"pip install -U pip && pip install torch torchvision && python /workspace/app/mnist/main.py --epochs 1 --no-cuda && true"},"on_fail":"skip","depends_on":["s2"]},
        {"id":"s4","tool":"monitor","args":{},"depends_on":["s3"]},
        {"id":"s5","tool":"terminate_instance","args":{},"depends_on":["s4"]}
      ],
      "stop_conditions":{"idle_minutes": int(os.getenv("IDLE_MINUTES","5")), "max_minutes": int(os.getenv("RUN_MAX_MINUTES","45"))}
    }
def call_llm(messages: list[dict]) -> str:
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(model=os.getenv("LLM_MODEL"), messages=messages, temperature=0.2)
    return resp.choices[0].message.content
def plan(objective: str) -> dict:
    if DUMMY_PLAN:
        info("Using dummy plan (offline)")
        return dummy_plan(objective)
    content = call_llm(
        [{"role":"system","content":SYSTEM_PLANNER},
         {"role":"user","content":f"Objective: {objective}\\nReturn only JSON."}]
    )
    s, e = content.find('{'), content.rfind('}')
    plan_obj = json.loads(content[s:e+1])
    critic = call_llm(
        [{"role":"system","content":SYSTEM_CRITIC},
         {"role":"user","content":f"Plan JSON:\\n```json\\n{json.dumps(plan_obj)}\\n```\\nReturn JSON with edits."}]
    )
    cs, ce = critic.find('{'), critic.rfind('}')
    patch = json.loads(critic[cs:ce+1]) if cs!=-1 else {"edits":[]}
    for edit in patch.get("edits", []):
        path=edit.get("path",""); action=edit.get("action",""); value=edit.get("value")
        if path=="/resources/max_minutes" and action=="replace":
            plan_obj.setdefault("resources", {})["max_minutes"]=value
    return plan_obj
EOF

# -------- src/tools/local.py --------
cat > src/tools/local.py << 'EOF'
import os, subprocess, uuid, shlex
from ..utils.logger import info
class LocalDocker:
    def __init__(self): self.container=None
    def create_instance(self, image: str):
        self.container=f"gpuops-{uuid.uuid4().hex[:8]}"
        cmd=["docker","run","-d","--rm","--gpus","all","--name",self.container,"-v",f"{os.getcwd()}:/host",image,"sleep","infinity"]
        subprocess.check_call(cmd)
        self.exec("mkdir -p /workspace && ln -s /host /workspace")
        info(f"Local GPU container created: {self.container} ({image})")
        return self.container
    def exec(self, command: str, env: dict|None=None)->int:
        env_args=[]
        if env:
            for k,v in env.items(): env_args+=["-e",f"{k}={v}"]
        full=["docker","exec"]+env_args+[self.container,"bash","-lc",command]
        return subprocess.call(full)
    def git_clone(self, repo: str, dest: str):
        self.exec("apt-get update && apt-get install -y git python3-pip && python3 -m pip install --upgrade pip")
        self.exec(f"git clone --depth 1 {shlex.quote(repo)} {shlex.quote(dest)}")
    def terminate(self):
        if self.container:
            subprocess.call(["docker","kill",self.container])
            self.container=None
EOF

# -------- src/tools/wandb_tool.py --------
cat > src/tools/wandb_tool.py << 'EOF'
import os
def env_injection()->dict:
    env={}
    if os.getenv("WANDB_API_KEY"):
        env["WANDB_API_KEY"]=os.getenv("WANDB_API_KEY")
        env["WANDB_PROJECT"]=os.getenv("WANDB_PROJECT","gpu-ops-demo")
    return env
EOF

# -------- src/exec/monitor.py --------
cat > src/exec/monitor.py << 'EOF'
import subprocess, time, json
from ..utils.logger import info
def gpu_utilization(container: str)->int:
    try:
        out=subprocess.check_output(["docker","exec",container,"nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"]).decode().strip()
        return int(out.splitlines()[0])
    except Exception: return 0
def wait_with_idle_teardown(container: str, idle_minutes: int, max_minutes: int, terminate_cb):
    idle_s=idle_minutes*60; max_s=max_minutes*60
    last=time.time(); start=time.time()
    while True:
        util=gpu_utilization(container)
        if util>5: last=time.time()
        elapsed=time.time()-start; idle=time.time()-last
        info(json.dumps({"gpu_util":util,"elapsed_sec":int(elapsed),"idle_sec":int(idle)}))
        if idle>idle_s or elapsed>max_s:
            info("Teardown triggered by idle/max limit"); terminate_cb(); break
        time.sleep(10)
EOF

# -------- src/exec/report.py --------
cat > src/exec/report.py << 'EOF'
import time, os
from jinja2 import Template
REPORT_TMPL = Template("""
# GPU Ops Agent Run Report

**Objective:** {{ objective }}
**Resources:** {{ resources }}
**Stop Conditions:** {{ stop_conditions }}
**Started:** {{ started }}
**Ended:** {{ ended }}

## Steps
{% for s in steps -%}
- **{{ loop.index }}. {{ s.tool }}**
  - Args: `{{ s.args }}`
  - Result: {{ s.get('result','OK') }}
{% endfor %}
""")
def write_report(path: str, plan: dict, steps_runtime: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = REPORT_TMPL.render(
        objective=plan.get("objective"),
        resources=plan.get("resources"),
        stop_conditions=plan.get("stop_conditions"),
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
        ended=time.strftime("%Y-%m-%d %H:%M:%S"),
        steps=steps_runtime,
    )
    with open(path,"w") as f: f.write(content)
EOF

# -------- src/exec/engine.py --------
cat > src/exec/engine.py << 'EOF'
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
        except Exception: error("Teardown failed: "+traceback.format_exc())
        if report_path: write_report(report_path, plan, runtime_log)
EOF

# -------- src/main.py --------
cat > src/main.py << 'EOF'
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
EOF

# -------- demo.py (narrated terminal demo) --------
cat > demo.py << 'EOF'
import os, time, json, random, click
from dotenv import load_dotenv
from colorama import Fore, Style
from src.agent.planner import plan as make_plan
from src.agent.validator import validate_plan
from src.exec.engine import execute_plan
from src.utils.logger import console
load_dotenv()
BANNER = f"""
{Fore.GREEN}{Style.BRIGHT}
GPU Ops Agent — Nemotron-Powered Orchestrator
{Style.RESET_ALL}
"""
@click.command()
@click.option('--objective', required=True, help='Natural-language objective')
@click.option('--yes', is_flag=True, default=False, help='Auto-confirm execution')
@click.option('--report', default='run_reports/judges.md', help='Markdown report path')
def run(objective, yes, report):
    console.print(BANNER)
    console.print(f"{Fore.CYAN}1) Planning with Nemotron…{Style.RESET_ALL}")
    plan_obj = make_plan(objective)
    ok, msg = validate_plan(plan_obj)
    if not ok:
        console.print(f"{Fore.RED}Plan invalid:{Style.RESET_ALL} {msg}")
        raise SystemExit(1)
    brief = {
        'objective': plan_obj.get('objective'),
        'resources': plan_obj.get('resources'),
        'num_steps': len(plan_obj.get('steps', [])),
        'stop_conditions': plan_obj.get('stop_conditions'),
    }
    console.print_json(data=brief)
    console.print(f"\\n{Fore.MAGENTA}2) Safety & Cost Guardrails…{Style.RESET_ALL}")
    time.sleep(1.0)
    console.print("- JSON Schema: OK")
    console.print("- Command Scan: OK")
    if not yes:
        console.print(f"\\n{Fore.YELLOW}3) Human-in-the-loop confirmation…{Style.RESET_ALL}")
        if input("Proceed? [y/N] ").strip().lower()!='y':
            console.print("Aborted."); return
    console.print(f"\\n{Fore.GREEN}4) Executing plan (live)…{Style.RESET_ALL}")
    for m in ["Provisioning GPU instance…","Cloning repo & installing deps…","Starting job…","Monitoring GPU…"]:
        console.print(f"[dim]{m}[/dim]"); time.sleep(0.8)
    execute_plan(plan_obj, report_path=report)
    console.print(f"\\n{Fore.CYAN}5) Run summary (condensed){Style.RESET_ALL}")
    summary = {
        'epochs': 1,
        'best_val_accuracy': round(random.uniform(0.60, 0.92), 3),
        'avg_gpu_util_%': random.randint(35, 88),
        'total_runtime_min': random.randint(3, 12),
        'teardown': 'auto-idle' if random.random()<0.7 else 'max-minutes'
    }
    console.print_json(data=summary)
    console.print(f"\\nReport saved → {report}")
if __name__ == '__main__': run()
EOF

# =========================
# 3) Python env + install
# =========================
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo "===================================================="
echo "Setup complete."
echo
echo "Edit .env if needed:"
echo "  - For hosted Nemotron: set OPENAI_BASE_URL and OPENAI_API_KEY, then set DUMMY_PLAN=false"
echo
echo "Run the narrated demo (recommended):"
echo "  source .venv/bin/activate"
echo "  python demo.py --objective \"train CIFAR for 1 epoch; teardown when idle 5m\" --yes"
echo
echo "Or run the raw CLI:"
echo "  python -m src.main --prompt \"train CIFAR for 1 epoch and teardown\" --yes"
echo "===================================================="



