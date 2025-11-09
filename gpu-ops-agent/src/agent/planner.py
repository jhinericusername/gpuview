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
        if edit.get("path")=="/resources/max_minutes" and edit.get("action")=="replace":
            plan_obj.setdefault("resources", {})["max_minutes"]=edit.get("value")
    return plan_obj
