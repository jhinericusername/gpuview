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
