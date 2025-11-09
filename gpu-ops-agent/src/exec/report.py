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
