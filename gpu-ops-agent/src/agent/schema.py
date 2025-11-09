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
