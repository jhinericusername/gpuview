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
