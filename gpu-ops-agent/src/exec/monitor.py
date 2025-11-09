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
