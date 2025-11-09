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
            subprocess.call(["docker","kill",self.container]); self.container=None
