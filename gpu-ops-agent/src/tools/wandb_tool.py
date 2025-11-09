import os
def env_injection()->dict:
    env={}
    if os.getenv("WANDB_API_KEY"):
        env["WANDB_API_KEY"]=os.getenv("WANDB_API_KEY")
        env["WANDB_PROJECT"]=os.getenv("WANDB_PROJECT","gpu-ops-demo")
    return env
