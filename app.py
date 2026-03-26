from fastapi import FastAPI
from tasks import TASK_REGISTRY
from baseline import run_baseline
from env import EmailTriageEnv

app = FastAPI()   # 🔥 THIS LINE IS CRITICAL

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/tasks")
def get_tasks():
    return [
        {
            "id": tid,
            "difficulty": t["difficulty"],
            "description": t["description"],
        }
        for tid, t in TASK_REGISTRY.items()
    ]

@app.get("/baseline")
def run_all_baselines():
    return {tid: run_baseline(tid) for tid in TASK_REGISTRY}

@app.post("/grader")
def run_grader():
    env = EmailTriageEnv()
    env.reset()
    return {"message": "Environment ready"}