"""
app.py — FastAPI server for AI Email Triage Environment.

Endpoints:
  GET  /          – project metadata
  GET  /health    – liveness probe
  GET  /tasks     – list available tasks
  POST /reset     – reset environment to a task
  POST /step      – submit an agent action
  GET  /state     – retrieve full environment state
  POST /baseline  – run deterministic baseline on a task
  POST /grader    – grade current episode
  POST /submit_score – submit to leaderboard
  GET  /leaderboard  – view leaderboard

Runs on port 7860 (Hugging Face Spaces default).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import EmailTriageEnv
from models import Action, ActionType, EmailCategory, Priority
from tasks import TASK_REGISTRY
from baseline import run_baseline as _run_baseline
from grader import grade_episode

# ═════════════════════════════════════════════════════════════════════════════
# App setup
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI Email Triage Environment",
    version="2.0.0",
    description=(
        "Production-grade OpenEnv environment simulating corporate "
        "email inbox triage. Supports classification, reply generation, "
        "priority assignment, and resolution workflows."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmailTriageEnv()

LEADERBOARD_FILE = Path("leaderboard.json")


# ═════════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ═════════════════════════════════════════════════════════════════════════════

class TaskRequest(BaseModel):
    task_id: str


class StepRequest(BaseModel):
    action_type: ActionType
    email_id: str
    classification: Optional[EmailCategory] = None
    reply_text: Optional[str] = None
    priority: Optional[Priority] = None


class ScoreSubmission(BaseModel):
    agent_name: str
    task_id: str


# ═════════════════════════════════════════════════════════════════════════════
# Leaderboard helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_leaderboard():
    if LEADERBOARD_FILE.exists():
        return json.loads(LEADERBOARD_FILE.read_text())
    return []


def _save_leaderboard(data):
    LEADERBOARD_FILE.write_text(json.dumps(data, indent=2))


# ═════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "name": "AI Email Triage Environment",
        "version": "2.0.0",
        "description": (
            "Simulates corporate email inbox triage — classify, reply, "
            "prioritise, and resolve incoming emails."
        ),
        "endpoints": [
            "/tasks", "/reset", "/step", "/state",
            "/baseline", "/grader", "/health",
            "/submit_score", "/leaderboard",
        ],
        "tasks_available": list(TASK_REGISTRY.keys()),
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/tasks")
def list_tasks():
    tasks = []
    for tid, cfg in TASK_REGISTRY.items():
        tasks.append({
            "id": tid,
            "difficulty": cfg["difficulty"],
            "description": cfg["description"],
            "required_actions": cfg["required_actions"],
            "num_emails": len(cfg["emails"]),
            "max_steps": cfg.get("max_steps", 100),
        })
    return {"tasks": tasks}


@app.post("/reset")
def reset(req: TaskRequest):
    try:
        obs = env.reset(req.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(req: StepRequest):
    action = Action(
        action_type=req.action_type,
        email_id=req.email_id,
        classification=req.classification,
        reply_text=req.reply_text,
        priority=req.priority,
    )
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def get_state():
    return env.state().model_dump()


@app.post("/baseline")
def baseline(req: TaskRequest):
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {req.task_id}",
        )
    return _run_baseline(req.task_id)


@app.post("/grader")
def grader():
    st = env.state()
    if not st.task_id:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first.",
        )
    cfg = TASK_REGISTRY[st.task_id]
    result = grade_episode(
        st.emails,
        st.statuses,
        cfg["required_actions"],
        st.action_history,
        st.mistakes,
    )
    return result


@app.post("/submit_score")
def submit_score(req: ScoreSubmission):
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {req.task_id}",
        )
    result = _run_baseline(req.task_id)
    entry = {
        "agent": req.agent_name,
        "task": req.task_id,
        "score": result["score"],
        "steps": len(result["steps"]),
        "mistakes": result["mistakes"],
    }
    lb = _load_leaderboard()
    lb.append(entry)
    lb.sort(key=lambda x: x["score"], reverse=True)
    _save_leaderboard(lb)
    return {"submitted": entry, "rank": lb.index(entry) + 1, "total_entries": len(lb)}


@app.get("/leaderboard")
def leaderboard():
    return {"leaderboard": _load_leaderboard()}


# ═════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)