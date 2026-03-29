from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import EmailTriageEnv
from models import Action, ActionType, EmailCategory, Priority
from tasks import TASK_REGISTRY
from baseline import run_baseline as _run_baseline
from grader import grade_episode

app = FastAPI(
    title="AI Email Triage Environment",
    version="2.2.0",
    description="OpenEnv-compliant email triage with satisfaction scoring, "
                "confidence calibration, and explainable rewards.")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"])

env = EmailTriageEnv()


class TaskRequest(BaseModel):
    task_id: str

class StepRequest(BaseModel):
    action_type: ActionType
    email_id: str
    classification: Optional[EmailCategory] = None
    reply_text: Optional[str] = None
    priority: Optional[Priority] = None
    confidence: Optional[float] = None

class ActionInput(BaseModel):
    action_type: str
    email_id: str
    classification: Optional[str] = None
    reply_text: Optional[str] = None
    priority: Optional[str] = None
    confidence: Optional[float] = None

class GradeRequest(BaseModel):
    task_id: Optional[str] = None
    actions: Optional[List[ActionInput]] = None


@app.get("/")
def root():
    return {
        "name": "AI Email Triage Environment",
        "version": "2.2.0",
        "endpoints": ["/tasks", "/reset", "/step", "/state",
                       "/baseline", "/grader", "/health"],
        "tasks": list(TASK_REGISTRY.keys()),
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/tasks")
def list_tasks():
    return {"tasks": [
        {"id": tid, "difficulty": cfg["difficulty"],
         "description": cfg["description"],
         "required_actions": cfg["required_actions"],
         "num_emails": len(cfg["emails"]),
         "max_steps": cfg.get("max_steps", 100)}
        for tid, cfg in TASK_REGISTRY.items()
    ]}

@app.post("/reset")
def reset(req: TaskRequest):
    try:
        obs = env.reset(req.task_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"observation": obs.model_dump()}

@app.post("/step")
def step(req: StepRequest):
    action = Action(
        action_type=req.action_type, email_id=req.email_id,
        classification=req.classification, reply_text=req.reply_text,
        priority=req.priority, confidence=req.confidence)
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward.model_dump(),
            "done": done, "info": info}

@app.get("/state")
def get_state():
    return env.state().model_dump()

@app.post("/baseline")
def baseline(req: TaskRequest):
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(400, f"Unknown task: {req.task_id}")
    return _run_baseline(req.task_id)

@app.post("/grader")
def grader(req: GradeRequest = None):
    """Grade current episode or evaluate a batch of actions."""
    if req and req.actions and req.task_id:
        if req.task_id not in TASK_REGISTRY:
            raise HTTPException(400, f"Unknown task: {req.task_id}")

        eval_env = EmailTriageEnv()
        eval_env.reset(req.task_id)

        step_results = []
        for a in req.actions:
            action = Action(
                action_type=ActionType(a.action_type),
                email_id=a.email_id,
                classification=EmailCategory(a.classification) if a.classification else None,
                reply_text=a.reply_text,
                priority=Priority(a.priority) if a.priority else None,
                confidence=a.confidence)
            obs, reward, done, info = eval_env.step(action)
            step_results.append({"reward": reward.value, "done": done})
            if done:
                break

        st = eval_env.state()
        cfg = TASK_REGISTRY[req.task_id]
        grading = grade_episode(st.emails, st.statuses,
                                cfg["required_actions"],
                                st.action_history, st.mistakes)
        return {"grading": grading, "step_results": step_results}

    st = env.state()
    if not st.task_id:
        raise HTTPException(400, "No active episode. Call /reset first.")
    cfg = TASK_REGISTRY[st.task_id]
    return grade_episode(st.emails, st.statuses, cfg["required_actions"],
                         st.action_history, st.mistakes)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)