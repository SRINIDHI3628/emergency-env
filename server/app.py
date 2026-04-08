"""
app.py
======
FastAPI server exposing the environment as an HTTP API.
Required endpoints for hackathon validation:
    POST /reset
    POST /step
    GET  /state
    GET  /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json

from env.environment import EmergencyEnv
from env.models import Observation
from tasks import TASKS

app = FastAPI(
    title="Emergency Resource Allocation - OpenEnv",
    description="AI environment for emergency dispatch decision making",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance (single session for demo)
_env: Optional[EmergencyEnv] = None
_current_task: str = "easy"


# ─── Request/Response schemas ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    hospital_id: int
    ambulance_id: int


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "running"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "EmergencyEnv"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    global _env, _current_task
    task_name = req.task if req is not None else "easy"
    if task_name not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_name}. Choose from {list(TASKS.keys())}")
    _current_task = task_name
    _env = EmergencyEnv(TASKS[task_name]["config"])
    observation = _env.reset()
    return {
        "task": task_name,
        "observation": observation.dict(),
        "message": f"Environment reset for task '{task_name}'",
    }


@app.post("/step")
def step(req: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    if _env.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start again.")

    action = {"hospital_id": req.hospital_id, "ambulance_id": req.ambulance_id}
    result = _env.step(action)
    return {
        "observation": result["observation"].dict(),
        "reward": result["reward"].dict(),
        "done": result["done"],
        "info": result["info"],
    }


@app.get("/state")
def get_state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    observation = Observation(**_env.state())
    return observation.dict()


@app.get("/tasks")
def list_tasks():
    return {
        name: {
            "description": data["config"]["description"],
            "num_patients": len(data["config"].get("patients", [])),
            "num_hospitals": len(data["config"]["hospitals"]),
            "num_ambulances": len(data["config"]["ambulances"]),
        }
        for name, data in TASKS.items()
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
