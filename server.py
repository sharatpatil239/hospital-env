"""
FastAPI server exposing HospitalEnv over HTTP.
Required for OpenEnv HTTP validation and Hugging Face Spaces deployment.
All responses are JSON. Errors return appropriate HTTP status codes.
"""

from __future__ import annotations
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from environment import Action, HospitalEnv, TASK_REGISTRY

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Hospital Decision-Making OpenEnv",
    description=(
        "OpenEnv-compliant benchmark: hospital triage and resource allocation "
        "under real-world constraints. Three tasks of increasing difficulty."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store environments per task
_envs: Dict[str, HospitalEnv] = {}


def _get_env(task_id: str) -> HospitalEnv:
    """Return (or create) the HospitalEnv for the given task_id."""
    if task_id not in TASK_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{task_id}'. Available: {list(TASK_REGISTRY.keys())}",
        )

    if task_id not in _envs:
        _envs[task_id] = HospitalEnv(task_id=task_id)

    return _envs[task_id]


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def root() -> Dict[str, Any]:
    """Environment metadata."""
    return {
        "name": "hospital-openenv",
        "version": "1.1.0",
        "tasks": list(TASK_REGISTRY.keys()),
        "description": "Hospital triage and resource allocation benchmark.",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.post("/reset")
def reset(
    task_id: str = Query(default="task_easy", description="Task identifier")
) -> Dict[str, Any]:
    """Reset environment and return initial observation."""
    env = _get_env(task_id)
    obs = env.reset()
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(
    action: Action,
    task_id: str = Query(default="task_easy", description="Task identifier"),
) -> Dict[str, Any]:
    """Apply action → return observation, reward, done, info"""
    env = _get_env(task_id)

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(
    task_id: str = Query(default="task_easy", description="Task identifier")
) -> Dict[str, Any]:
    """Return full environment state (safe serialization)"""
    env = _get_env(task_id)

    try:
        state = env.state()

        # If Pydantic object
        if hasattr(state, "model_dump"):
            return state.model_dump()

        # If dict → recursively serialize
        if isinstance(state, dict):
            def serialize(obj):
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                if isinstance(obj, list):
                    return [serialize(i) for i in obj]
                if isinstance(obj, dict):
                    return {k: serialize(v) for k, v in obj.items()}
                return obj

            return serialize(state)

        return state

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State error: {str(e)}")


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    """List all tasks"""
    return {
        "tasks": [
            {
                "id": "task_easy",
                "name": "Single Patient Triage",
                "difficulty": "easy",
                "max_steps": 1,
                "description": "One critical patient, ample resources.",
            },
            {
                "id": "task_medium",
                "name": "Multi-Patient Prioritization",
                "difficulty": "medium",
                "max_steps": 1,
                "description": "Five patients, limited ICU beds and doctors.",
            },
            {
                "id": "task_hard",
                "name": "Critical Overload with Dynamic Events",
                "difficulty": "hard",
                "max_steps": 3,
                "description": "Eight patients, only 2 ICU beds, dynamic emergency spike.",
            },
        ]
    }


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Hospital OpenEnv server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)