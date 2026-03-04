import os
import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# --- Data Models ---
class RunState(BaseModel):
    run_id: str
    dataset_path: str
    target: str
    metric: str
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    progress: float = 0.0
    message: str = "Initializing..."
    error: Optional[str] = None
    meta: Dict[str, Any] = {}

# --- In-Memory Store ---
# In a real app, use SQLite/Postgres. For now, a dict is fine.
_RUNS: Dict[str, RunState] = {}

# --- Functions ---

def create_run(run_id: str, dataset_path: str, target: str, metric: str) -> RunState:
    """Initializes a new run record."""
    state = RunState(
        run_id=run_id,
        dataset_path=dataset_path,
        target=target,
        metric=metric
    )
    _RUNS[run_id] = state
    return state

def get_run(run_id: str) -> RunState:
    """Retrieves a run by ID."""
    if run_id not in _RUNS:
        raise FileNotFoundError(f"Run {run_id} not found")
    return _RUNS[run_id]

def load_state(run_id: str) -> RunState:
    """Alias for get_run to match runner.py expectation."""
    return get_run(run_id)

def save_state(state: RunState):
    """Updates the run record."""
    _RUNS[state.run_id] = state

def update_run_status(run_id: str, message: str, progress: float):
    """Helper to safely update status from anywhere."""
    try:
        if run_id in _RUNS:
            _RUNS[run_id].message = message
            _RUNS[run_id].progress = progress
    except Exception as e:
        logger.error(f"Failed to update run status: {e}")