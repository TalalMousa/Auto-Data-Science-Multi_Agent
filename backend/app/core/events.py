import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def log_event(run_dir: str, level: str, message: str, details: Any = None):
    """
    Appends a log event to events.jsonl in the run directory.
    """
    event = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "message": message,
        "details": details or {}
    }
    
    # Ensure directory exists before writing
    os.makedirs(run_dir, exist_ok=True)
    
    events_path = os.path.join(run_dir, "events.jsonl")
    
    try:
        with open(events_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.error(f"Failed to write event: {e}")

def get_events(run_dir: str) -> List[Dict[str, Any]]:
    """
    Reads all events from events.jsonl so the Frontend can show them.
    """
    events_path = os.path.join(run_dir, "events.jsonl")
    events = []
    
    if os.path.exists(events_path):
        try:
            with open(events_path, "r") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read events: {e}")
            
    return events