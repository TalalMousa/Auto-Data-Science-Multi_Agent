from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
import shutil
import json
import joblib
import pandas as pd

from backend.app.core.runner import run_workflow
from backend.app.core.events import get_events

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "../uploads")
RUNS_DIR = os.path.join(BASE_DIR, "../runs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

class RunRequest(BaseModel):
    filename: str
    target: str
    metric: str = "accuracy"

class PredictRequest(BaseModel):
    run_id: str
    inputs: dict

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/runs")
async def start_run(request: RunRequest, background_tasks: BackgroundTasks):
    run_id = uuid.uuid4().hex[:8]
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    data_path = os.path.join(UPLOAD_DIR, request.filename)
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="File not found")

    background_tasks.add_task(
        run_workflow, 
        run_id=run_id, 
        run_dir=run_dir, 
        data_path=data_path, 
        target=request.target,
        metric=request.metric
    )
    return {"run_id": run_id, "status": "started"}

@app.get("/runs/{run_id}")
def get_run_status(run_id: str):
    path = os.path.join(RUNS_DIR, run_id, "state.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"status": "RUNNING"}

@app.get("/runs/{run_id}/events")
def get_run_events(run_id: str):
    return {"events": get_events(os.path.join(RUNS_DIR, run_id))}

@app.get("/runs/{run_id}/model")
def get_model(run_id: str):
    path = os.path.join(RUNS_DIR, run_id, "model.pkl")
    if os.path.exists(path):
        return FileResponse(path, filename=f"model_{run_id}.pkl", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Model not found")

@app.get("/runs/{run_id}/plot")
def get_plot(run_id: str):
    path = os.path.join(RUNS_DIR, run_id, "plot.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Plot not found")

@app.get("/runs/{run_id}/importance")
def get_importance(run_id: str):
    path = os.path.join(RUNS_DIR, run_id, "importance.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Importance plot not found")

# --- NEW: PREDICT ENDPOINT ---
@app.post("/predict")
async def make_prediction(request: PredictRequest):
    run_dir = os.path.join(RUNS_DIR, request.run_id)
    model_path = os.path.join(run_dir, "model.pkl")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        pipeline = joblib.load(model_path)
        input_df = pd.DataFrame({k: [v] for k, v in request.inputs.items()})
        prediction = pipeline.predict(input_df)[0]
        
        if hasattr(prediction, "item"):
            prediction = prediction.item()
            
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))