# DS Auto RAG

An autonomous data science app that takes a CSV, chooses a target column, runs a LangGraph-based multi-agent workflow, benchmarks multiple ML models, and exposes the best model through a simple Streamlit UI.

## What It Does

- Uploads a CSV from the Streamlit frontend.
- Runs an LLM-driven agent graph for:
  - dataset analysis
  - preprocessing and feature-planning
  - training-plan validation
  - final reporting
- Trains multiple supervised models for classification or regression.
- Selects the best model using a task-appropriate metric.
- Saves:
  - a trained model artifact
  - a performance plot
  - a feature importance plot when available
- Lets the user test the saved model directly in the UI.

## Current Architecture

The app uses:

- `FastAPI` for the backend API
- `Streamlit` for the frontend
- `LangGraph` for agent orchestration
- `langchain-openai` for LLM-based agents
- `scikit-learn` and `xgboost` for training

### Agent Flow

The backend workflow is defined in [backend/app/core/runner.py](/C:/Users/Basel/Downloads/DS_Auto_Rag/backend/app/core/runner.py).

1. `Analyzer`
   Reads the dataset snapshot and infers the task type and recommended metric.
2. `Planner`
   Chooses columns to drop, date columns, missing indicators, scaling, and candidate models.
3. `Trainer`
   Validates the plan, then the deterministic training tool executes it.
4. `Reporter`
   Summarizes the final run for the UI.

## Project Structure

```text
backend/
  app/
    main.py
    core/
      runner.py
      llm_agents.py
      ml_planning.py
      ml_training.py
      ml_visuals.py
      ml_artifacts.py
      events.py
  runs/
  uploads/
frontend/
  streamlit_app.py
requirements.txt
README.md
```

## Requirements

- Python 3.10+ recommended
- An OpenAI API key

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Environment Variables

The LLM agents require an OpenAI API key.

Set it in PowerShell before starting the backend:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key"
```

Optional:

```powershell
$env:OPENAI_MODEL="gpt-4o-mini"
```

## Running the App

### 1. Start the Backend

```powershell
python -m uvicorn backend.app.main:app --reload
```

Backend default URL:

- `http://127.0.0.1:8000`

### 2. Start the Frontend

Open a second terminal, activate the same virtual environment, then run:

```powershell
python -m streamlit run frontend/streamlit_app.py
```

Frontend default URL:

- `http://localhost:8501`

## How to Use

1. Open the Streamlit UI.
2. Upload a CSV file.
3. Select the target column.
4. Launch the agent workflow.
5. Wait for training to complete.
6. Review:
   - leaderboard
   - performance plot
   - feature importance
   - reasoning
7. Use the `Predict` tab to test the saved model.

## Supported Modeling

### Classification

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- HistGradientBoosting Classifier
- AdaBoost Classifier
- Extra Trees Classifier
- XGBoost Classifier (when installed)

### Regression

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor
- AdaBoost Regressor
- Extra Trees Regressor
- XGBoost Regressor (when installed)

## Feature Handling

The system combines LLM planning with tool-side guardrails.

It will try to drop columns that look like:

- row counters / index columns
- IDs / record identifiers
- postal / ZIP code fields
- name-like columns
- free-text columns
- high-cardinality categorical noise

Date-like columns can be expanded into:

- year
- month
- day
- day of week

## Generated Artifacts

For each run, the backend can save files under `backend/runs/<run_id>/`:

- `state.json`
- `events.jsonl`
- `model.pkl`
- `plot.png`
- `importance.png`

## API Endpoints

Defined in [backend/app/main.py](/C:/Users/Basel/Downloads/DS_Auto_Rag/backend/app/main.py).

- `POST /upload`
- `POST /runs`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/events`
- `GET /runs/{run_id}/model`
- `GET /runs/{run_id}/plot`
- `GET /runs/{run_id}/importance`
- `POST /predict`

## Troubleshooting

### `ModuleNotFoundError`

Usually means packages were installed into a different Python interpreter.

Use:

```powershell
python -m pip install -r requirements.txt
```

Check the active interpreter:

```powershell
where python
where pip
python -m pip show fastapi
python -m pip show langchain-openai
```

### `OPENAI_API_KEY is required for LLM-based agents`

Set the key before starting the backend:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key"
```

### XGBoost not being used

Make sure dependencies were reinstalled after `xgboost` was added:

```powershell
python -m pip install -r requirements.txt
```

## Notes

- Old runs are not automatically migrated when the workflow logic changes.
- If results look stale or storage grows too large, clear `backend/runs` and `backend/uploads`.
- The planning agents are LLM-based, but actual model fitting and artifact creation are done with deterministic tools.
