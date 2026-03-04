import logging
from typing import Any, Dict, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from backend.app.core.events import log_event
from backend.app.core.llm_agents import (
    analyzer_agent_llm,
    planner_agent_llm,
    reporter_agent_llm,
    trainer_agent_llm,
)
from backend.app.core.ml_artifacts import read_csv_safe, write_state
from backend.app.core.ml_planning import build_dataset_snapshot
from backend.app.core.ml_training import model_candidates, train_and_evaluate

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    run_id: str
    run_dir: str
    data_path: str
    target: str
    metric: str
    dataframe: pd.DataFrame
    dataset_snapshot: Dict[str, Any]
    analysis: Dict[str, Any]
    profile: Dict[str, Any]
    plan: Dict[str, Any]
    training_summary: Dict[str, Any]
    report: Dict[str, Any]
    final_payload: Dict[str, Any]


def _supported_models() -> Dict[str, Any]:
    return {
        "classification": list(model_candidates("classification", None).keys()),
        "regression": list(model_candidates("regression", None).keys()),
    }


def _normalize_task_type(value: Any) -> str:
    return "classification" if str(value).strip().lower() == "classification" else "regression"


def _normalize_metric(value: Any, task_type: str) -> str:
    text = str(value).strip().lower()
    if task_type == "classification":
        mapping = {
            "f1": "F1",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
        }
        return mapping.get(text, "F1")
    mapping = {
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R2",
        "mse": "MSE",
    }
    return mapping.get(text, "RMSE")


def _normalize_direction(value: Any, metric_name: str) -> str:
    text = str(value).strip().lower()
    if text in {"maximize", "minimize"}:
        return text
    return "maximize" if metric_name in {"F1", "Accuracy", "Precision", "Recall", "R2"} else "minimize"


def _normalize_scaling(value: Any) -> str:
    text = str(value).strip().lower()
    return text if text in {"standard", "robust", "minmax"} else "standard"


def _tool_forced_drop_columns(dataset_snapshot: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    forced = []
    reasons = []

    for feature in dataset_snapshot.get("features", []):
        flags = feature.get("risk_flags", {})
        name = feature.get("name")
        if not name:
            continue

        if flags.get("looks_like_row_index"):
            forced.append(name)
            reasons.append(f"Dropped {name} because the tool marked it as a row counter / index feature.")
            continue

        if flags.get("looks_like_identifier"):
            forced.append(name)
            reasons.append(f"Dropped {name} because the tool marked it as an identifier-like column.")
            continue

        if flags.get("looks_like_name"):
            forced.append(name)
            reasons.append(f"Dropped {name} because the tool marked it as a name-like field with weak predictive semantics.")
            continue

        if flags.get("looks_like_free_text"):
            forced.append(name)
            reasons.append(f"Dropped {name} because the tool marked it as free text that would degrade one-hot encoding.")
            continue

        if task_type == "regression" and flags.get("looks_like_postal_code"):
            forced.append(name)
            reasons.append(f"Dropped {name} because postal/ZIP-style codes are unsafe as raw regression features.")
            continue

        if flags.get("high_cardinality") and not feature.get("numeric_summary"):
            forced.append(name)
            reasons.append(f"Dropped {name} because the tool marked it as a high-cardinality categorical/code feature.")
            continue

        if task_type == "regression" and flags.get("looks_like_postal_code") is False and flags.get("high_cardinality"):
            sample_values = feature.get("sample_values", [])
            if sample_values and all(str(value).isdigit() for value in sample_values[:3]):
                forced.append(name)
                reasons.append(f"Dropped {name} because it looks like a high-cardinality code column for regression.")

    unique_forced = []
    seen = set()
    for column in forced:
        if column not in seen:
            seen.add(column)
            unique_forced.append(column)
    return {"columns": unique_forced, "reasons": reasons}


def run_workflow(run_id: str, run_dir: str, data_path: str, target: str, metric: str = "accuracy") -> None:
    write_state(
        run_dir,
        {
            "status": "RUNNING",
            "meta": {
                "summary": "Run started. Waiting for LLM agent pipeline.",
            },
        },
    )

    def analyzer_agent(state: AgentState) -> AgentState:
        log_event(state["run_dir"], "ANALYZER", "Profiling dataset and sending snapshot to the analyzer LLM.")
        df = read_csv_safe(state["data_path"])
        if state["target"] not in df.columns:
            raise ValueError(f"Target column '{state['target']}' was not found in the uploaded CSV.")

        cleaned = df.dropna(subset=[state["target"]]).copy()
        if cleaned.empty:
            raise ValueError("The selected target column has no usable rows after removing missing values.")

        dataset_snapshot = build_dataset_snapshot(cleaned, state["target"], state["metric"], _supported_models())
        analysis = analyzer_agent_llm(dataset_snapshot)

        profile = {
            "rows": dataset_snapshot["rows"],
            "columns": dataset_snapshot["columns"],
            "feature_columns": [item["name"] for item in dataset_snapshot["features"]],
            "numeric_columns": [item["name"] for item in dataset_snapshot["features"] if "numeric_summary" in item],
            "categorical_columns": [item["name"] for item in dataset_snapshot["features"] if "numeric_summary" not in item],
            "task_type": analysis.get("task_type"),
            "target_unique_values": dataset_snapshot["target"]["unique_count"],
            "target_missing_rows_removed": int(df[state["target"]].isna().sum()),
            "class_distribution": dataset_snapshot["target"].get("top_values", {}),
            "missing_by_column": {
                item["name"]: item["missing_ratio"]
                for item in dataset_snapshot["features"]
                if item["missing_ratio"] > 0
            },
            "analysis_summary": analysis.get("analysis_summary", ""),
            "data_risks": analysis.get("data_risks", []),
        }

        normalized_task_type = _normalize_task_type(analysis.get("task_type"))
        analysis["task_type"] = normalized_task_type
        analysis["recommended_metric"] = _normalize_metric(analysis.get("recommended_metric"), normalized_task_type)

        log_event(
            state["run_dir"],
            "ANALYZER",
            f"Analyzer selected {analysis.get('task_type', 'unknown')} with metric {analysis.get('recommended_metric', 'n/a')}.",
        )
        return {"dataframe": cleaned, "dataset_snapshot": dataset_snapshot, "analysis": analysis, "profile": profile}

    def planner_agent(state: AgentState) -> AgentState:
        log_event(state["run_dir"], "PLANNER", "Planner LLM is designing preprocessing and feature engineering.")
        plan = planner_agent_llm(state["dataset_snapshot"], state["analysis"])

        plan["task_type"] = _normalize_task_type(plan.get("task_type") or state["analysis"].get("task_type"))
        preprocessing = plan.setdefault("preprocessing", {})
        preprocessing["scaling"] = _normalize_scaling(preprocessing.get("scaling", "standard"))
        preprocessing["class_weight"] = str(preprocessing.get("class_weight", "none")).lower()
        preprocessing["stratify_split"] = bool(preprocessing.get("stratify_split", plan["task_type"] == "classification"))
        plan["candidate_models"] = [
            name
            for name in plan.get("candidate_models", state["analysis"].get("candidate_models", []))
            if name in _supported_models().get(plan["task_type"], [])
        ]
        if len(plan["candidate_models"]) < 4:
            plan["candidate_models"] = _supported_models()[plan["task_type"]][:5]

        forced_drop = _tool_forced_drop_columns(state["dataset_snapshot"], plan["task_type"])
        current_drop = list(plan.get("drop_columns", []))
        for column in forced_drop["columns"]:
            if column not in current_drop:
                current_drop.append(column)
        plan["drop_columns"] = current_drop

        if not plan.get("feature_engineering_reasoning"):
            plan["feature_engineering_reasoning"] = ["Planner did not add extra transformations beyond the allowed tool set."]
        if forced_drop["reasons"]:
            plan["feature_engineering_reasoning"].extend(forced_drop["reasons"])
        if not preprocessing.get("reasoning"):
            preprocessing["reasoning"] = ["Planner selected the most appropriate preprocessing from the supported tool options."]
        if not plan.get("modeling_reasoning"):
            plan["modeling_reasoning"] = ["Planner selected the strongest supported models for the inferred task type."]

        log_event(
            state["run_dir"],
            "PLANNER",
            f"Planner chose {len(plan['candidate_models'])} candidate models and {preprocessing['scaling']} scaling.",
        )
        return {"plan": plan}

    def trainer_agent(state: AgentState) -> AgentState:
        log_event(state["run_dir"], "TRAINER", "Trainer LLM is validating the plan before execution.")
        training_brief = trainer_agent_llm(state["dataset_snapshot"], state["analysis"], state["plan"])

        state["plan"]["task_type"] = _normalize_task_type(training_brief.get("task_type", state["plan"]["task_type"]))
        state["plan"]["primary_metric"] = _normalize_metric(
            training_brief.get("primary_metric", state["analysis"].get("recommended_metric", "F1")),
            state["plan"]["task_type"],
        )
        state["plan"]["score_direction"] = _normalize_direction(
            training_brief.get("score_direction", "maximize"),
            state["plan"]["primary_metric"],
        )

        candidate_models = [
            name
            for name in training_brief.get("candidate_models", state["plan"].get("candidate_models", []))
            if name in _supported_models().get(state["plan"]["task_type"], [])
        ]
        if len(candidate_models) >= 4:
            state["plan"]["candidate_models"] = candidate_models[:6]

        state["plan"]["trainer_execution_reasoning"] = training_brief.get("execution_reasoning", [])

        log_event(
            state["run_dir"],
            "TRAINER",
            f"Training tool will optimize {state['plan']['primary_metric']} across {len(state['plan']['candidate_models'])} models.",
        )

        training_summary = train_and_evaluate(
            run_dir=state["run_dir"],
            df=state["dataframe"],
            target=state["target"],
            plan=state["plan"],
        )
        log_event(
            state["run_dir"],
            "SUCCESS",
            f"Best model: {training_summary['best_model']} using {state['plan']['primary_metric']}.",
        )
        return {"plan": state["plan"], "training_summary": training_summary}

    def reporter_agent(state: AgentState) -> AgentState:
        log_event(state["run_dir"], "REPORTER", "Reporter LLM is generating the final technical summary.")
        report = reporter_agent_llm(state["analysis"], state["plan"], state["training_summary"])

        summary = report.get("summary") or (
            f"Best model: {state['training_summary']['best_model']} | "
            f"{state['plan']['primary_metric']}: {state['training_summary']['best_metrics']['score']:.4f}"
        )

        payload = {
            "status": "COMPLETED",
            "meta": {
                "summary": summary,
                "run_id": state["run_id"],
                "target": state["target"],
                "task_type": state["plan"]["task_type"],
                "primary_metric": state["plan"]["primary_metric"],
                "score_direction": state["plan"]["score_direction"],
                "profile": state["profile"],
                "analysis": state["analysis"],
                "plan": state["plan"],
                "report": report,
                "results": state["training_summary"]["results"],
                "best_model": state["training_summary"]["best_model"],
                "best_metrics": state["training_summary"]["best_metrics"],
                "features": state["training_summary"]["features"],
                "feature_schema": state["training_summary"]["feature_schema"],
                "categorical_options": state["training_summary"]["categorical_options"],
                "numeric_columns": state["training_summary"]["engineered_numeric_columns"],
                "categorical_columns": state["training_summary"]["engineered_categorical_columns"],
                "top_features": state["training_summary"]["top_features"],
            },
        }
        write_state(state["run_dir"], payload)
        return {"report": report, "final_payload": payload}

    workflow = StateGraph(AgentState)
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("trainer", trainer_agent)
    workflow.add_node("reporter", reporter_agent)
    workflow.set_entry_point("analyzer")
    workflow.add_edge("analyzer", "planner")
    workflow.add_edge("planner", "trainer")
    workflow.add_edge("trainer", "reporter")
    workflow.add_edge("reporter", END)
    app = workflow.compile()

    try:
        app.invoke(
            {
                "run_id": run_id,
                "run_dir": run_dir,
                "data_path": data_path,
                "target": target,
                "metric": metric,
            }
        )
    except Exception as exc:
        logger.exception("Workflow failed")
        log_event(run_dir, "ERROR", str(exc))
        write_state(run_dir, {"status": "FAILED", "error": str(exc)})
