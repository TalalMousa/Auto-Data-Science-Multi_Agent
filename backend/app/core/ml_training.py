import math
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from backend.app.core.events import log_event
from backend.app.core.ml_artifacts import FeatureEngineeringTransformer, PredictionArtifact, make_one_hot_encoder
from backend.app.core.ml_planning import build_feature_schema
from backend.app.core.ml_visuals import extract_importance, save_importance_plot, save_performance_plot

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover - optional until dependencies are installed
    XGBClassifier = None
    XGBRegressor = None


def build_preprocessor(numeric_columns: List[str], categorical_columns: List[str], scaling: str) -> ColumnTransformer:
    transformers = []
    scaler = StandardScaler()
    if scaling == "robust":
        scaler = RobustScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()

    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", make_one_hot_encoder())]),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No usable features remain after planning.")
    return ColumnTransformer(transformers=transformers)


def model_candidates(task_type: str, class_weight: Optional[str], selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
    normalized_task_type = "classification" if str(task_type).lower() == "classification" else "regression"
    effective_class_weight = None if str(class_weight).lower() in {"none", "null", ""} else class_weight

    if normalized_task_type == "classification":
        available = {
            "LogisticRegression": LogisticRegression(max_iter=2000, class_weight=effective_class_weight, random_state=42),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1, class_weight=effective_class_weight),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=42),
            "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
            "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight=effective_class_weight),
        }
        if XGBClassifier is not None:
            available["XGBoostClassifier"] = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            )
    else:
        available = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
            "AdaBoostRegressor": AdaBoostRegressor(random_state=42),
            "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        }
        if XGBRegressor is not None:
            available["XGBoostRegressor"] = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
            )

    if not selected_models:
        return available

    filtered = {name: available[name] for name in selected_models if name in available}
    return filtered or available


def evaluate_predictions(task_type: str, primary_metric: str, y_true: pd.Series, predictions) -> Dict[str, Any]:
    if task_type == "regression":
        mse_value = mean_squared_error(y_true, predictions)
        metrics = {
            "R2": r2_score(y_true, predictions),
            "RMSE": math.sqrt(mse_value),
            "MAE": mean_absolute_error(y_true, predictions),
            "MSE": mse_value,
        }
        return {"metrics": metrics, "score": metrics[primary_metric]}

    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, predictions, average="weighted", zero_division=0),
        "F1": f1_score(y_true, predictions, average="weighted", zero_division=0),
    }
    return {"metrics": metrics, "score": metrics[primary_metric]}


def train_and_evaluate(run_dir: str, df: pd.DataFrame, target: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    task_type = "classification" if str(plan["task_type"]).lower() == "classification" else "regression"
    input_columns = [column for column in df.columns if column not in {target, *plan.get("drop_columns", [])}]
    if not input_columns:
        raise ValueError("All features were removed by the planner. Please choose a different target.")

    X = df[input_columns].copy()
    y = df[target].copy()

    feature_schema = build_feature_schema(df, input_columns, plan.get("date_columns", []))
    categorical_options = {
        field["name"]: field.get("options", [])
        for field in feature_schema
        if field["kind"] in {"select", "boolean"}
    }

    engineered_sample = FeatureEngineeringTransformer(plan).fit_transform(X)
    numeric_columns = engineered_sample.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in engineered_sample.columns if column not in numeric_columns]
    scaling = str(plan["preprocessing"].get("scaling", "standard")).lower()
    preprocessor = build_preprocessor(numeric_columns, categorical_columns, scaling)

    stratify_target = y if task_type == "classification" and bool(plan["preprocessing"].get("stratify_split", True)) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_target)

    candidates = model_candidates(
        task_type,
        plan["preprocessing"].get("class_weight"),
        plan.get("candidate_models"),
    )
    primary_metric = plan["primary_metric"]
    maximize = plan["score_direction"] == "maximize"

    results: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None
    best_pipeline: Optional[Pipeline] = None
    best_predictions = None

    for model_name, estimator in candidates.items():
        log_event(run_dir, "TRAINER", f"Training {model_name}...")
        try:
            pipeline = Pipeline(
                [
                    ("feature_engineering", FeatureEngineeringTransformer(plan)),
                    ("preprocessor", clone(preprocessor)),
                    ("model", clone(estimator)),
                ]
            )
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            evaluation = evaluate_predictions(task_type, primary_metric, y_test, predictions)

            entry = {"model": model_name, "score": float(evaluation["score"])}
            entry.update({key: float(value) for key, value in evaluation["metrics"].items()})
            results.append(entry)

            is_better = best_result is None
            if best_result is not None:
                is_better = entry["score"] > best_result["score"] if maximize else entry["score"] < best_result["score"]
            if is_better:
                best_result = entry
                best_pipeline = pipeline
                best_predictions = predictions
        except Exception as exc:
            log_event(run_dir, "WARNING", f"{model_name} failed: {exc}")
            results.append({"model": model_name, "score": None, "error": str(exc)})

    if best_result is None or best_pipeline is None or best_predictions is None:
        raise RuntimeError("All candidate models failed to train.")

    importance_df = extract_importance(best_pipeline)
    save_performance_plot(run_dir, task_type, y_test, best_predictions, best_result["model"], best_result)
    save_importance_plot(run_dir, importance_df, best_result["model"])

    artifact = PredictionArtifact(
        pipeline=best_pipeline,
        feature_schema=feature_schema,
        target_column=target,
        task_type=task_type,
        plan=plan,
        best_model=best_result["model"],
        primary_metric=primary_metric,
    )
    joblib.dump(artifact, f"{run_dir}/model.pkl")

    return {
        "results": results,
        "best_model": best_result["model"],
        "best_metrics": {key: value for key, value in best_result.items() if key not in {"model", "error"}},
        "features": input_columns,
        "feature_schema": feature_schema,
        "categorical_options": categorical_options,
        "engineered_numeric_columns": numeric_columns,
        "engineered_categorical_columns": categorical_columns,
        "top_features": [] if importance_df is None else importance_df.to_dict(orient="records"),
    }
