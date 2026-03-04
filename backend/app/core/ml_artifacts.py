import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="ISO-8859-1")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_state(run_dir: str, payload: Dict[str, Any]) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "state.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def coerce_bool(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return value


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, plan: Dict[str, Any]):
        self.plan = plan

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineeringTransformer":
        self.columns_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()

        for column in getattr(self, "columns_in_", []):
            if column not in frame.columns:
                frame[column] = np.nan

        frame = frame.reindex(columns=getattr(self, "columns_in_", list(frame.columns)), fill_value=np.nan)

        for column in self.plan.get("date_columns", []):
            if column not in frame.columns:
                continue
            parsed = pd.to_datetime(frame[column], errors="coerce")
            frame[f"{column}_year"] = parsed.dt.year
            frame[f"{column}_month"] = parsed.dt.month
            frame[f"{column}_day"] = parsed.dt.day
            frame[f"{column}_dayofweek"] = parsed.dt.dayofweek
            frame.drop(columns=[column], inplace=True)

        for column in self.plan.get("missing_indicator_columns", []):
            if column in frame.columns:
                frame[f"{column}_was_missing"] = frame[column].isna().astype(int)

        drop_columns = [column for column in self.plan.get("drop_columns", []) if column in frame.columns]
        if drop_columns:
            frame = frame.drop(columns=drop_columns, errors="ignore")

        bool_columns = frame.select_dtypes(include=["bool"]).columns.tolist()
        for column in bool_columns:
            frame[column] = frame[column].astype(int)

        return frame


class PredictionArtifact:
    def __init__(
        self,
        pipeline: Pipeline,
        feature_schema: List[Dict[str, Any]],
        target_column: str,
        task_type: str,
        plan: Dict[str, Any],
        best_model: str,
        primary_metric: str,
    ):
        self.pipeline = pipeline
        self.feature_schema = feature_schema
        self.feature_names = [item["name"] for item in feature_schema]
        self.target_column = target_column
        self.task_type = task_type
        self.plan = plan
        self.best_model = best_model
        self.primary_metric = primary_metric

    def predict(self, raw_inputs: Any) -> np.ndarray:
        if isinstance(raw_inputs, pd.DataFrame):
            frame = raw_inputs.copy()
        elif isinstance(raw_inputs, dict):
            frame = pd.DataFrame([raw_inputs])
        else:
            frame = pd.DataFrame(raw_inputs)

        for field in self.feature_schema:
            column = field["name"]
            if column not in frame.columns:
                frame[column] = np.nan
            if field["kind"] == "number":
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            elif field["kind"] == "boolean":
                frame[column] = frame[column].map(coerce_bool)
            else:
                frame[column] = frame[column].where(frame[column].notna(), None)

        ordered = frame.reindex(columns=self.feature_names, fill_value=np.nan)
        return self.pipeline.predict(ordered)
