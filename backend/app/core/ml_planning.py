from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _date_parse_rate(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    sample = non_null.astype(str).head(200)
    parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean())


def _numeric_summary(series: pd.Series) -> Dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {}
    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,
    }


def _looks_like_monotonic_counter(series: pd.Series) -> bool:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 20:
        return False
    if not np.allclose(clean, np.round(clean)):
        return False
    diffs = np.diff(clean.to_numpy())
    if len(diffs) == 0:
        return False
    positive_steps = np.isin(diffs, [0, 1]).mean()
    strictly_increasing = (diffs >= 0).mean()
    return bool(strictly_increasing >= 0.98 and positive_steps >= 0.9)


def _looks_like_postal_code(column: str, series: pd.Series) -> bool:
    name = column.strip().lower()
    if any(token in name for token in ["postal", "postcode", "zip"]):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    sample = non_null.astype(str).head(200).str.strip()
    if sample.empty:
        return False

    digit_like = sample.str.fullmatch(r"\d{4,10}(-\d{4})?").fillna(False)
    if digit_like.mean() < 0.9:
        return False

    unique_ratio = float(non_null.nunique(dropna=True) / max(len(non_null), 1))
    return bool(unique_ratio >= 0.1)


def _looks_like_name_field(column: str, series: pd.Series) -> bool:
    name = column.strip().lower()
    if any(token in name for token in ["name", "firstname", "first_name", "lastname", "last_name", "fullname", "full_name", "customername", "contact"]):
        return True

    if pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()
    if len(non_null) < 20:
        return False

    sample = non_null.astype(str).head(200).str.strip()
    if sample.empty:
        return False

    token_count = sample.str.split().str.len().mean()
    alpha_ratio = sample.str.contains(r"[A-Za-z]", regex=True).mean()
    digit_ratio = sample.str.contains(r"\d", regex=True).mean()
    unique_ratio = float(non_null.nunique(dropna=True) / max(len(non_null), 1))
    return bool(token_count >= 1.5 and alpha_ratio >= 0.8 and digit_ratio <= 0.2 and unique_ratio >= 0.4)


def _looks_like_free_text(column: str, series: pd.Series) -> bool:
    name = column.strip().lower()
    if any(token in name for token in ["address", "street", "description", "comment", "notes", "message", "text", "title", "email", "phone"]):
        return True

    if pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()
    if len(non_null) < 20:
        return False

    sample = non_null.astype(str).head(200).str.strip()
    if sample.empty:
        return False

    avg_length = sample.str.len().mean()
    whitespace_ratio = sample.str.contains(r"\s", regex=True).mean()
    unique_ratio = float(non_null.nunique(dropna=True) / max(len(non_null), 1))
    return bool((avg_length >= 12 and whitespace_ratio >= 0.4) or (unique_ratio >= 0.7 and avg_length >= 8))


def _looks_like_identifier(column: str, series: pd.Series) -> bool:
    name = column.strip().lower()
    if any(token in name for token in ["id", "uuid", "guid", "serial", "record", "row", "index"]):
        return True

    non_null = series.dropna()
    if len(non_null) < 20:
        return False

    unique_ratio = float(non_null.nunique(dropna=True) / max(len(non_null), 1))
    if unique_ratio < 0.95:
        return False

    if _looks_like_monotonic_counter(series):
        return True

    if pd.api.types.is_numeric_dtype(series):
        return True

    sample = non_null.astype(str).head(200)
    has_spaces = sample.str.contains(r"\s", regex=True).mean() if not sample.empty else 0.0
    avg_length = sample.str.len().mean() if not sample.empty else 0.0
    return bool(has_spaces <= 0.1 and avg_length <= 24)


def build_dataset_snapshot(df: pd.DataFrame, target: str, requested_metric: str, supported_models: Dict[str, List[str]]) -> Dict[str, Any]:
    feature_columns = [column for column in df.columns if column != target]
    target_series = df[target]

    target_snapshot: Dict[str, Any] = {
        "name": target,
        "dtype": str(target_series.dtype),
        "missing_ratio": float(target_series.isna().mean()),
        "unique_count": int(target_series.nunique(dropna=False)),
        "sample_values": [str(value) for value in target_series.dropna().astype(str).head(8).tolist()],
    }
    if pd.api.types.is_numeric_dtype(target_series):
        target_snapshot["numeric_summary"] = _numeric_summary(target_series)
    else:
        target_snapshot["top_values"] = (
            target_series.astype(str).value_counts(dropna=False).head(10).to_dict()
        )

    features: List[Dict[str, Any]] = []
    for column in feature_columns:
        series = df[column]
        unique_ratio = float(series.nunique(dropna=True) / max(series.notna().sum(), 1))
        looks_like_counter = _looks_like_monotonic_counter(series)
        looks_like_postal = _looks_like_postal_code(column, series)
        looks_like_name = _looks_like_name_field(column, series)
        looks_like_text = _looks_like_free_text(column, series)
        looks_like_id = _looks_like_identifier(column, series)

        snapshot: Dict[str, Any] = {
            "name": column,
            "dtype": str(series.dtype),
            "missing_ratio": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=False)),
            "unique_ratio": unique_ratio,
            "sample_values": [str(value) for value in series.dropna().astype(str).head(5).tolist()],
            "date_parse_rate": _date_parse_rate(series) if series.dtype == "object" else 0.0,
            "risk_flags": {
                "looks_like_row_index": looks_like_counter or column.strip().lower() in {"row", "rownumber", "row_number", "index", "unnamed: 0"},
                "looks_like_identifier": looks_like_id,
                "looks_like_postal_code": looks_like_postal,
                "looks_like_name": looks_like_name,
                "looks_like_free_text": looks_like_text,
                "high_cardinality": bool(unique_ratio >= 0.5 and series.nunique(dropna=True) >= 25),
            },
        }
        if pd.api.types.is_numeric_dtype(series):
            snapshot["numeric_summary"] = _numeric_summary(series)
        features.append(snapshot)

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "requested_metric": requested_metric,
        "target": target_snapshot,
        "features": features,
        "supported_models": supported_models,
    }


def build_feature_schema(df: pd.DataFrame, input_columns: List[str], date_columns: List[str]) -> List[Dict[str, Any]]:
    schema: List[Dict[str, Any]] = []

    for column in input_columns:
        series = df[column]
        entry: Dict[str, Any] = {
            "name": column,
            "nullable": bool(series.isna().any()),
            "sample_values": [str(value) for value in series.dropna().astype(str).head(3).tolist()],
        }

        if column in date_columns:
            example = series.dropna().astype(str).iloc[0] if series.dropna().shape[0] else "2024-01-31"
            entry.update(
                {
                    "kind": "date",
                    "dtype": "date",
                    "default": str(example),
                    "help": "Date value used for calendar feature extraction.",
                }
            )
        elif pd.api.types.is_bool_dtype(series):
            entry.update(
                {
                    "kind": "boolean",
                    "dtype": "bool",
                    "default": bool(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else False,
                    "options": [True, False],
                    "help": "Boolean input expected by the trained pipeline.",
                }
            )
        elif pd.api.types.is_numeric_dtype(series):
            median_value = pd.to_numeric(series, errors="coerce").median()
            entry.update(
                {
                    "kind": "number",
                    "dtype": "number",
                    "default": 0.0 if pd.isna(median_value) else float(median_value),
                    "help": "Numeric value; median from training data used as default.",
                }
            )
        else:
            categories = series.dropna().astype(str).unique().tolist()
            default_value = str(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else ""
            if len(categories) <= 40:
                entry.update(
                    {
                        "kind": "select",
                        "dtype": "category",
                        "default": default_value,
                        "options": sorted(str(item) for item in categories),
                        "help": "Choose one of the categories observed during training.",
                    }
                )
            else:
                entry.update(
                    {
                        "kind": "text",
                        "dtype": "text",
                        "default": default_value,
                        "help": "Free-text categorical value. Unseen values are handled safely.",
                    }
                )

        schema.append(entry)

    return schema
