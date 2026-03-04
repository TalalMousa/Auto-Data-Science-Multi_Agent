import json
import os
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI


ANALYZER_PROMPT = """
You are the Analyzer agent inside an autonomous machine learning system.
Your role is to read a dataset snapshot produced by deterministic tools and infer the correct ML framing.

Rules:
- Use only the supplied dataset snapshot. Do not invent columns, values, or evidence.
- Infer task_type as either "classification" or "regression".
- Prefer "classification" when the target behaves like labels, categories, booleans, discrete codes, or a small finite set even if the raw dtype is numeric.
- Prefer "regression" when the target is a continuous numeric quantity with many distinct values.
- Recommend the best default metric for the inferred task_type.
- Call out concrete data risks that could affect modeling quality.
- Return JSON only. No markdown.

JSON schema:
{
  "task_type": "classification or regression",
  "recommended_metric": "F1|Accuracy|Precision|Recall|RMSE|MAE|R2|MSE",
  "analysis_summary": "2-4 sentence technical assessment",
  "data_risks": ["risk 1", "risk 2"],
  "candidate_models": ["model_name", "model_name", "model_name", "model_name"]
}
"""


PLANNER_PROMPT = """
You are the Planner agent for an autonomous machine learning workflow.
You must design an execution plan that the training tool can follow exactly.

Available transformations:
- Drop weak or harmful columns using `drop_columns`.
- Mark date columns for automatic expansion into year, month, day, and weekday using `date_columns`.
- Mark columns that should get missing-value indicator flags using `missing_indicator_columns`.
- Choose a scaling strategy: "standard", "robust", or "minmax".
- For classification only, choose class weighting: "balanced" or "none".
- For classification only, choose whether to stratify the train/test split.
- Choose 4 to 6 candidate models from the supported list only.

Hard constraints:
- Never invent a column name.
- Only select date columns when the snapshot strongly suggests date-like values.
- If a feature has `risk_flags.looks_like_row_index`, `risk_flags.looks_like_identifier`, or `risk_flags.looks_like_postal_code`, treat it as unsafe by default and place it in `drop_columns` unless there is overwhelming evidence it is genuinely predictive.
- If a feature has `risk_flags.looks_like_name` or `risk_flags.looks_like_free_text`, drop it by default. The execution tool only supports one-hot encoding, so raw names, addresses, notes, and free-text columns are usually harmful noise.
- For regression, row counters, record IDs, line numbers, postal codes, and ZIP-like codes must be dropped because the execution tool does not support semantic encoding for them.
- High-cardinality code-like or text-like columns should usually be dropped rather than one-hot encoded.
- `candidate_models` must be chosen only from the supplied supported list.
- Keep the plan compact and executable.
- Return JSON only. No markdown.

JSON schema:
{
  "task_type": "classification or regression",
  "date_columns": ["col"],
  "drop_columns": ["col"],
  "missing_indicator_columns": ["col"],
  "preprocessing": {
    "scaling": "standard|robust|minmax",
    "class_weight": "balanced|none",
    "stratify_split": true,
    "reasoning": ["decision 1", "decision 2"]
  },
  "feature_engineering_reasoning": ["decision 1", "decision 2"],
  "modeling_reasoning": ["decision 1", "decision 2"],
  "candidate_models": ["model_name", "model_name", "model_name", "model_name"]
}
"""


TRAINER_PROMPT = """
You are the Trainer agent coordinator.
You do not train models directly. You validate the analyzer and planner decisions before the deterministic training tool runs.

Rules:
- Keep the plan executable by the downstream tool.
- You may refine `candidate_models`, `primary_metric`, and `score_direction`.
- Keep 4 to 6 candidate models only, chosen from the supported list.
- Keep `primary_metric` valid for the selected task_type.
- Use "maximize" for F1, Accuracy, Precision, Recall, and R2.
- Use "minimize" for RMSE, MAE, and MSE.
- Return JSON only. No markdown.

JSON schema:
{
  "task_type": "classification or regression",
  "primary_metric": "F1|Accuracy|Precision|Recall|RMSE|MAE|R2|MSE",
  "score_direction": "maximize|minimize",
  "candidate_models": ["model_name", "model_name", "model_name", "model_name"],
  "execution_reasoning": ["decision 1", "decision 2"]
}
"""


REPORTER_PROMPT = """
You are the Reporter agent for an autonomous machine learning workflow.
Summarize the completed run for a technical end user.

Rules:
- Base every statement on the provided analysis, plan, and training results.
- Explain why the selected model won relative to the chosen metric.
- Mention the key preprocessing or feature engineering choices that mattered.
- Mention one practical caution or limitation.
- Return JSON only. No markdown.

JSON schema:
{
  "summary": "1-2 sentence concise summary for the run header",
  "model_summary": "Why the best model won",
  "reasoning_summary": ["important point 1", "important point 2"],
  "caution": "One realistic limitation or warning"
}
"""


def _model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def build_llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for LLM-based agents.")
    return ChatOpenAI(model=_model_name(), temperature=0.1)


def _extract_json(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM response did not contain a JSON object: {raw_text}")
    return json.loads(text[start : end + 1])


def _invoke_json(prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    llm = build_llm()
    message = (
        f"{prompt.strip()}\n\n"
        f"Input payload:\n{json.dumps(payload, indent=2, ensure_ascii=True)}\n\n"
        "Return only JSON."
    )
    response = llm.invoke(message)
    content = response.content if isinstance(response.content, str) else str(response.content)
    return _extract_json(content)


def analyzer_agent_llm(dataset_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return _invoke_json(ANALYZER_PROMPT, dataset_snapshot)


def planner_agent_llm(dataset_snapshot: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    return _invoke_json(
        PLANNER_PROMPT,
        {
            "dataset_snapshot": dataset_snapshot,
            "analysis": analysis,
        },
    )


def trainer_agent_llm(dataset_snapshot: Dict[str, Any], analysis: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    return _invoke_json(
        TRAINER_PROMPT,
        {
            "dataset_snapshot": dataset_snapshot,
            "analysis": analysis,
            "plan": plan,
        },
    )


def reporter_agent_llm(analysis: Dict[str, Any], plan: Dict[str, Any], training_summary: Dict[str, Any]) -> Dict[str, Any]:
    compact_results: List[Dict[str, Any]] = []
    for item in training_summary.get("results", []):
        compact_results.append({key: value for key, value in item.items() if key != "error"})
    return _invoke_json(
        REPORTER_PROMPT,
        {
            "analysis": analysis,
            "plan": plan,
            "training_summary": {
                "best_model": training_summary.get("best_model"),
                "best_metrics": training_summary.get("best_metrics"),
                "results": compact_results,
                "top_features": training_summary.get("top_features", []),
            },
        },
    )
