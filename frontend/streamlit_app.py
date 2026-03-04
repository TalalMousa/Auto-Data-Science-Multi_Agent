import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
MAX_SAFE_ROWS = 50000

st.set_page_config(page_title="Auto Data Science Agent", layout="wide")
st.title("Auto Data Science Agent")
st.caption("Upload a CSV, train multiple models through the LangGraph agent workflow, and test the winning model.")


def _request_json(method: str, url: str, **kwargs):
    response = requests.request(method, url, timeout=120, **kwargs)
    response.raise_for_status()
    return response.json()


def _render_prediction_input(field: dict, key_prefix: str):
    key = f"{key_prefix}_{field['name']}"
    label = field["name"]
    help_text = field.get("help")
    default = field.get("default")
    kind = field.get("kind")

    if kind == "number":
        return st.number_input(label, value=float(default or 0.0), help=help_text, key=key)
    if kind in {"select", "boolean"}:
        options = field.get("options", [])
        if not options:
            return st.text_input(label, value="" if default is None else str(default), help=help_text, key=key)
        default_index = options.index(default) if default in options else 0
        return st.selectbox(label, options=options, index=default_index, help=help_text, key=key)
    return st.text_input(label, value="" if default is None else str(default), help=help_text, key=key)


def _sort_results(frame: pd.DataFrame, meta: dict) -> pd.DataFrame:
    ascending = meta.get("score_direction") == "minimize"
    if "score" not in frame.columns:
        return frame
    return frame.sort_values(by="score", ascending=ascending, na_position="last")


def _show_upload_panel():
    st.sidebar.header("Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if not uploaded_file:
        return

    try:
        uploaded_file.seek(0)
        preview = pd.read_csv(uploaded_file, nrows=5)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        preview = pd.read_csv(uploaded_file, encoding="ISO-8859-1", nrows=5)

    uploaded_file.seek(0)
    total_rows = max(sum(1 for _ in uploaded_file) - 1, 0)
    uploaded_file.seek(0)

    st.sidebar.info(f"Rows detected: {total_rows:,}")
    sample_size = total_rows
    if total_rows > MAX_SAFE_ROWS:
        st.sidebar.warning("Large dataset detected. Sampling keeps training responsive and reduces memory pressure.")
        sample_size = st.sidebar.slider("Sample size", 1000, MAX_SAFE_ROWS, min(20000, MAX_SAFE_ROWS))

    target_column = st.sidebar.selectbox("Target column", preview.columns)
    if not st.sidebar.button("Launch agent workflow", use_container_width=True):
        return

    with st.status("Preparing dataset for the backend agent workflow..."):
        try:
            try:
                uploaded_file.seek(0)
                data = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                data = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

            if total_rows > sample_size:
                data = data.sample(n=sample_size, random_state=42)
                st.toast(f"Training sample reduced to {sample_size:,} rows.")

            temp_filename = f"sampled_{uploaded_file.name}"
            data.to_csv(temp_filename, index=False)
            try:
                with open(temp_filename, "rb") as handle:
                    upload_response = _request_json(
                        "POST",
                        f"{API_URL}/upload",
                        files={"file": (temp_filename, handle, "text/csv")},
                    )
            finally:
                os.remove(temp_filename)

            metric = "rmse" if pd.api.types.is_numeric_dtype(data[target_column]) else "f1"
            run_response = _request_json(
                "POST",
                f"{API_URL}/runs",
                json={
                    "filename": upload_response["filename"],
                    "target": target_column,
                    "metric": metric,
                },
            )
        except requests.RequestException as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Backend request failed: {detail}")
            return

        st.session_state["run_id"] = run_response["run_id"]
        st.session_state["status"] = "RUNNING"
        st.session_state.pop("final", None)
        st.rerun()


def _show_running_state(run_id: str):
    status_col, log_col = st.columns([1, 1])

    with status_col:
        st.subheader("Run status")
        state = _request_json("GET", f"{API_URL}/runs/{run_id}")
        st.info(f"Status: {state.get('status', 'RUNNING')}")
        st.code(state.get("meta", {}).get("summary", "Agent workflow in progress."), language="text")

    with log_col:
        st.subheader("Agent logs")
        events = _request_json("GET", f"{API_URL}/runs/{run_id}/events").get("events", [])
        log_lines = [f"{item.get('timestamp', '')} [{item.get('level', 'INFO')}] {item.get('message', '')}" for item in events]
        st.code("\n".join(log_lines) or "Waiting for the first event...", language="text")

    if state.get("status") in {"COMPLETED", "FAILED"}:
        st.session_state["status"] = state["status"]
        st.session_state["final"] = state
        st.rerun()

    time.sleep(2)
    st.rerun()


def _show_results(run_id: str, meta: dict):
    st.success(meta.get("summary", "Training complete."))
    overview_tab, visuals_tab, predict_tab, download_tab, reasoning_tab = st.tabs(
        ["Overview", "Visuals", "Predict", "Download", "Reasoning"]
    )

    with overview_tab:
        results = meta.get("results", [])
        if not results:
            st.warning("No model results were returned.")
        else:
            results_df = _sort_results(pd.DataFrame(results), meta)
            winner = results_df.iloc[0]
            st.subheader(f"Best model: {winner['model']}")

            metric_columns = [column for column in results_df.columns if column not in {"model", "score", "error"}]
            metric_boxes = st.columns(max(len(metric_columns), 1))
            if metric_columns:
                for index, column in enumerate(metric_columns):
                    value = winner[column]
                    metric_boxes[index].metric(column, f"{value:.4f}" if pd.notna(value) else "n/a")
            else:
                metric_boxes[0].metric(meta.get("primary_metric", "Score"), f"{winner['score']:.4f}")

            st.dataframe(results_df, use_container_width=True)

    with visuals_tab:
        plot_col, importance_col = st.columns(2)

        plot_response = requests.get(f"{API_URL}/runs/{run_id}/plot", timeout=120)
        if plot_response.ok:
            plot_col.image(plot_response.content, caption="Performance plot", use_container_width=True)
        else:
            plot_col.warning("Performance plot is not available for this run.")

        importance_response = requests.get(f"{API_URL}/runs/{run_id}/importance", timeout=120)
        if importance_response.ok:
            importance_col.image(importance_response.content, caption="Feature importance", use_container_width=True)
        else:
            importance_col.info("Feature importance is not available for the winning model.")

        top_features = meta.get("top_features", [])
        if top_features:
            st.subheader("Top ranked features")
            st.dataframe(pd.DataFrame(top_features), use_container_width=True)

    with predict_tab:
        schema = meta.get("feature_schema", [])
        if not schema:
            st.warning("This run does not include prediction input metadata.")
        else:
            st.subheader("Test the saved model")
            with st.form("predict_form"):
                columns = st.columns(2)
                inputs = {}
                for index, field in enumerate(schema):
                    with columns[index % 2]:
                        inputs[field["name"]] = _render_prediction_input(field, run_id)

                if st.form_submit_button("Predict", use_container_width=True):
                    try:
                        prediction = _request_json(
                            "POST",
                            f"{API_URL}/predict",
                            json={"run_id": run_id, "inputs": inputs},
                        )["prediction"]
                        st.success(f"Prediction: {prediction}")
                    except requests.RequestException as exc:
                        detail = str(exc)
                        if exc.response is not None:
                            try:
                                detail = exc.response.json().get("detail", exc.response.text)
                            except ValueError:
                                detail = exc.response.text
                        st.error(f"Prediction failed: {detail}")

    with download_tab:
        model_response = requests.get(f"{API_URL}/runs/{run_id}/model", timeout=120)
        if model_response.ok:
            st.download_button(
                label="Download trained model",
                data=model_response.content,
                file_name=f"model_{run_id}.pkl",
                mime="application/octet-stream",
            )
        else:
            st.warning("The model artifact is not available.")

        profile = meta.get("profile", {})
        if profile:
            st.subheader("Dataset profile")
            profile_cols = st.columns(4)
            profile_cols[0].metric("Rows", profile.get("rows", 0))
            profile_cols[1].metric("Columns", profile.get("columns", 0))
            profile_cols[2].metric("Task type", profile.get("task_type", "n/a"))
            profile_cols[3].metric("Primary metric", meta.get("primary_metric", "n/a"))

    with reasoning_tab:
        plan = meta.get("plan", {})
        analysis = meta.get("analysis", {})
        report = meta.get("report", {})
        if not plan:
            st.warning("No planner output is available.")
        else:
            if analysis:
                st.subheader("Analyzer assessment")
                st.write(analysis.get("analysis_summary", ""))
                for line in analysis.get("data_risks", []):
                    st.write(f"- {line}")

            st.subheader("Feature engineering decisions")
            for line in plan.get("feature_engineering_reasoning", []):
                st.write(f"- {line}")

            st.subheader("Preprocessing decisions")
            for line in plan.get("preprocessing", {}).get("reasoning", []):
                st.write(f"- {line}")

            st.subheader("Modeling strategy")
            for line in plan.get("modeling_reasoning", []):
                st.write(f"- {line}")

            info_col_1, info_col_2 = st.columns(2)
            with info_col_1:
                st.write("Dropped columns")
                st.code("\n".join(plan.get("drop_columns", [])) or "None", language="text")
                st.write("Date engineered columns")
                st.code("\n".join(plan.get("date_columns", [])) or "None", language="text")
            with info_col_2:
                st.write("Missing indicators")
                st.code("\n".join(plan.get("missing_indicator_columns", [])) or "None", language="text")
                st.write("Candidate models")
                st.code("\n".join(plan.get("candidate_models", [])) or "None", language="text")

            if plan.get("trainer_execution_reasoning"):
                st.subheader("Trainer execution decisions")
                for line in plan.get("trainer_execution_reasoning", []):
                    st.write(f"- {line}")

            if report:
                st.subheader("Reporter summary")
                st.write(report.get("model_summary", ""))
                for line in report.get("reasoning_summary", []):
                    st.write(f"- {line}")
                if report.get("caution"):
                    st.caption(f"Caution: {report['caution']}")


_show_upload_panel()

run_id = st.session_state.get("run_id")
if run_id and st.session_state.get("status") == "RUNNING":
    _show_running_state(run_id)

if st.session_state.get("status") == "COMPLETED":
    final_state = st.session_state.get("final", {})
    _show_results(run_id, final_state.get("meta", {}))
elif st.session_state.get("status") == "FAILED":
    final_state = st.session_state.get("final", {})
    st.error(final_state.get("error", "Training failed."))
