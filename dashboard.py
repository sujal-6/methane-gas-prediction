import json
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data.preprocessing import DataPreprocessor
from src.data.sequence_builder import build_sequences
from src.evaluation.metrics import evaluate
from src.inference.predictor import MethanePredictor
from src.models.lstm import LSTMModel
from src.models.rnn import RNNModel
from src.models.tcn import TCN


MODEL_DIR = "models"
REPORT_DIR = "reports"
DATA_PATH = "data/raw/Methane Gas From Different Crops.csv"
WINDOW_SIZE = 14


@st.cache_resource
def load_preprocessor() -> DataPreprocessor:
    return joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))


@st.cache_resource
def load_base_models(num_features: int) -> Dict[str, MethanePredictor]:
    models = {}
    for name, ctor in {
        "TCN": lambda: TCN(num_features=num_features),
        "LSTM": lambda: LSTMModel(input_dim=num_features),
        "RNN": lambda: RNNModel(input_dim=num_features),
    }.items():
        path = os.path.join(MODEL_DIR, f"{name.lower()}.pt")
        if os.path.exists(path):
            base = ctor()
            state = torch.load(path, map_location="cpu")
            base.load_state_dict(state)
            models[name] = MethanePredictor(base, device="cpu")
    return models


def load_comparison_report() -> Dict:
    path = os.path.join(REPORT_DIR, "model_comparison.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(
        page_title="Methane Gas Prediction Dashboard",
        layout="wide",
    )

    st.title("Methane Gas Prediction From Crops")
    st.markdown(
        "Production-style benchmarking of **TCN**, **LSTM**, and **RNN** for daily methane prediction."
    )

    # Load data and models
    df_raw = pd.read_csv(DATA_PATH)
    preprocessor = load_preprocessor()
    df_proc = preprocessor.transform(df_raw)
    X, y = build_sequences(df_proc, window=WINDOW_SIZE)
    models = load_base_models(num_features=X.shape[2])

    tabs = st.tabs(["Overview", "Per-run Analysis", "Interactive Prediction"])

    # Overview tab
    with tabs[0]:
        st.subheader("Global Model Comparison")
        comparison = load_comparison_report()
        if comparison:
            df_metrics = (
                pd.DataFrame(comparison)
                .T.reset_index()
                .rename(columns={"index": "Model"})
            )
            st.dataframe(df_metrics, use_container_width=True)
        else:
            st.info("No comparison report found. Run `python train.py` first.")

        st.subheader("Predicted vs Actual (Sample)")
        model_name = st.selectbox(
            "Visualize predictions for model", list(models.keys()), index=0
        )
        predictor = models[model_name]

        with torch.no_grad():
            preds = []
            for i in range(len(X)):
                preds.append(predictor.predict(X[i]))
        preds = np.array(preds)

        sample_len = min(300, len(y))
        chart_df = pd.DataFrame(
            {
                "Actual": y[:sample_len],
                "Predicted": preds[:sample_len],
            }
        )
        st.line_chart(chart_df)

    # Per-run analysis
    with tabs[1]:
        st.subheader("Per-run Performance")
        runs = sorted(df_raw["Run_ID"].unique().tolist())
        run_id = st.selectbox("Select Run_ID", runs)

        run_df = df_proc[df_proc["Run_ID"] == run_id].sort_values("Day")
        X_run, y_run = build_sequences(run_df, window=WINDOW_SIZE)

        cols = st.columns(len(models) or 1)
        for i, (name, predictor) in enumerate(models.items()):
            with cols[i]:
                preds = np.array([predictor.predict(x) for x in X_run])
                m = evaluate(y_run, preds)
                st.markdown(f"**{name}**")
                st.json(m)

        st.subheader("Run-level Curves")
        model_name = st.selectbox(
            "Model for run-level curve", list(models.keys()), key="run_curve_model"
        )
        predictor = models[model_name]
        preds = np.array([predictor.predict(x) for x in X_run])
        chart_df = pd.DataFrame({"Actual": y_run, "Predicted": preds})
        st.line_chart(chart_df)

    # Interactive prediction
    with tabs[2]:
        st.subheader("What-if Scenario Prediction")
        st.markdown(
            "Upload a CSV with the same schema as the training data (excluding `Daily_Methane_m3`) "
            "or manually configure the last window of timesteps."
        )

        uploaded = st.file_uploader("Upload sequence CSV", type=["csv"])
        selected_model_name = st.selectbox(
            "Model for prediction", list(models.keys()), key="interactive_model"
        )

        if uploaded is not None:
            df_in = pd.read_csv(uploaded)
            missing_cols = [
                c
                for c in [
                    "Crop",
                    "Run_ID",
                    "Day",
                    "Temperature_C",
                    "pH",
                    "VFA_mgL",
                    "VS_percent",
                    "CN_Ratio",
                    "Lignin_percent",
                ]
                if c not in df_in.columns
            ]
            if missing_cols:
                st.error(f"Missing columns in uploaded CSV: {missing_cols}")
            else:
                df_in_proc = preprocessor.transform(df_in)
                seq = (
                    df_in_proc.sort_values("Day")
                    .iloc[-WINDOW_SIZE:][
                        [
                            "Crop",
                            "Temperature_C",
                            "pH",
                            "VFA_mgL",
                            "VS_percent",
                            "CN_Ratio",
                            "Lignin_percent",
                        ]
                    ]
                    .values
                )
                predictor = models[selected_model_name]
                pred = predictor.predict(seq)
                st.success(f"Predicted methane production: **{pred:.3f} m³**")
        else:
            st.info("Upload a CSV to run an interactive prediction.")


if __name__ == "__main__":
    main()

