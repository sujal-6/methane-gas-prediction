from enum import Enum
from typing import List, Literal

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.preprocessing import DataPreprocessor
from src.inference.predictor import MethanePredictor
from src.models.lstm import LSTMModel
from src.models.rnn import RNNModel
from src.models.tcn import TCN


class ModelName(str, Enum):
    tcn = "tcn"
    lstm = "lstm"
    rnn = "rnn"


class Timestep(BaseModel):
    Crop: str
    Run_ID: int
    Day: int
    Temperature_C: float
    pH: float
    VFA_mgL: float
    VS_percent: float
    CN_Ratio: float
    Lignin_percent: float


class PredictRequest(BaseModel):
    model: ModelName = Field(default=ModelName.tcn, description="Model to use for prediction")
    window_size: int = Field(default=14, description="Sequence length used during training")
    sequence: List[Timestep]


class PredictResponse(BaseModel):
    model: ModelName
    predicted_methane_m3: float


app = FastAPI(title="Methane Prediction API", version="1.0.0")


def _load_preprocessor(model_dir: str = "models") -> DataPreprocessor:
    try:
        return joblib.load(f"{model_dir}/preprocessor.joblib")
    except Exception as exc:
        raise RuntimeError("Preprocessor not found. Please run training first.") from exc


def _load_model(model_name: ModelName, num_features: int, model_dir: str = "models") -> MethanePredictor:
    path = f"{model_dir}/{model_name.value}.pt"
    if not torch.cuda.is_available():
        map_location = "cpu"
    else:
        map_location = None

    if model_name == ModelName.tcn:
        base = TCN(num_features=num_features)
    elif model_name == ModelName.lstm:
        base = LSTMModel(input_dim=num_features)
    else:
        base = RNNModel(input_dim=num_features)

    state = torch.load(path, map_location=map_location)
    base.load_state_dict(state)
    return MethanePredictor(base)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if len(payload.sequence) < payload.window_size:
        raise HTTPException(
            status_code=422,
            detail=f"At least window_size={payload.window_size} timesteps are required",
        )

    df = pd.DataFrame([s.dict() for s in payload.sequence])

    preprocessor = _load_preprocessor()
    df_proc = preprocessor.transform(df)

    feature_cols = [
        "Crop",
        "Temperature_C",
        "pH",
        "VFA_mgL",
        "VS_percent",
        "CN_Ratio",
        "Lignin_percent",
    ]

    seq = df_proc.sort_values("Day").iloc[-payload.window_size :][feature_cols].values
    num_features = seq.shape[1]

    predictor = _load_model(payload.model, num_features=num_features)
    pred = predictor.predict(seq)

    return PredictResponse(model=payload.model, predicted_methane_m3=float(pred))
