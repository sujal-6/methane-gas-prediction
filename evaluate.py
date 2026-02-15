import os
import torch
import pandas as pd
import numpy as np

from src.data.preprocessing import DataPreprocessor
from src.data.sequence_builder import build_sequences
from src.evaluation.metrics import evaluate
from src.evaluation.plots import plot_predictions
from src.evaluation.report import generate_report
from src.models.tcn import TCN

MODEL_PATH = "models/tcn.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Please run `python train.py` first."
    )

df = pd.read_csv("data/raw/Methane Gas From Different Crops.csv")

prep = DataPreprocessor()
df = prep.fit_transform(df)

X, y = build_sequences(df)

model = TCN(num_features=X.shape[2])
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

with torch.no_grad():
    preds = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()

metrics = evaluate(y, preds)

print("\nEvaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

plot_predictions(y[:200], preds[:200])
generate_report(metrics, model_name="TCN")
