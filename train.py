import os
import json
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data.preprocessing import DataPreprocessor
from src.data.sequence_builder import build_sequences
from src.evaluation.metrics import evaluate
from src.models.tcn import TCN
from src.models.lstm import LSTMModel
from src.models.rnn import RNNModel
from src.models.trainer import train_model
from src.utils.io import save_model
from src.utils.seed import set_seed

CONFIG_PATH = "config/config.yaml"

def load_config(path: str) -> Dict:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)

def main() -> None:
    cfg = load_config(CONFIG_PATH)
    set_seed(cfg["training"]["seed"])

    os.makedirs(cfg["output"]["model_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["report_dir"], exist_ok=True)

    df = pd.read_csv(cfg["data"]["raw_path"])

    preprocessor = DataPreprocessor()
    df_proc = preprocessor.fit_transform(df)

    joblib.dump(preprocessor, os.path.join(cfg["output"]["model_dir"], "preprocessor.joblib"))

    X, y = build_sequences(df_proc, window=cfg["data"]["window_size"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["data"]["test_split"], random_state=cfg["training"]["seed"]
    )

    num_features = X.shape[2]

    models = {
        "tcn": TCN(
            num_features=num_features,
            channels=[64, 128, 256],
            kernel_size=cfg["models"]["tcn"]["kernel_size"],
        ),
        "lstm": LSTMModel(input_dim=num_features, hidden=cfg["models"]["lstm"]["hidden_size"]),
        "rnn": RNNModel(input_dim=num_features, hidden=cfg["models"]["rnn"]["hidden_size"]),
    }

    comparison: Dict[str, Dict] = {}

    for name, model in models.items():
        print(f"\nTraining {name.upper()}")
        trained_model, last_loss = train_model(
            model,
            X_train,
            y_train,
            epochs=cfg["training"]["epochs"],
            lr=cfg["training"]["learning_rate"],
            batch_size=cfg["training"]["batch_size"],
        )

        with torch.no_grad():
            preds = trained_model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()

        metrics = evaluate(y_test, preds)
        metrics["train_final_mse"] = last_loss
        comparison[name.upper()] = metrics

        model_path = os.path.join(cfg["output"]["model_dir"], f"{name}.pt")
        meta = {
            "model_name": name,
            "metrics": metrics,
        }
        save_model(trained_model, model_path, metadata=meta)

        print(f"Saved {name.upper()} model to {model_path}")

    comparison_path = os.path.join(cfg["output"]["report_dir"], "model_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=4)

    print(f"\n Model comparison report saved to {comparison_path}")


if __name__ == "__main__":
    main()
