# Methane Gas Prediction From Crops

End-to-end machine learning system to predict **daily methane production (m³)** from crop-based anaerobic digestion using time-series deep learning. The pipeline uses **Temporal Convolutional Networks (TCN)** as the primary model and benchmarks performance against **LSTM** and **vanilla RNN**.

---

## Project Overview

- **Goal:** Accurately predict daily methane output from crop digestion using multivariate time-series data.
- **Models:** TCN (primary), LSTM, RNN — trained and compared with a unified framework.
- **Outputs:** Trained models, evaluation metrics (RMSE, MAE, MAPE, R²), comparison reports, REST API, and Streamlit dashboard.

---

## Dataset

| Type | Description |
|------|-------------|
| **Location** | `data/raw/Methane Gas From Different Crops.csv` |
| **Type** | Multivariate time-series, multiple runs/batches |

**Input features**

- `Crop` (categorical)
- `Run_ID` (batch/experiment identifier)
- `Day` (time index)
- `Temperature_C`, `pH`, `VFA_mgL`, `VS_percent`, `CN_Ratio`, `Lignin_percent`

**Target**

- `Daily_Methane_m3`

---

## Project Structure

```
methane_prediction/
├── api/
│   └── app.py              # FastAPI: /health, /predict
├── config/
│   └── config.yaml         # Data, training, model config
├── data/
│   └── raw/                # Raw CSV dataset
├── models/                 # Saved .pt models + preprocessor.joblib
├── reports/                # model_comparison.json, per-model reports
├── src/
│   ├── data/               # preprocessing, sequence_builder, schema, drift
│   ├── evaluation/         # metrics, plots, report
│   ├── inference/          # predictor
│   ├── models/             # tcn, lstm, rnn, trainer
│   └── utils/              # io, logger, seed
├── dashboard.py            # Streamlit dashboard
├── train.py                # Train TCN, LSTM, RNN; save models & comparison
├── evaluate.py             # Evaluate a single model (e.g. TCN)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone or navigate to project
cd methane_prediction

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Optional: PyYAML, joblib, streamlit if not already installed
pip install pyyaml joblib streamlit
```

---

## Usage

### 1. Train models and generate reports

Uses `config/config.yaml` for paths, window size, train/test split, and hyperparameters.

```bash
python train.py
```

- Fits `DataPreprocessor`, saves to `models/preprocessor.joblib`
- Builds sequences (default window = 14), splits train/test
- Trains TCN, LSTM, RNN; evaluates each (RMSE, MAE, MAPE, R²)
- Saves models to `models/*.pt` and metadata to `models/*_meta.json`
- Writes `reports/model_comparison.json`

### 2. Run the REST API

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

- **GET /health** — Health check (`{"status": "ok"}`)
- **POST /predict** — Prediction with model selection and sequence input (see API section below)

### 3. Run the Streamlit dashboard

```bash
streamlit run dashboard.py
```

- **Overview:** Global model comparison table, predicted vs actual curves
- **Per-run Analysis:** Select `Run_ID`, view per-model metrics and curves
- **Interactive Prediction:** Upload a CSV (same schema as training) and get methane prediction for chosen model

### 4. Evaluate a single model (e.g. TCN)

```bash
python evaluate.py
```

Uses `models/tcn.pt` by default; prints metrics and generates report/plots.

---

## Configuration

Edit `config/config.yaml`:

| Section | Key | Description |
|---------|-----|-------------|
| `data` | `raw_path` | Path to raw CSV |
| `data` | `window_size` | Sequence length (default 14) |
| `data` | `test_split` | Test fraction (default 0.2) |
| `training` | `batch_size`, `epochs`, `learning_rate`, `seed` | Training settings |
| `models` | `tcn` / `lstm` / `rnn` | Model hyperparameters (channels, kernel_size, hidden_size) |
| `output` | `model_dir`, `report_dir` | Where to save models and reports |

---

## API: `/predict`

**Request (JSON)**

```json
{
  "model": "tcn",
  "window_size": 14,
  "sequence": [
    {
      "Crop": "Corn",
      "Run_ID": 1,
      "Day": 1,
      "Temperature_C": 35.0,
      "pH": 7.2,
      "VFA_mgL": 1200,
      "VS_percent": 8.5,
      "CN_Ratio": 25,
      "Lignin_percent": 12
    }
  ]
}
```

- `model`: `"tcn"` \| `"lstm"` \| `"rnn"`
- `window_size`: Number of timesteps (must match training; default 14)
- `sequence`: List of timesteps; **at least `window_size`** rows. The last `window_size` rows (by `Day`) are used for prediction.

**Response**

```json
{
  "model": "tcn",
  "predicted_methane_m3": 1.234
}
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root mean squared error |
| **MAE** | Mean absolute error |
| **MAPE** | Mean absolute percentage error (%) |
| **R²** | Coefficient of determination |

---

## Reproducibility

- Random seed set in `config.yaml` (`training.seed`) and applied in `train.py` via `set_seed()`.
- Dependencies are listed in `requirements.txt`; pin versions for exact reproducibility.
- Preprocessor and model artifacts are saved under `models/` and loaded by the API and dashboard.

---