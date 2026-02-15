import numpy as np
from scipy.stats import ks_2samp

def data_drift(reference: np.ndarray, current: np.ndarray, alpha=0.05):
    """
    Kolmogorov–Smirnov test for drift detection
    """
    drift_features = {}

    for i in range(reference.shape[1]):
        stat, p_value = ks_2samp(reference[:, i], current[:, i])
        drift_features[f"feature_{i}"] = {
            "p_value": float(p_value),
            "drift_detected": p_value < alpha
        }

    return drift_features


def prediction_drift(y_ref, y_curr, alpha=0.05):
    stat, p = ks_2samp(y_ref, y_curr)
    return {
        "p_value": float(p),
        "drift_detected": p < alpha
    }
