import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true, y_pred, title="Predicted vs Actual Methane"):
    plt.figure(figsize=(8, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Methane (m³)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_error_distribution(y_true, y_pred):
    errors = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=30)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
