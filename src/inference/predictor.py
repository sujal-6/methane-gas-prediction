import torch
import numpy as np

class MethanePredictor:
    def __init__(self, model, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, sequence: np.ndarray):
        # sequence: shape (T, F)
        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            x = x.to(self.device)
            prediction = self.model(x)
        return float(prediction.cpu().numpy().squeeze())
