from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


def train_model(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> Tuple[torch.nn.Module, float]:
    """
    Generic trainer for sequence regression models.

    Returns:
        (trained_model, final_epoch_loss)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    last_loss = 0.0
    for epoch in range(epochs):
        model.train()
        losses = []

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        last_loss = float(sum(losses) / len(losses))
        print(f"Epoch {epoch + 1}: MSE={last_loss:.4f}")

    return model, last_loss
