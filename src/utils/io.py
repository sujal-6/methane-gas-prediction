import torch
import json
import os

def save_model(model, path: str, metadata: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

    if metadata:
        meta_path = path.replace(".pt", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

def load_model(model, path: str, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
