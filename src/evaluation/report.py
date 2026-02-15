import json
import os
from datetime import datetime

def generate_report(metrics: dict, model_name: str, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "model": model_name,
        "metrics": metrics,
        "generated_at": datetime.utcnow().isoformat()
    }

    path = os.path.join(output_dir, f"{model_name}_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=4)

    return path
