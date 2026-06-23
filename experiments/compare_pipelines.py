import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"


def load_metrics(name):
    path = RESULTS_DIR / name / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    metrics = [
        item
        for item in [
            load_metrics("plain"),
            load_metrics("resnet"),
        ]
        if item is not None
    ]

    if not metrics:
        raise SystemExit("No metrics.json files found. Train at least one pipeline first.")

    keys = sorted({key for row in metrics for key in row.keys()})
    output_path = RESULTS_DIR / "comparison.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
