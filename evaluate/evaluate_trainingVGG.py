import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import prepare_plain_dataframe, split_dataframe
from utils.vgg.data_generator import ClothesDataGenerator


DEFAULT_MODEL_PATH = ROOT / "weights" / "vgg" / "vgg_model.h5"
DEFAULT_LABELS_PATH = ROOT / "weights" / "vgg" / "label_map.json"
DEFAULT_SPLITS_DIR = ROOT / "weights" / "vgg" / "splits"
DEFAULT_OUTPUT_DIR = ROOT / "evaluate" / "results" / "vgg"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained VGG classifier on the test set.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--csv-path", default=str(ROOT / "styles.csv"))
    parser.add_argument("--image-dir", default=str(ROOT / "data" / "images"))
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-per-class", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-warmup", action="store_true", help="Include first-call TensorFlow overhead in timing.")
    return parser.parse_args()


def load_idx_to_label(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(idx): label for idx, label in raw.items()}


def load_test_dataframe(args):
    test_split = Path(args.splits_dir) / "test.csv"
    if test_split.exists():
        return pd.read_csv(test_split)

    df = prepare_plain_dataframe(
        args.csv_path,
        args.image_dir,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    _, _, test_df = split_dataframe(df, "label_name", seed=args.seed)
    return test_df


def save_confusion_matrix(path, matrix, class_names):
    cm_df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    cm_df.index.name = "actual"
    cm_df.columns.name = "predicted"
    cm_df.to_csv(path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model)
    idx_to_label = load_idx_to_label(args.labels)
    class_names = [idx_to_label[idx] for idx in sorted(idx_to_label)]
    label_map = {label: idx for idx, label in idx_to_label.items()}

    test_df = load_test_dataframe(args)
    test_df = test_df[test_df["label_name"].isin(label_map)].reset_index(drop=True)
    test_gen = ClothesDataGenerator(
        test_df,
        args.image_dir,
        label_map,
        batch_size=args.batch_size,
        shuffle=False,
    )

    if not args.no_warmup and len(test_gen) > 0:
        warmup_x, _ = test_gen[0]
        model.predict(warmup_x[:1], verbose=0)

    y_true = test_df["label_name"].map(label_map).to_numpy(dtype=np.int64)

    start = time.perf_counter()
    probabilities = model.predict(test_gen, verbose=1)
    inference_time = time.perf_counter() - start

    y_pred = np.argmax(probabilities, axis=1)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "model_path": str(args.model),
        "test_size": int(len(test_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "inference_time_seconds": float(inference_time),
        "inference_time_per_image_seconds": float(inference_time / len(test_df)) if len(test_df) else 0.0,
        "params": int(model.count_params()),
        "classes": class_names,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    save_confusion_matrix(output_dir / "confusion_matrix.csv", cm, class_names)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))
    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))
    print(f"\nSaved evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()