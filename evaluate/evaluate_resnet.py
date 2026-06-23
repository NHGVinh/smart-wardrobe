import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import DEFAULT_MAX_PER_CLASS, prepare_resnet_dataframe, split_dataframe
from src.models.resnet import MultiTaskResNet
from src.preprocessing.resnetPP import data_transforms, load_resnet_image


DEFAULT_CHECKPOINT_PATH = ROOT / "weights" / "resnet" / "resnet_model.pth"
DEFAULT_LABELS_PATH = ROOT / "weights" / "resnet" / "labels_map.pth"
DEFAULT_SPLITS_DIR = ROOT / "weights" / "resnet" / "splits"
DEFAULT_OUTPUT_DIR = ROOT / "evaluate" / "results" / "resnet"


class ResNetEvalDataset(Dataset):
    def __init__(self, dataframe, image_dir, cat_to_idx, style_to_idx):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.cat_to_idx = cat_to_idx
        self.style_to_idx = style_to_idx

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = self.image_dir / f"{row['id']}.jpg"
        image = data_transforms["val"](load_resnet_image(image_path))
        category = self.cat_to_idx[row["articleType"]]
        style = self.style_to_idx[row["usage"]]
        return image, category, style


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained ResNet multitask classifier.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--csv-path", default=str(ROOT / "styles.csv"))
    parser.add_argument("--image-dir", default=str(ROOT / "data" / "images"))
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-warmup", action="store_true", help="Include first-call PyTorch overhead in timing.")
    return parser.parse_args()


def load_labels(path, device):
    labels = torch.load(path, weights_only=False, map_location=device)
    idx_to_cat = {int(idx): label for idx, label in labels["cat"].items()}
    idx_to_style = {int(idx): label for idx, label in labels["style"].items()}
    cat_to_idx = {label: idx for idx, label in idx_to_cat.items()}
    style_to_idx = {label: idx for idx, label in idx_to_style.items()}
    return idx_to_cat, idx_to_style, cat_to_idx, style_to_idx


def load_test_dataframe(args):
    test_split = Path(args.splits_dir) / "test.csv"
    if test_split.exists():
        return pd.read_csv(test_split)

    df = prepare_resnet_dataframe(args.csv_path, args.image_dir, max_per_class=args.max_per_class)
    _, _, test_df = split_dataframe(df, "articleType", seed=args.seed)
    return test_df


def save_confusion_matrix(path, matrix, class_names):
    cm_df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    cm_df.index.name = "actual"
    cm_df.columns.name = "predicted"
    cm_df.to_csv(path)


def summarize_task(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
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
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    idx_to_cat, idx_to_style, cat_to_idx, style_to_idx = load_labels(args.labels, device)
    category_names = [idx_to_cat[idx] for idx in sorted(idx_to_cat)]
    style_names = [idx_to_style[idx] for idx in sorted(idx_to_style)]

    test_df = load_test_dataframe(args)
    test_df = test_df[
        test_df["articleType"].isin(cat_to_idx)
        & test_df["usage"].isin(style_to_idx)
    ].reset_index(drop=True)

    dataset = ResNetEvalDataset(test_df, args.image_dir, cat_to_idx, style_to_idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = MultiTaskResNet(len(category_names), len(style_names), weights=None)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    if not args.no_warmup and len(dataset) > 0:
        warmup_x, _, _ = dataset[0]
        with torch.no_grad():
            model(warmup_x.unsqueeze(0).to(device))

    y_true_cat = []
    y_pred_cat = []
    y_true_style = []
    y_pred_style = []

    start = time.perf_counter()
    with torch.no_grad():
        for inputs, categories, styles in dataloader:
            inputs = inputs.to(device)
            cat_out, style_out = model(inputs)
            y_pred_cat.extend(torch.argmax(cat_out, dim=1).cpu().numpy().tolist())
            y_pred_style.extend(torch.argmax(style_out, dim=1).cpu().numpy().tolist())
            y_true_cat.extend(categories.numpy().tolist())
            y_true_style.extend(styles.numpy().tolist())
    inference_time = time.perf_counter() - start

    y_true_cat = np.array(y_true_cat)
    y_pred_cat = np.array(y_pred_cat)
    y_true_style = np.array(y_true_style)
    y_pred_style = np.array(y_pred_style)

    cat_labels = list(range(len(category_names)))
    style_labels = list(range(len(style_names)))
    category_cm = confusion_matrix(y_true_cat, y_pred_cat, labels=cat_labels)
    style_cm = confusion_matrix(y_true_style, y_pred_style, labels=style_labels)
    category_report = classification_report(
        y_true_cat,
        y_pred_cat,
        labels=cat_labels,
        target_names=category_names,
        output_dict=True,
        zero_division=0,
    )
    style_report = classification_report(
        y_true_style,
        y_pred_style,
        labels=style_labels,
        target_names=style_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "test_size": int(len(dataset)),
        "category": summarize_task(y_true_cat, y_pred_cat, category_names),
        "style": summarize_task(y_true_style, y_pred_style, style_names),
        "joint_accuracy": float(np.mean((y_true_cat == y_pred_cat) & (y_true_style == y_pred_style))),
        "inference_time_seconds": float(inference_time),
        "inference_time_per_image_seconds": float(inference_time / len(dataset)) if len(dataset) else 0.0,
        "params": int(sum(param.numel() for param in model.parameters())),
        "category_classes": category_names,
        "style_classes": style_names,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(output_dir / "category_report.json", "w", encoding="utf-8") as f:
        json.dump(category_report, f, ensure_ascii=False, indent=2)
    with open(output_dir / "style_report.json", "w", encoding="utf-8") as f:
        json.dump(style_report, f, ensure_ascii=False, indent=2)
    save_confusion_matrix(output_dir / "category_confusion_matrix.csv", category_cm, category_names)
    save_confusion_matrix(output_dir / "style_confusion_matrix.csv", style_cm, style_names)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nCategory classification report:")
    print(classification_report(y_true_cat, y_pred_cat, labels=cat_labels, target_names=category_names, zero_division=0))
    print("Category confusion matrix:")
    print(pd.DataFrame(category_cm, index=category_names, columns=category_names))
    print("\nStyle classification report:")
    print(classification_report(y_true_style, y_pred_style, labels=style_labels, target_names=style_names, zero_division=0))
    print("Style confusion matrix:")
    print(pd.DataFrame(style_cm, index=style_names, columns=style_names))
    print(f"\nSaved evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
