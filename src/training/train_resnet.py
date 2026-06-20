import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.resnet_dataset import ResNetFashionDataset
from src.data.splits import prepare_resnet_dataframe, save_split_csvs, split_dataframe
from src.models.resnet import MultiTaskResNet, freeze_resnet_friend_style
from src.preprocessing.resnetPP import build_resnet_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ResNet18 multi-task pipeline.")
    parser.add_argument("--csv-path", default=str(ROOT / "styles.csv"))
    parser.add_argument("--image-dir", default=str(ROOT / "data" / "images"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "results" / "resnet"))
    parser.add_argument("--max-per-class", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    return parser.parse_args()


def run_epoch(model, dataloader, dataset_size, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    running_loss = 0.0
    correct_cat = 0
    correct_style = 0

    for inputs, labels_cat, labels_style in dataloader:
        inputs = inputs.to(device)
        labels_cat = labels_cat.to(device)
        labels_style = labels_style.to(device)
        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            out_cat, out_style = model(inputs)
            loss = criterion(out_cat, labels_cat) + criterion(out_style, labels_style)
            _, preds_cat = torch.max(out_cat, 1)
            _, preds_style = torch.max(out_style, 1)
            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct_cat += torch.sum(preds_cat == labels_cat).item()
        correct_style += torch.sum(preds_style == labels_style).item()

    loss = running_loss / dataset_size
    acc_cat = correct_cat / dataset_size
    acc_style = correct_style / dataset_size
    return loss, acc_cat, acc_style, (acc_cat + acc_style) / 2


def write_history(path, rows):
    fields = [
        "epoch",
        "train_loss",
        "train_category_accuracy",
        "train_style_accuracy",
        "train_mean_accuracy",
        "val_loss",
        "val_category_accuracy",
        "val_style_accuracy",
        "val_mean_accuracy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = prepare_resnet_dataframe(args.csv_path, args.image_dir, args.max_per_class)
    train_df, val_df, test_df = split_dataframe(df, "articleType", seed=args.seed)
    save_split_csvs(output_dir / "splits", train_df, val_df, test_df)

    categories = list(df["articleType"].drop_duplicates())
    styles = list(df["usage"].drop_duplicates())
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    style_to_idx = {style: i for i, style in enumerate(styles)}
    idx_to_cat = {i: cat for cat, i in cat_to_idx.items()}
    idx_to_style = {i: style for style, i in style_to_idx.items()}
    torch.save({"cat": idx_to_cat, "style": idx_to_style}, output_dir / "labels_map.pth")

    datasets = {
        "train": ResNetFashionDataset(
            train_df,
            args.image_dir,
            cat_to_idx,
            style_to_idx,
            build_resnet_transforms(train=True),
        ),
        "val": ResNetFashionDataset(
            val_df,
            args.image_dir,
            cat_to_idx,
            style_to_idx,
            build_resnet_transforms(train=False),
        ),
        "test": ResNetFashionDataset(
            test_df,
            args.image_dir,
            cat_to_idx,
            style_to_idx,
            build_resnet_transforms(train=False),
        ),
    }
    dataloaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=(name == "train"), num_workers=args.num_workers)
        for name, dataset in datasets.items()
    }

    weights = models.ResNet18_Weights.DEFAULT if args.weights == "imagenet" else None
    model = MultiTaskResNet(len(cat_to_idx), len(style_to_idx), weights=weights).to(device)
    freeze_resnet_friend_style(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_state = None
    best_val_mean_acc = -1.0
    history = []
    train_start = time.perf_counter()

    for epoch in range(args.epochs):
        train_loss, train_cat, train_style, train_mean = run_epoch(
            model, dataloaders["train"], len(datasets["train"]), criterion, optimizer, device, train=True
        )
        val_loss, val_cat, val_style, val_mean = run_epoch(
            model, dataloaders["val"], len(datasets["val"]), criterion, optimizer, device, train=False
        )
        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_category_accuracy": train_cat,
            "train_style_accuracy": train_style,
            "train_mean_accuracy": train_mean,
            "val_loss": val_loss,
            "val_category_accuracy": val_cat,
            "val_style_accuracy": val_style,
            "val_mean_accuracy": val_mean,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_mean > best_val_mean_acc:
            best_val_mean_acc = val_mean
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    train_time = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    eval_start = time.perf_counter()
    test_loss, test_cat, test_style, test_mean = run_epoch(
        model, dataloaders["test"], len(datasets["test"]), criterion, optimizer, device, train=False
    )
    eval_time = time.perf_counter() - eval_start

    torch.save(model.state_dict(), output_dir / "model.pth")
    write_history(output_dir / "history.csv", history)
    metrics = {
        "pipeline": "resnet",
        "preprocessing": "resnetPP",
        "model": "resnet",
        "task": "article_type_plus_usage",
        "device": str(device),
        "weights": args.weights,
        "num_rows": int(len(df)),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "num_categories": int(len(cat_to_idx)),
        "num_styles": int(len(style_to_idx)),
        "epochs_ran": int(args.epochs),
        "train_time_seconds": train_time,
        "eval_time_seconds": eval_time,
        "test_loss": test_loss,
        "test_category_accuracy": test_cat,
        "test_style_accuracy": test_style,
        "test_mean_accuracy": test_mean,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
