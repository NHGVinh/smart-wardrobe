import argparse
import csv
import json
import sys
import time
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import prepare_plain_dataframe, save_split_csvs, split_dataframe
from src.models.VGG import build_clothes_model
from utils.vgg.data_generator import ClothesDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Train the VGG TensorFlow pipeline.")
    parser.add_argument("--csv-path", default=str(ROOT / "styles.csv"))
    parser.add_argument("--image-dir", default=str(ROOT / "data" / "images"))
    parser.add_argument("--output-dir", default=str(ROOT / "weights" / "vgg"))
    parser.add_argument("--max-per-class", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_history(path, history):
    keys = list(history.history.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", *keys])
        writer.writeheader()
        for idx in range(len(history.history[keys[0]])):
            row = {"epoch": idx + 1}
            row.update({key: history.history[key][idx] for key in keys})
            writer.writerow(row)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_plain_dataframe(
        args.csv_path,
        args.image_dir,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    train_df, val_df, test_df = split_dataframe(df, "label_name", seed=args.seed)
    save_split_csvs(output_dir / "splits", train_df, val_df, test_df)

    classes = sorted(df["label_name"].unique())
    label_map = {label: idx for idx, label in enumerate(classes)}
    idx_to_label = {idx: label for label, idx in label_map.items()}

    train_gen = ClothesDataGenerator(train_df, args.image_dir, label_map, batch_size=args.batch_size, shuffle=True)
    val_gen = ClothesDataGenerator(val_df, args.image_dir, label_map, batch_size=args.batch_size, shuffle=False)
    test_gen = ClothesDataGenerator(test_df, args.image_dir, label_map, batch_size=args.batch_size, shuffle=False)

    model = build_clothes_model(input_shape=(128, 128, 3), num_classes=len(label_map))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    train_start = time.perf_counter()
    history = model.fit(train_gen, epochs=args.epochs, validation_data=val_gen, verbose=1)
    train_time = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    eval_time = time.perf_counter() - eval_start

    model.save(output_dir / "vgg_model.h5")
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, ensure_ascii=False, indent=2)
    write_history(output_dir / "history.csv", history)

    metrics = {
        "pipeline": "vgg",
        "preprocessing": "vggPP",
        "model": "vgg16",
        "task": "grouped_category",
        "classes": classes,
        "num_rows": int(len(df)),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "epochs_ran": int(len(history.history["loss"])),
        "train_time_seconds": train_time,
        "eval_time_seconds": eval_time,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
