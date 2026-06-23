import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.resnet import MultiTaskResNet
from src.preprocessing.resnetPP import data_transforms, load_resnet_image
from utils.resnet.embedding_recommender import load_wardrobe_csv, suggest_outfit_by_embedding


DEFAULT_LABELS_PATH = ROOT / "weights" / "resnet" / "labels_map.pth"
DEFAULT_CHECKPOINTS = [
    ROOT / "weights" / "resnet" / "resnet_model.pth",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the ResNet multitask model.")
    parser.add_argument("image_path", nargs="?", default=str(ROOT / "test_images" / "test_pants_1.png"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument(
        "--wardrobe-csv",
        default=None,
        help="Optional CSV with image_path, category, style columns for KNN outfit suggestions.",
    )
    return parser.parse_args()


def resolve_checkpoint(path):
    if path:
        checkpoint = Path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"ResNet checkpoint not found: {checkpoint}")
        return checkpoint

    for checkpoint in DEFAULT_CHECKPOINTS:
        if checkpoint.exists():
            return checkpoint

    options = ", ".join(str(path) for path in DEFAULT_CHECKPOINTS)
    raise FileNotFoundError(f"ResNet checkpoint not found. Expected one of: {options}")


def predict_resnet(image_path, checkpoint_path=None, labels_path=None):
    checkpoint = resolve_checkpoint(checkpoint_path)
    labels_path = Path(labels_path or DEFAULT_LABELS_PATH)
    if not labels_path.exists():
        raise FileNotFoundError(f"ResNet labels map not found: {labels_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels = torch.load(labels_path, weights_only=False, map_location=device)
    idx_to_cat = labels["cat"]
    idx_to_style = labels["style"]

    model = MultiTaskResNet(len(idx_to_cat), len(idx_to_style), weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    batch = data_transforms["val"](load_resnet_image(image_path)).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        cat_out, style_out = model(batch)
        cat_probs = torch.softmax(cat_out, dim=1)
        style_probs = torch.softmax(style_out, dim=1)
        cat_conf, cat_pred = torch.max(cat_probs, 1)
        style_conf, style_pred = torch.max(style_probs, 1)
    elapsed = time.perf_counter() - start

    return {
        "category": idx_to_cat[int(cat_pred.item())],
        "category_confidence": float(cat_conf.item()),
        "style": idx_to_style[int(style_pred.item())],
        "style_confidence": float(style_conf.item()),
        "checkpoint": str(checkpoint),
        "inference_time": elapsed,
        "model": model,
        "device": device,
    }


def main():
    args = parse_args()
    result = predict_resnet(args.image_path, args.checkpoint, args.labels)

    print("Model: ResNet")
    print("Checkpoint:", result["checkpoint"])
    print(f"Category: {result['category']} ({result['category_confidence']:.4f})")
    print(f"Style: {result['style']} ({result['style_confidence']:.4f})")
    print(f"Inference time: {result['inference_time']:.4f} seconds")

    if args.wardrobe_csv:
        wardrobe_df = load_wardrobe_csv(args.wardrobe_csv)
        outfit = suggest_outfit_by_embedding(
            args.image_path,
            result["category"],
            result["style"],
            wardrobe_df,
            result["model"],
            result["device"],
        )

        print("\nKNN outfit suggestion:")
        for slot, image_path in outfit.items():
            print(f"{slot}: {image_path if image_path else 'No match'}")


if __name__ == "__main__":
    main()
