import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.resnet import MultiTaskResNet
from src.preprocessing.resnetPP import build_resnet_transforms, load_resnet_image


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the ResNet pipeline.")
    parser.add_argument("image_path")
    parser.add_argument("--checkpoint", default=str(ROOT / "experiments" / "results" / "resnet" / "model.pth"))
    parser.add_argument("--labels", default=str(ROOT / "experiments" / "results" / "resnet" / "labels_map.pth"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels = torch.load(args.labels, weights_only=False, map_location=device)
    idx_to_cat = labels["cat"]
    idx_to_style = labels["style"]

    model = MultiTaskResNet(len(idx_to_cat), len(idx_to_style), weights=None)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    batch = build_resnet_transforms(train=False)(load_resnet_image(args.image_path)).unsqueeze(0).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        cat_out, style_out = model(batch)
        cat_probs = torch.softmax(cat_out, dim=1)
        style_probs = torch.softmax(style_out, dim=1)
        cat_conf, cat_pred = torch.max(cat_probs, 1)
        style_conf, style_pred = torch.max(style_probs, 1)
    elapsed = time.perf_counter() - start

    print(f"Category: {idx_to_cat[int(cat_pred.item())]} ({float(cat_conf.item()):.4f})")
    print(f"Style: {idx_to_style[int(style_pred.item())]} ({float(style_conf.item()):.4f})")
    print(f"Inference time: {elapsed:.4f} seconds")


if __name__ == "__main__":
    main()
