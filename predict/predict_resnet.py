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
    parser = argparse.ArgumentParser(description="Run inference with the ResNet multitask model on multiple images.")
    # Sửa lại để nhận vào nhiều file hoặc 1 folder
    parser.add_argument("image_paths", nargs="*", default=[str(ROOT / "test_images")], help="Path to image(s) or directory.")
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

def load_resnet_model(checkpoint_path=None, labels_path=None):
    """Hàm tách riêng để chỉ load model 1 lần"""
    print("[*] Đang khởi động mô hình ResNet...")
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
    
    print(f"[*] Đã load xong checkpoint: {checkpoint.name} trên {device}")
    return model, idx_to_cat, idx_to_style, device

def main():
    args = parse_args()
    
    # 1. Tìm tất cả các ảnh cần chạy
    all_images = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    for p in args.image_paths:
        path = Path(p)
        if path.is_dir():
            for f in path.glob("*.*"):
                if f.suffix.lower() in valid_exts:
                    all_images.append(f)
        elif path.is_file() and path.suffix.lower() in valid_exts:
            all_images.append(path)
            
    if not all_images:
        print("❌ Không tìm thấy bức ảnh nào để xử lý!")
        return

    # 2. LOAD MODEL (Chỉ 1 lần duy nhất)
    model, idx_to_cat, idx_to_style, device = load_resnet_model(args.checkpoint, args.labels)
    
    wardrobe_df = None
    if args.wardrobe_csv:
        wardrobe_df = load_wardrobe_csv(args.wardrobe_csv)

    print(f"\n[*] Bắt đầu xử lý {len(all_images)} bức ảnh...")

    # 3. VÒNG LẶP QUA TỪNG ẢNH
    for img_path in all_images:
        print("\n" + "="*50)
        print(f"PROCESSING IMAGE: {img_path.name}")
        print("="*50)
        
        total_start = time.perf_counter()
        
        # Tiền xử lý
        start_pp = time.perf_counter()
        try:
            img = load_resnet_image(img_path)
            batch = data_transforms["val"](img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"❌ Lỗi khi đọc ảnh {img_path.name}: {e}")
            continue
        preprocess_time = time.perf_counter() - start_pp

        # Dự đoán
        start_inf = time.perf_counter()
        with torch.no_grad():
            cat_out, style_out = model(batch)
            cat_probs = torch.softmax(cat_out, dim=1)
            style_probs = torch.softmax(style_out, dim=1)
            
            cat_conf, cat_pred = torch.max(cat_probs, 1)
            style_conf, style_pred = torch.max(style_probs, 1)
            
            # Lấy top 3
            top3_cat_conf, top3_cat_idx = torch.topk(cat_probs, min(3, len(idx_to_cat)), dim=1)
            top3_style_conf, top3_style_idx = torch.topk(style_probs, min(3, len(idx_to_style)), dim=1)
            
        inference_time = time.perf_counter() - start_inf
        
        category_name = idx_to_cat[int(cat_pred.item())]
        style_name = idx_to_style[int(style_pred.item())]

        # In kết quả
        print(f"Prediction: {category_name} (Confidence: {float(cat_conf.item()):.4f})")
        print(f"Style: {style_name} (Confidence: {float(style_conf.item()):.4f})")
        
        print("\nTop 3 Category predictions:")
        for i in range(len(top3_cat_idx[0])):
            lbl = idx_to_cat[int(top3_cat_idx[0][i].item())]
            cnf = float(top3_cat_conf[0][i].item())
            print(f" - {lbl}: {cnf:.4f}")

        # Gợi ý Outfit (KNN)
        rec_time = 0.0
        if wardrobe_df is not None:
            start_rec = time.perf_counter()
            outfit = suggest_outfit_by_embedding(
                str(img_path), category_name, style_name, wardrobe_df, model, device
            )
            rec_time = time.perf_counter() - start_rec
            
            print("\nKNN outfit suggestion:")
            for slot, item_path in outfit.items():
                print(f"{slot}: {item_path if item_path else 'No match'}")

        total_time = time.perf_counter() - total_start
        
        # Báo cáo thời gian
        print("\n[ TIMING REPORT ]")
        print(f">> Preprocess time:   {preprocess_time:.4f} seconds")
        print(f">> Inference time:    {inference_time:.4f} seconds")
        if wardrobe_df is not None:
            print(f">> KNN Search time:   {rec_time:.4f} seconds")
        print(f">> TOTAL TIME:        {total_time:.4f} seconds")

if __name__ == "__main__":
    main()