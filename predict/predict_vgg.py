import argparse
import json
import sys
import time
import os
import glob
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL_PATH = ROOT / "weights" / "vgg" / "vgg_model.h5"
DEFAULT_LABELS_PATH = ROOT / "weights" / "vgg" / "label_map.json"

from src.preprocessing.vggPP import preprocess_vgg_image
from utils.vgg.color_extractor import extract_color_features, rgb_to_color_name
from utils.vgg.style_engine import infer_style


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the VGG clothes classifier.")
    # Thay đổi mặc định thành quét cả thư mục test_images
    parser.add_argument("image_path", nargs="?", default=str(ROOT / "test_images"))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    return parser.parse_args()


def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        idx_to_label = json.load(f)
    return {int(k): v for k, v in idx_to_label.items()}


# Đã truyền model và idx_to_label vào đây để không phải load lại mỗi lần gọi
def predict_vgg(image_path, model, idx_to_label):
    
    # --- 1. Preprocess ---
    pre_start = time.perf_counter()
    img = preprocess_vgg_image(image_path, target_size=(128, 128))
    batch = np.expand_dims(img, axis=0)
    pre_time = time.perf_counter() - pre_start

    # --- 2. Inference ---
    inf_start = time.perf_counter()
    pred = model.predict(batch, verbose=0)
    inf_time = time.perf_counter() - inf_start

    pred_class = int(np.argmax(pred))
    confidence = float(np.max(pred))
    top3 = np.argsort(pred[0])[-3:][::-1]
    predicted_class = idx_to_label[pred_class]

    # --- 3. Style Analysis ---
    style_start = time.perf_counter()
    rgb, complexity = extract_color_features(image_path)
    color = rgb_to_color_name(rgb)
    style, scores = infer_style(
        predicted_class,
        color,
        pattern_score=0.3,
        color_complexity=complexity,
    )
    style_time = time.perf_counter() - style_start

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "top3": [(idx_to_label[int(i)], float(pred[0][i])) for i in top3],
        "color": color,
        "color_complexity": int(complexity),
        "style": style,
        "style_scores": scores,
        "time_pre": pre_time,
        "time_inf": inf_time,
        "time_style": style_time
    }


def main():
    args = parse_args()
    
    # 1. LOAD MODEL VÀ NHÃN (Chỉ load 1 lần duy nhất để tối ưu)
    print("--- Loading VGG Model & Label Map ---")
    model = tf.keras.models.load_model(args.model)
    idx_to_label = load_label_map(args.labels)

    # 2. XÁC ĐỊNH LÀ CHẠY 1 ẢNH HAY QUÉT CẢ THƯ MỤC
    path_obj = Path(args.image_path)
    image_paths = []
    
    if path_obj.is_dir():
        # Quét tất cả file .png và .jpg nếu truyền vào là thư mục
        image_paths.extend(glob.glob(str(path_obj / "*.png")))
        image_paths.extend(glob.glob(str(path_obj / "*.jpg")))
    elif path_obj.is_file():
        image_paths = [str(path_obj)]
    else:
        print(f"Error: Path {args.image_path} does not exist.")
        sys.exit(1)

    print(f"\n=> Found {len(image_paths)} images to process.\n")

    # 3. VÒNG LẶP XỬ LÝ HÀNG LOẠT
    for img_path in image_paths:
        file_name = Path(img_path).name
        print("=" * 50)
        print(f"PROCESSING IMAGE: {file_name}")
        print("=" * 50)

        total_start = time.perf_counter()
        
        # Gọi hàm dự đoán
        result = predict_vgg(img_path, model, idx_to_label)
        
        total_time = time.perf_counter() - total_start

        # In kết quả
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")
        print("\nTop 3 predictions:")
        for label, score in result["top3"]:
            print(f" - {label}: {score:.4f}")

        print("\nRule-based attributes:")
        print(f"Color: {result['color']} (Complexity: {result['color_complexity']})")
        print(f"Style: {result['style']}")

        # Báo cáo thời gian (Tiếng Anh)
        print("\n[ TIMING REPORT ]")
        print(f">> Preprocess time:   {result['time_pre']:.4f} seconds")
        print(f">> Inference time:    {result['time_inf']:.4f} seconds")
        print(f">> Style analysis:    {result['time_style']:.4f} seconds")
        print(f">> TOTAL TIME:        {total_time:.4f} seconds\n")


if __name__ == "__main__":
    main()