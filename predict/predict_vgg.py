import argparse
import json
import sys
import time
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
    parser.add_argument("image_path", nargs="?", default=str(ROOT / "test_images" / "test_pants_1.png"))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    return parser.parse_args()


def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        idx_to_label = json.load(f)
    return {int(k): v for k, v in idx_to_label.items()}


def predict_vgg(image_path, model_path, labels_path):
    model = tf.keras.models.load_model(model_path)
    idx_to_label = load_label_map(labels_path)

    img = preprocess_vgg_image(image_path, target_size=(128, 128))
    batch = np.expand_dims(img, axis=0)

    start = time.perf_counter()
    pred = model.predict(batch, verbose=0)
    elapsed = time.perf_counter() - start

    pred_class = int(np.argmax(pred))
    confidence = float(np.max(pred))
    top3 = np.argsort(pred[0])[-3:][::-1]
    predicted_class = idx_to_label[pred_class]

    rgb, complexity = extract_color_features(image_path)
    color = rgb_to_color_name(rgb)
    style, scores = infer_style(
        predicted_class,
        color,
        pattern_score=0.3,
        color_complexity=complexity,
    )

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "top3": [(idx_to_label[int(i)], float(pred[0][i])) for i in top3],
        "color": color,
        "color_complexity": int(complexity),
        "style": style,
        "style_scores": scores,
        "inference_time": elapsed,
    }


def main():
    args = parse_args()
    result = predict_vgg(args.image_path, args.model, args.labels)

    print("Model: VGG")
    print("Prediction:", result["prediction"])
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Inference time: {result['inference_time']:.4f} seconds")

    print("\nTop 3 predictions:")
    for label, score in result["top3"]:
        print(f"{label}: {score:.4f}")

    print("\nRule-based attributes:")
    print("Color:", result["color"])
    print("Color complexity:", result["color_complexity"])
    print("Style:", result["style"])
    print("Style scores:", result["style_scores"])


if __name__ == "__main__":
    main()
