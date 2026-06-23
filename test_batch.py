import json
import os
import time

import numpy as np
import tensorflow as tf

from src.preprocessing.vggPP import preprocess_vgg_image


model = tf.keras.models.load_model("weights/vgg/vgg_model.h5")

with open("weights/vgg/label_map.json", "r", encoding="utf-8") as f:
    idx_to_label = json.load(f)

idx_to_label = {int(k): v for k, v in idx_to_label.items()}

TEST_FOLDER = "test_images"


def predict_image(image_path):
    img = preprocess_vgg_image(image_path, target_size=(128, 128))
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)
    top3 = np.argsort(pred[0])[-3:][::-1]

    return pred_class, confidence, top3, pred[0]


results = []

for file in os.listdir(TEST_FOLDER):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(TEST_FOLDER, file)

    inference_start = time.perf_counter()
    pred_class, confidence, top3, probs = predict_image(path)
    inference_time = time.perf_counter() - inference_start

    print("\n========================")
    print("File:", file)
    print("Prediction:", idx_to_label[pred_class])
    print("Confidence:", round(float(confidence), 4))
    print(f"Inference time: {inference_time:.4f} seconds")

    print("Top 3:")
    for i in top3:
        print(f"  {idx_to_label[i]}: {probs[i]:.4f}")

    results.append((file, idx_to_label[pred_class], confidence, inference_time))


print("\n===== SUMMARY =====")
print("Total images:", len(results))

if results:
    inference_times = [item[3] for item in results]
    total_inference_time = sum(inference_times)
    avg_inference_time = total_inference_time / len(inference_times)

    print(f"Total inference time: {total_inference_time:.4f} seconds")
    print(f"Average inference time per image: {avg_inference_time:.4f} seconds")
