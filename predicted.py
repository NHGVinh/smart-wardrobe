import numpy as np
import tensorflow as tf
import json
import time
from utils.preprocess import preprocess_image


# =========================
# 1. Load model
# =========================

model = tf.keras.models.load_model("clothes_model.h5")


# =========================
# 2. Load label map
# =========================

with open("label_map.json", "r", encoding="utf-8") as f:
    idx_to_label = json.load(f)

# convert key từ string → int
idx_to_label = {int(k): v for k, v in idx_to_label.items()}


# =========================
# 3. Preprocess ảnh
# =========================

user_img_path = "test_images/test_pants_1.png"   

img = preprocess_image(user_img_path, target_size=(128, 128), mode = "default")

# thêm batch dimension
img = np.expand_dims(img, axis=0)


# =========================
# 4. Predict
# =========================

inference_start = time.perf_counter()
pred = model.predict(img)
inference_time = time.perf_counter() - inference_start

pred_class = np.argmax(pred)
confidence = np.max(pred)


# =========================
# 5. In kết quả
# =========================

print("Prediction:", idx_to_label[pred_class])
print("Confidence:", confidence)
print(f"Inference time: {inference_time:.4f} seconds")


# =========================
# 6. In top-3 
# =========================

top3 = np.argsort(pred[0])[-3:][::-1]

print("\nTop 3 predictions:")
for i in top3:
    print(f"{idx_to_label[i]}: {pred[0][i]:.4f}")

# chạy thử logic xuất style
from utils.style_engine import infer_style
predicted_class = idx_to_label[pred_class]
from utils.color_extractor import extract_color_features, rgb_to_color_name
rgb, complexity = extract_color_features(user_img_path)
color = rgb_to_color_name(rgb)

style, scores = infer_style(
    predicted_class,
    color,
    pattern_score=0.3,
    color_complexity=complexity
)

print("Color:", color)
print("Color complexity:", complexity)
print("Style:", style)

print("Predicted class:", predicted_class)
print("Inferred style:", style)
print("Style scores:", scores)

import cv2
import matplotlib.pyplot as plt


import cv2
import matplotlib.pyplot as plt


import cv2
import matplotlib.pyplot as plt


import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_item_image_path(item):
    if item.get("source") == "User uploaded item":
        return item["image_path"]

    image_id = item.get("id")
    return f"data/images/{image_id}.jpg"


def fit_image_to_canvas(img, canvas_w, canvas_h):
    h, w = img.shape[:2]

    # Preserve aspect ratio and avoid enlarging low-resolution images.
    scale = min(canvas_w / w, canvas_h / h, 1.0)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))

    if resized_w != w or resized_h != h:
        img = cv2.resize(
            img,
            (resized_w, resized_h),
            interpolation=cv2.INTER_AREA
        )

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    x = (canvas_w - resized_w) // 2
    y = (canvas_h - resized_h) // 2
    canvas[y:y + resized_h, x:x + resized_w] = img
    return canvas


def show_outfit_balanced(outfit):
    loaded = []

    # 1. Load ảnh gốc
    for slot, item in outfit.items():
        if item is None:
            continue

        img_path = get_item_image_path(item)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Cannot load image: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        loaded.append({
            "slot": slot,
            "img": img,
            "path": img_path,
            "w": w,
            "h": h,
            "area": w * h
        })

    if len(loaded) == 0:
        print("No images to show.")
        return

    # 2. Lấy 2 ảnh nhỏ nhất theo diện tích
    canvas_w = max(320, int(np.median([x["w"] for x in loaded])))
    canvas_h = max(320, int(np.median([x["h"] for x in loaded])))

    # 3. Tính target size trung bình từ 2 ảnh nhỏ nhất
    target_w = canvas_w
    target_h = canvas_h

    # chống target quá nhỏ
    target_w = max(target_w, 320)
    target_h = max(target_h, 320)

    print("Display canvas size:", target_w, "x", target_h)

    # 4. Show grid
    cols = 2
    rows = int(np.ceil(len(loaded) / cols))

    plt.figure(figsize=(cols * 4, rows * 4))

    for i, item in enumerate(loaded, 1):
        img = item["img"]

        # Resize về target size chung
        img_resized = fit_image_to_canvas(img, target_w, target_h)

        plt.subplot(rows, cols, i)
        plt.imshow(img_resized)
        plt.title(item["slot"])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

from utils.recommender import recommend_outfit
gender = input("Choose gender (Men/Women/Unisex): ").strip()

if gender not in ["Men", "Women", "Unisex"]:
    gender = "Unisex"

outfit = recommend_outfit(
    csv_path="styles.csv",
    input_class=predicted_class,
    input_color=color,
    input_style=style,
    user_image_path=user_img_path,
    gender=gender
)
show_outfit_balanced(outfit)
