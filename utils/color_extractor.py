from utils.preprocess import preprocess_image
import numpy as np
from sklearn.cluster import KMeans
import cv2


def extract_color_features(image_path, target_size=(128, 128), k=3):
    # 👉 dùng ROI đã crop
    img = preprocess_image(image_path, target_size)
    h, w, _ = img.shape

    # crop vùng trung tâm 50%
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)
    y1 = int(h * 0.25)
    y2 = int(h * 0.75)

    img = img[y1:y2, x1:x2]
    # convert về dạng pixel
    pixels = img.reshape(-1, 3)

    # scale lại về 0-255 nếu preprocess đã normalize
    pixels = (pixels * 255).astype(np.uint8)

    # KMeans
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)

    dominant = kmeans.cluster_centers_[np.argmax(counts)]

    # ===== color complexity =====
    total = np.sum(counts)
    significant_clusters = sum(c / total > 0.15 for c in counts)

    return dominant.astype(int), significant_clusters

# lấy màu gần nhất
import numpy as np

# palette gần giống styles.csv
COLOR_DICT = {
    "Black": [0, 0, 0],
    "White": [255, 255, 255],
    "Grey": [128, 128, 128],
    "Navy Blue": [0, 0, 128],
    "Blue": [0, 0, 255],
    "Green": [0, 128, 0],
    "Olive": [128, 128, 0],
    "Red": [255, 0, 0],
    "Maroon": [128, 0, 0],
    "Yellow": [255, 255, 0],
    "Orange": [255, 165, 0],
    "Pink": [255, 192, 203],
    "Purple": [128, 0, 128],
    "Brown": [139, 69, 19],
    "Beige": [245, 245, 220],
    "Khaki": [195, 176, 145],
    "Multi": [128, 128, 64]  # fallback
}


def rgb_to_color_name(rgb):
    rgb = np.array(rgb)

    min_dist = float("inf")
    closest_color = None

    for name, value in COLOR_DICT.items():
        dist = np.linalg.norm(rgb - np.array(value))
        if dist < min_dist:
            min_dist = dist
            closest_color = name

    return closest_color