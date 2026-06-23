import numpy as np
from sklearn.cluster import KMeans

from src.preprocessing.vggPP import preprocess_vgg_image_with_mask


def extract_color_features(image_path, target_size=(128, 128), k=3):
    img, item_mask = preprocess_vgg_image_with_mask(image_path, target_size)
    pixels = (img[item_mask] * 255).astype(np.uint8)

    k = min(k, len(pixels))
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]

    total = np.sum(counts)
    significant_clusters = sum(c / total > 0.15 for c in counts)

    return dominant.astype(int), significant_clusters


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
    "Multi": [128, 128, 64],
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
