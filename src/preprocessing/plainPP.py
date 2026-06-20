import cv2
import numpy as np


def preprocess_plain_image(img_path, target_size=(128, 128), mode="default"):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    if mode == "shoes":
        return _preprocess_shoes(img, target_size)
    return _preprocess_default(img, target_size)


def _preprocess_default(img, target_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= 500:
            x, y, w, h = cv2.boundingRect(largest)
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)
            img = img[y1:y2, x1:x2]

    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img / 255.0).astype(np.float32)


def _preprocess_shoes(img, target_size):
    h, w = img.shape[:2]
    size = min(h, w)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, cx + size // 2)
    y2 = min(h, cy + size // 2)

    img = img[y1:y2, x1:x2]
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img / 255.0).astype(np.float32)
