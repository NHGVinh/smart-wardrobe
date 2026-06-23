import cv2
import numpy as np


def preprocess_vgg_image(img_path, target_size=(128, 128)):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    roi = _crop_main_object(img)
    padded = _resize_with_white_padding(roi, target_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return (rgb / 255.0).astype(np.float32)


def preprocess_vgg_image_with_mask(img_path, target_size=(128, 128)):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    roi = _crop_main_object(img)
    padded, mask = _resize_with_white_padding(roi, target_size, return_mask=True)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return (rgb / 255.0).astype(np.float32), mask


def _crop_main_object(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return img

    x, y, w, h = cv2.boundingRect(largest)
    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    return img[y1:y2, x1:x2]


def _resize_with_white_padding(img, target_size, return_mask=False):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)

    x = (target_w - resized_w) // 2
    y = (target_h - resized_h) // 2
    canvas[y:y + resized_h, x:x + resized_w] = resized

    if not return_mask:
        return canvas

    mask = np.zeros((target_h, target_w), dtype=bool)
    mask[y:y + resized_h, x:x + resized_w] = True
    return canvas, mask


def preprocess_plain_image(img_path, target_size=(128, 128), mode="default"):
    return preprocess_vgg_image(img_path, target_size=target_size)
