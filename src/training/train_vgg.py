import os
import sys
import json
import time
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import prepare_plain_dataframe, split_dataframe
from src.models.VGG import build_clothes_model
from utils.vgg.data_generator import ClothesDataGenerator


# =========================
# 1. Config
# =========================

CSV_PATH = str(ROOT / "styles.csv")
IMAGE_DIR = str(ROOT / "data" / "images")
WEIGHTS_DIR = ROOT / "weights" / "vgg"
MODEL_PATH = WEIGHTS_DIR / "vgg_model.h5"
LABEL_MAP_PATH = WEIGHTS_DIR / "label_map.json"
TARGET_SIZE = (128, 128)

MAX_PER_CLASS = 750
BATCH_SIZE = 32
EPOCHS = 30


# =========================
# 2. Prepare dataframe
# =========================

df = prepare_plain_dataframe(
    CSV_PATH,
    IMAGE_DIR,
    max_per_class=MAX_PER_CLASS,
    seed=42,
)


# bá» nhá»¯ng item khÃ´ng thuá»™c nhÃ³m cáº§n train
print("Prepared images:", len(df))
print(df["label_name"].value_counts())


# =========================
# 3. Create label map
# =========================

selected_classes = sorted(df["label_name"].unique())

label_map = {label: idx for idx, label in enumerate(selected_classes)}
idx_to_label = {idx: label for label, idx in label_map.items()}

df["label"] = df["label_name"].map(label_map)

NUM_CLASSES = len(label_map)

print("Selected classes:", selected_classes)
print("NUM_CLASSES:", NUM_CLASSES)


# =========================
# 4. Train / Val / Test split
# =========================

train_df, val_df, test_df = split_dataframe(df, stratify_col="label_name", seed=42)

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))


# =========================
# 5. Data generators
# =========================

train_gen = ClothesDataGenerator(
    train_df,
    IMAGE_DIR,
    label_map,
    batch_size=BATCH_SIZE,
    target_size=TARGET_SIZE,
    shuffle=True
)

val_gen = ClothesDataGenerator(
    val_df,
    IMAGE_DIR,
    label_map,
    batch_size=BATCH_SIZE,
    target_size=TARGET_SIZE,
    shuffle=False
)

test_gen = ClothesDataGenerator(
    test_df,
    IMAGE_DIR,
    label_map,
    batch_size=BATCH_SIZE,
    target_size=TARGET_SIZE,
    shuffle=False
)


# =========================
# 6. Build model
# =========================

model = build_clothes_model(
    input_shape=(128, 128, 3),
    num_classes=NUM_CLASSES
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# 7. EarlyStopping
# =========================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# =========================
# 8. Train
# =========================

train_start = time.perf_counter()

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[early_stop],
    verbose=1
)

train_time = time.perf_counter() - train_start
epochs_ran = len(history.history["loss"])

print(f"Training time: {train_time:.2f} seconds")
print(f"Training time per epoch: {train_time / epochs_ran:.2f} seconds")


# =========================
# 9. Evaluate
# =========================

eval_start = time.perf_counter()

test_loss, test_acc = model.evaluate(test_gen)

eval_time = time.perf_counter() - eval_start

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
print(f"Evaluation time: {eval_time:.2f} seconds")


# =========================
# 10. Save model + label map
# =========================

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

model.save(str(MODEL_PATH))

with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(idx_to_label, f, ensure_ascii=False, indent=4)

print(f"Saved model to {MODEL_PATH}")
print(f"Saved label map to {LABEL_MAP_PATH}")
