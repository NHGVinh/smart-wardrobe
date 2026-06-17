import os
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils.data_generator import ClothesDataGenerator
from models.clothes_model import build_clothes_model


# =========================
# 1. Config
# =========================

CSV_PATH = "styles.csv"
IMAGE_DIR = "data/images"
TARGET_SIZE = (128, 128)

MAX_PER_CLASS = 750
BATCH_SIZE = 32
EPOCHS = 30


# =========================
# 2. Read CSV
# =========================

df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

print("Total rows:", len(df))


# =========================
# 3. Map articleType -> grouped class
# =========================

def map_article_type(article_type):
    if article_type in ["Tshirts", "Shirts", "Tops", "Kurtas"]:
        return "Topwear"

    elif article_type in ["Jeans", "Trousers", "Shorts", "Track Pants"]:
        return "Bottomwear"

    elif (
        "Shoes" in str(article_type)
        or article_type in ["Heels", "Flats", "Sandals", "Flip Flops"]
    ):
        return "Shoes"

    elif article_type in ["Dresses"]:
        return "Dress"

    elif article_type in ["Handbags", "Watches", "Belts", "Wallets"]:
        return "Accessories"

    else:
        return None


df["label_name"] = df["articleType"].apply(map_article_type)

# bỏ những item không thuộc nhóm cần train
df = df.dropna(subset=["label_name"])

print("Classes after mapping:")
print(df["label_name"].value_counts())


# =========================
# 4. Balance dataset
# =========================

df = (
    df.groupby("label_name", group_keys=False)
      .head(MAX_PER_CLASS)
)

print("Number of images after balancing:", len(df))
print(df["label_name"].value_counts())


# =========================
# 5. Create label map
# =========================

selected_classes = sorted(df["label_name"].unique())

label_map = {label: idx for idx, label in enumerate(selected_classes)}
idx_to_label = {idx: label for label, idx in label_map.items()}

df["label"] = df["label_name"].map(label_map)

NUM_CLASSES = len(label_map)

print("Selected classes:", selected_classes)
print("NUM_CLASSES:", NUM_CLASSES)


# =========================
# 6. Train / Val / Test split
# =========================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_name"]
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["label_name"]
)

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))


# =========================
# 7. Data generators
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
# 8. Build model
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
# 9. EarlyStopping
# =========================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# =========================
# 10. Train
# =========================

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[early_stop],
    verbose=1
)


# =========================
# 11. Evaluate
# =========================

test_loss, test_acc = model.evaluate(test_gen)

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)


# =========================
# 12. Save model + label map
# =========================

model.save("clothes_model.h5")

with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(idx_to_label, f, ensure_ascii=False, indent=4)

print("Saved model to clothes_model.h5")
print("Saved label map to label_map.json")