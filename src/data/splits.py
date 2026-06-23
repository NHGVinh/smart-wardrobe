from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


CURRENT_TF_CLASSES = {
    "Topwear": ["Tshirts", "Shirts", "Tops", "Kurtas"],
    "Bottomwear": ["Jeans", "Trousers", "Shorts", "Track Pants"],
    "Shoes": ["Heels", "Flats", "Sandals", "Flip Flops"],
    "Dress": ["Dresses"],
    "Accessories": ["Handbags", "Watches", "Belts", "Wallets"],
}

DEFAULT_MAX_PER_CLASS = 600

RESNET_TARGET_CATEGORIES = [
    "Tshirts",
    "Shirts",
    "Top",
    "Tops",
    "Sweaters",
    "Jackets",
    "Jeans",
    "Trousers",
    "Shorts",
    "Skirts",
    "Track Pants",
    "Casual Shoes",
    "Formal Shoes",
    "Sports Shoes",
    "Heels",
    "Flats",
    "Dresses",
]


def map_vgg_article_type(article_type):
    article_type = str(article_type)
    if article_type in CURRENT_TF_CLASSES["Topwear"]:
        return "Topwear"
    if article_type in CURRENT_TF_CLASSES["Bottomwear"]:
        return "Bottomwear"
    if "Shoes" in article_type or article_type in CURRENT_TF_CLASSES["Shoes"]:
        return "Shoes"
    if article_type in CURRENT_TF_CLASSES["Dress"]:
        return "Dress"
    if article_type in CURRENT_TF_CLASSES["Accessories"]:
        return "Accessories"
    return None


def filter_existing_images(df, image_dir):
    image_dir = Path(image_dir)
    return df[df["id"].apply(lambda image_id: (image_dir / f"{image_id}.jpg").exists())].copy()


def prepare_vgg_dataframe(csv_path, image_dir, max_per_class=DEFAULT_MAX_PER_CLASS, seed=42):
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df["label_name"] = df["articleType"].apply(map_vgg_article_type)
    df = df.dropna(subset=["label_name"])
    df = filter_existing_images(df, image_dir)
    return (
        df.sample(frac=1, random_state=seed)
        .groupby("label_name", group_keys=False)
        .head(max_per_class)
        .reset_index(drop=True)
    )


def prepare_resnet_dataframe(csv_path, image_dir, max_per_class=DEFAULT_MAX_PER_CLASS, seed=42):
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df["label_name"] = df["articleType"].apply(map_vgg_article_type)
    df = df.dropna(subset=["label_name", "usage"])
    df = filter_existing_images(df, image_dir)
    return (
        df.sample(frac=1, random_state=seed)
        .groupby("label_name", group_keys=False)
        .head(max_per_class)
        .reset_index(drop=True)
    )


def split_dataframe(df, stratify_col, seed=42, test_size=0.2, val_size=0.2):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[stratify_col],
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df[stratify_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split_csvs(output_dir, train_df, val_df, test_df):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


map_plain_article_type = map_vgg_article_type
prepare_plain_dataframe = prepare_vgg_dataframe
