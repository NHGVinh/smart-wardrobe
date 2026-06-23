from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from src.preprocessing.resnetPP import data_transforms, load_resnet_image


TOPS_LIST = ["Topwear", "Tshirts", "Shirts", "Top", "Tops", "Sweaters", "Jackets"]
BOTTOMS_LIST = ["Bottomwear", "Jeans", "Trousers", "Shorts", "Skirts", "Track Pants"]
SHOES_LIST = ["Shoes", "Casual Shoes", "Formal Shoes", "Sports Shoes", "Heels", "Flats", "Sandals", "Flip Flops"]
DRESS_LIST = ["Dress", "Dresses", "Kurtas"]


def load_wardrobe_csv(csv_path):
    wardrobe_df = pd.read_csv(csv_path)
    required_columns = {"image_path", "category", "style"}
    missing_columns = required_columns - set(wardrobe_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Wardrobe CSV is missing required column(s): {missing}")
    return wardrobe_df


def extract_resnet_embedding(model, image_path, device):
    image_path = Path(image_path)
    if not image_path.exists():
        return None

    batch = data_transforms["val"](load_resnet_image(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.resnet(batch)

    return features.squeeze(0).cpu().numpy()


def find_best_by_embedding(target_embedding, candidates_df, model, device, embedding_cache):
    if candidates_df.empty or target_embedding is None:
        return None

    paths = []
    vectors = []

    for path in candidates_df["image_path"]:
        if path not in embedding_cache:
            embedding_cache[path] = extract_resnet_embedding(model, path, device)

        embedding = embedding_cache[path]
        if embedding is None:
            continue

        paths.append(path)
        vectors.append(embedding)

    if not vectors:
        return None

    knn = NearestNeighbors(n_neighbors=1, metric="cosine")
    knn.fit(np.array(vectors))

    _, indices = knn.kneighbors(target_embedding.reshape(1, -1))
    return paths[int(indices[0][0])]


def suggest_outfit_by_embedding(
    user_item_path,
    target_category,
    target_style,
    wardrobe_df,
    model,
    device,
    embedding_cache=None,
):
    if embedding_cache is None:
        embedding_cache = {}

    if wardrobe_df.empty:
        return {"Top": None, "Bottom": None, "Shoes": None}

    target_embedding = extract_resnet_embedding(model, user_item_path, device)
    embedding_cache[str(user_item_path)] = target_embedding

    matching_items = wardrobe_df[wardrobe_df["style"] == target_style]
    outfit = {"Top": None, "Bottom": None, "Shoes": None}

    def best_match(category_list):
        subset = matching_items[matching_items["category"].isin(category_list)]
        return find_best_by_embedding(target_embedding, subset, model, device, embedding_cache)

    if target_category in TOPS_LIST:
        outfit["Top"] = str(user_item_path)
        outfit["Bottom"] = best_match(BOTTOMS_LIST)
        outfit["Shoes"] = best_match(SHOES_LIST)
    elif target_category in DRESS_LIST:
        outfit["Top"] = str(user_item_path)
        outfit["Shoes"] = best_match(SHOES_LIST)
    elif target_category in BOTTOMS_LIST:
        outfit["Bottom"] = str(user_item_path)
        outfit["Top"] = best_match(TOPS_LIST)
        outfit["Shoes"] = best_match(SHOES_LIST)
    elif target_category in SHOES_LIST:
        outfit["Shoes"] = str(user_item_path)
        outfit["Top"] = best_match(TOPS_LIST)
        outfit["Bottom"] = best_match(BOTTOMS_LIST)
    else:
        outfit["Top"] = str(user_item_path)

    return outfit
