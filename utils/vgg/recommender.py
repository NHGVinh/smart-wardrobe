from pathlib import Path

import pandas as pd


NEUTRAL_COLORS = ["Black", "White", "Grey", "Navy Blue", "Brown", "Beige", "Khaki"]
BRIGHT_COLORS = ["Red", "Yellow", "Orange", "Pink", "Purple", "Multi"]


def map_article_type(article_type):
    if article_type in ["Tshirts", "Shirts", "Tops", "Kurtas"]:
        return "Topwear"
    if article_type in ["Jeans", "Trousers", "Shorts", "Track Pants"]:
        return "Bottomwear"
    if "Shoes" in str(article_type) or article_type in ["Heels", "Flats", "Sandals", "Flip Flops"]:
        return "Shoes"
    if article_type in ["Dresses"]:
        return "Dress"
    if article_type in ["Handbags", "Watches", "Belts", "Wallets"]:
        return "Accessories"
    return None


def is_bright_color(color):
    return color in BRIGHT_COLORS


def pick_one(df):
    if df is None or len(df) == 0:
        return None
    return df.sample(1).iloc[0].to_dict()


def add_image_paths(df, csv_path, image_dir=None):
    if "image_path" in df.columns:
        return df

    image_dir = Path(image_dir) if image_dir else Path(csv_path).resolve().parent / "data" / "images"
    df = df.copy()
    df["image_path"] = df["id"].apply(lambda image_id: str(image_dir / f"{image_id}.jpg"))
    return df


def filter_by_style(df, style):
    style_df = df[df["usage"] == style]
    if len(style_df) < 20:
        return df
    return style_df


def filter_by_neutral_color(df):
    return df[df["baseColour"].isin(NEUTRAL_COLORS)]


def filter_by_preferred_colors(df, colors):
    colors = [color for color in colors if color is not None]
    return df[df["baseColour"].isin(colors)]


def make_user_item(user_image_path, input_class, input_color, input_style, gender):
    return {
        "source": "User uploaded item",
        "image_path": user_image_path,
        "group": input_class,
        "baseColour": input_color,
        "usage": input_style,
        "gender": gender,
        "productDisplayName": "Your uploaded item",
        "articleType": input_class,
    }


def recommend_outfit(
    csv_path,
    input_class,
    input_color,
    input_style="Casual",
    user_image_path=None,
    gender="Unisex",
    image_dir=None,
):
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df = add_image_paths(df, csv_path, image_dir=image_dir)

    if gender != "Unisex":
        df = df[(df["gender"] == gender) | (df["gender"] == "Unisex")]

    df["group"] = df["articleType"].apply(map_article_type)
    df = df.dropna(subset=["group"])
    style_df = filter_by_style(df, input_style)

    outfit = {}
    user_item = make_user_item(
        user_image_path=user_image_path,
        input_class=input_class,
        input_color=input_color,
        input_style=input_style,
        gender=gender,
    )

    if input_class == "Dress":
        outfit["Dress"] = user_item
    elif input_class == "Topwear":
        outfit["Topwear"] = user_item
    else:
        outfit["Topwear"] = pick_one(style_df[style_df["group"] == "Topwear"])

    if input_class == "Dress":
        pass
    elif input_class == "Bottomwear":
        outfit["Bottomwear"] = user_item
    else:
        candidates = style_df[style_df["group"] == "Bottomwear"]
        if input_class == "Topwear" and is_bright_color(input_color):
            neutral_candidates = filter_by_neutral_color(candidates)
            if len(neutral_candidates) > 0:
                candidates = neutral_candidates
        outfit["Bottomwear"] = pick_one(candidates)

    if input_class == "Shoes":
        outfit["Shoes"] = user_item
    else:
        candidates = style_df[style_df["group"] == "Shoes"]
        preferred_colors = [input_color]
        bottomwear = outfit.get("Bottomwear")
        if bottomwear is not None:
            preferred_colors.append(bottomwear.get("baseColour"))

        matched = filter_by_preferred_colors(candidates, preferred_colors)
        if len(matched) > 0:
            candidates = matched
        else:
            neutral_candidates = filter_by_neutral_color(candidates)
            if len(neutral_candidates) > 0:
                candidates = neutral_candidates
        outfit["Shoes"] = pick_one(candidates)

    if input_class == "Accessories":
        outfit["Accessories"] = user_item
    else:
        candidates = style_df[style_df["group"] == "Accessories"]
        matched = filter_by_preferred_colors(candidates, [input_color])
        if len(matched) > 0:
            candidates = matched
        else:
            neutral_candidates = filter_by_neutral_color(candidates)
            if len(neutral_candidates) > 0:
                candidates = neutral_candidates
        outfit["Accessories"] = pick_one(candidates)

    return outfit
