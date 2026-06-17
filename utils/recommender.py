import pandas as pd


NEUTRAL_COLORS = [
    "Black", "White", "Grey", "Navy Blue", "Brown", "Beige", "Khaki"
]

BRIGHT_COLORS = [
    "Red", "Yellow", "Orange", "Pink", "Purple", "Multi"
]


def map_article_type(article_type):
    """
    Map articleType gốc trong styles.csv thành nhóm lớn.
    Phải giống logic lúc train CNN.
    """

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

    return None


def is_bright_color(color):
    return color in BRIGHT_COLORS


def pick_one(df):
    """
    Random 1 item từ dataframe.
    Nếu không có item thì trả None.
    """
    if df is None or len(df) == 0:
        return None

    return df.sample(1).iloc[0].to_dict()


def filter_by_style(df, style):
    """
    Ưu tiên item cùng usage/style.
    Nếu không đủ dữ liệu thì fallback về toàn bộ df.
    """
    style_df = df[df["usage"] == style]

    if len(style_df) < 20:
        return df

    return style_df


def filter_by_neutral_color(df):
    return df[df["baseColour"].isin(NEUTRAL_COLORS)]


def filter_by_preferred_colors(df, colors):
    colors = [c for c in colors if c is not None]
    return df[df["baseColour"].isin(colors)]


def make_user_item(user_image_path, input_class, input_color, input_style, gender):
    """
    Tạo item đại diện cho đồ user upload.
    Item này sẽ được giữ cố định trong outfit.
    """
    return {
        "source": "User uploaded item",
        "image_path": user_image_path,
        "group": input_class,
        "baseColour": input_color,
        "usage": input_style,
        "gender": gender,
        "productDisplayName": "Your uploaded item",
        "articleType": input_class
    }


def recommend_outfit(
    csv_path,
    input_class,
    input_color,
    input_style="Casual",
    user_image_path=None,
    gender="Unisex"
):
    """
    Gợi ý outfit dựa trên món user upload.

    input_class:
        Topwear / Bottomwear / Shoes / Dress / Accessories

    input_color:
        màu chính đã extract từ ảnh user upload

    input_style:
        style đã suy luận bằng style_engine

    user_image_path:
        đường dẫn ảnh user upload để show lại đúng ảnh gốc

    gender:
        Men / Women / Unisex

    Logic:
    - Slot nào trùng input_class thì dùng chính ảnh user upload.
    - Các slot còn lại lấy từ styles.csv.
    - Ưu tiên cùng gender.
    - Ưu tiên cùng style.
    - Nếu áo màu rực thì quần ưu tiên màu trung tính.
    - Giày ưu tiên màu giống áo hoặc quần.
    - Phụ kiện ưu tiên màu giống item upload, nếu không thì màu trung tính.
    """

    df = pd.read_csv(csv_path, on_bad_lines="skip")

    # =========================
    # 1. Filter gender
    # =========================
    if gender != "Unisex":
        df = df[(df["gender"] == gender) | (df["gender"] == "Unisex")]

    # =========================
    # 2. Map articleType -> group
    # =========================
    df["group"] = df["articleType"].apply(map_article_type)
    df = df.dropna(subset=["group"])

    # =========================
    # 3. Filter style
    # =========================
    style_df = filter_by_style(df, input_style)

    outfit = {}

    user_item = make_user_item(
        user_image_path=user_image_path,
        input_class=input_class,
        input_color=input_color,
        input_style=input_style,
        gender=gender
    )

    # =========================
    # 4. Topwear
    # =========================
    if input_class == "Topwear":
        outfit["Topwear"] = user_item
    else:
        candidates = style_df[style_df["group"] == "Topwear"]
        outfit["Topwear"] = pick_one(candidates)

    # =========================
    # 5. Bottomwear
    # =========================
    if input_class == "Bottomwear":
        outfit["Bottomwear"] = user_item
    else:
        candidates = style_df[style_df["group"] == "Bottomwear"]

        # Nếu input là áo màu rực, quần nên trung tính
        if input_class == "Topwear" and is_bright_color(input_color):
            neutral_candidates = filter_by_neutral_color(candidates)

            if len(neutral_candidates) > 0:
                candidates = neutral_candidates

        outfit["Bottomwear"] = pick_one(candidates)

    # =========================
    # 6. Shoes
    # =========================
    if input_class == "Shoes":
        outfit["Shoes"] = user_item
    else:
        candidates = style_df[style_df["group"] == "Shoes"]

        preferred_colors = [input_color]

        # Nếu đã chọn được quần thì giày ưu tiên theo màu quần
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

    # =========================
    # 7. Accessories
    # =========================
    if input_class == "Accessories":
        outfit["Accessories"] = user_item
    else:
        candidates = style_df[style_df["group"] == "Accessories"]

        # Phụ kiện ưu tiên theo màu item upload
        matched = filter_by_preferred_colors(candidates, [input_color])

        if len(matched) > 0:
            candidates = matched
        else:
            neutral_candidates = filter_by_neutral_color(candidates)
            if len(neutral_candidates) > 0:
                candidates = neutral_candidates

        outfit["Accessories"] = pick_one(candidates)

    return outfit