def infer_style(predicted_class, color, pattern_score=0.3,color_complexity = 1):
    """
    predicted_class: Topwear / Bottomwear / Shoes / Dress / Accessories
    color: string, ví dụ "Black", "Red", "Blue"
    pattern_score: float (0 → trơn, 1 → nhiều họa tiết)
    """

    scores = {
        "Formal": 0,
        "Casual": 0,
        "Sports": 0,
        "Streetwear": 0
    }

    # =========================
    # 1. RULE BASED ON COLOR
    # =========================

    neutral_colors = ["Black", "White", "Grey", "Navy Blue", "Brown"]
    bright_colors = ["Red", "Yellow", "Orange", "Pink"]
    cool_colors = ["Blue", "Green"]

    if color in neutral_colors:
        scores["Formal"] += 2
        scores["Casual"] += 1

    elif color in bright_colors:
        scores["Streetwear"] += 2

    elif color in cool_colors:
        scores["Casual"] += 2

    # =========================
    # 2. RULE BASED ON ITEM TYPE
    # =========================

    if predicted_class == "Topwear":
        scores["Casual"] += 2
        scores["Formal"] += 1

    elif predicted_class == "Bottomwear":
        scores["Casual"] += 2
        scores["Formal"] += 1

    elif predicted_class == "Shoes":
        scores["Casual"] += 1
        scores["Sports"] += 2

    elif predicted_class == "Dress":
        scores["Formal"] += 2
        scores["Casual"] += 1

    elif predicted_class == "Accessories":
        scores["Formal"] += 1
        scores["Casual"] += 1

    # =========================
    # 3. RULE BASED ON PATTERN
    # =========================

    if pattern_score < 0.3:
        scores["Formal"] += 2

    elif pattern_score < 0.6:
        scores["Casual"] += 2

    else:
        scores["Streetwear"] += 2

    # =========================
    # 4. FINAL DECISION
    # =========================

    final_style = max(scores, key=scores.get)

    return final_style, scores