from src.database.db import (
    BOTTOM_CATEGORIES,
    DRESS_CATEGORIES,
    SHOE_CATEGORIES,
    TOP_CATEGORIES,
    get_items_by_category,
    get_items_by_style_and_category,
)


def recommend_outfit_from_wardrobe(selected_item):
    if not selected_item:
        return {
            "input": None,
            "top": None,
            "bottom": None,
            "shoes": None,
            "dress": None,
        }

    category = selected_item["category"]
    style = selected_item.get("style")
    selected_id = selected_item.get("id")

    outfit = {
        "input": selected_item,
        "top": None,
        "bottom": None,
        "shoes": None,
        "dress": None,
    }

    if category in TOP_CATEGORIES:
        outfit["top"] = selected_item
        outfit["bottom"] = _pick_item(BOTTOM_CATEGORIES, style, selected_id)
        outfit["shoes"] = _pick_item(SHOE_CATEGORIES, style, selected_id)
    elif category in BOTTOM_CATEGORIES:
        outfit["bottom"] = selected_item
        outfit["top"] = _pick_item(TOP_CATEGORIES, style, selected_id)
        outfit["shoes"] = _pick_item(SHOE_CATEGORIES, style, selected_id)
    elif category in SHOE_CATEGORIES:
        outfit["shoes"] = selected_item
        outfit["top"] = _pick_item(TOP_CATEGORIES, style, selected_id)
        outfit["bottom"] = _pick_item(BOTTOM_CATEGORIES, style, selected_id)
    elif category in DRESS_CATEGORIES:
        outfit["dress"] = selected_item
        outfit["shoes"] = _pick_item(SHOE_CATEGORIES, style, selected_id)
    else:
        outfit["top"] = selected_item

    return outfit


def _pick_item(categories, style, exclude_id):
    if style:
        same_style_items = get_items_by_style_and_category(
            style,
            categories,
            exclude_id=exclude_id,
        )
        if same_style_items:
            return same_style_items[0]

    fallback_items = [
        item
        for item in get_items_by_category(categories)
        if item.get("id") != exclude_id
    ]
    return fallback_items[0] if fallback_items else None
