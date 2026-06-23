import shutil
import sqlite3
import uuid
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "wardrobe.db"
CLOSET_DIR = BASE_DIR / "my_closet"

TOP_CATEGORIES = [
    "Tshirts",
    "Shirts",
    "Tops",
    "Jackets",
    "Sweaters",
    "Kurtas",
]

BOTTOM_CATEGORIES = [
    "Jeans",
    "Trousers",
    "Shorts",
    "Track Pants",
    "Skirts",
]

SHOE_CATEGORIES = [
    "Casual Shoes",
    "Formal Shoes",
    "Sports Shoes",
    "Heels",
    "Flats",
    "Sandals",
]

DRESS_CATEGORIES = [
    "Dresses",
]


def initialize_database():
    CLOSET_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wardrobe_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL UNIQUE,
                model_name TEXT,
                category TEXT NOT NULL,
                style TEXT,
                color TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()


def save_uploaded_image(uploaded_file) -> str:
    CLOSET_DIR.mkdir(parents=True, exist_ok=True)

    original_name = _uploaded_file_name(uploaded_file)
    suffix = Path(original_name).suffix.lower() or ".jpg"
    target = CLOSET_DIR / f"{uuid.uuid4().hex}{suffix}"

    if isinstance(uploaded_file, (str, Path)):
        shutil.copyfile(Path(uploaded_file), target)
    elif isinstance(uploaded_file, dict) and "content" in uploaded_file:
        target.write_bytes(uploaded_file["content"])
    elif hasattr(uploaded_file, "getbuffer"):
        target.write_bytes(bytes(uploaded_file.getbuffer()))
    elif hasattr(uploaded_file, "read"):
        target.write_bytes(uploaded_file.read())
    else:
        raise TypeError("Unsupported uploaded_file type.")

    return _relative_path(target)


def add_wardrobe_item(
    image_path,
    category,
    style=None,
    color=None,
    model_name=None,
    confidence=None,
):
    initialize_database()
    relative_image_path = _normalize_relative_path(image_path)
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO wardrobe_items (
                image_path,
                model_name,
                category,
                style,
                color,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_path) DO UPDATE SET
                model_name = excluded.model_name,
                category = excluded.category,
                style = excluded.style,
                color = excluded.color,
                confidence = excluded.confidence
            """,
            (
                relative_image_path,
                model_name,
                category,
                style,
                color,
                confidence,
            ),
        )
        conn.commit()

        if cursor.lastrowid:
            return get_wardrobe_item_by_id(cursor.lastrowid)

    return _get_wardrobe_item_by_path(relative_image_path)


def get_all_wardrobe_items():
    initialize_database()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, image_path, model_name, category, style, color, confidence, created_at
            FROM wardrobe_items
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_wardrobe_item_by_id(item_id):
    initialize_database()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, image_path, model_name, category, style, color, confidence, created_at
            FROM wardrobe_items
            WHERE id = ?
            """,
            (item_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def delete_wardrobe_item(item_id):
    initialize_database()
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM wardrobe_items WHERE id = ?", (item_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_items_by_category(categories):
    initialize_database()
    categories = _as_list(categories)
    if not categories:
        return []

    placeholders = ", ".join("?" for _ in categories)
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, image_path, model_name, category, style, color, confidence, created_at
            FROM wardrobe_items
            WHERE category IN ({placeholders})
            ORDER BY created_at DESC, id DESC
            """,
            categories,
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_items_by_style(style):
    initialize_database()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, image_path, model_name, category, style, color, confidence, created_at
            FROM wardrobe_items
            WHERE style = ?
            ORDER BY created_at DESC, id DESC
            """,
            (style,),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_items_by_style_and_category(style, categories, exclude_id=None):
    initialize_database()
    categories = _as_list(categories)
    if not categories:
        return []

    params = [style, *categories]
    placeholders = ", ".join("?" for _ in categories)
    exclude_clause = ""
    if exclude_id is not None:
        exclude_clause = "AND id != ?"
        params.append(exclude_id)

    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, image_path, model_name, category, style, color, confidence, created_at
            FROM wardrobe_items
            WHERE style = ?
              AND category IN ({placeholders})
              {exclude_clause}
            ORDER BY created_at DESC, id DESC
            """,
            params,
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def _connect():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row):
    return dict(row)


def _uploaded_file_name(uploaded_file):
    if isinstance(uploaded_file, dict):
        return uploaded_file.get("filename") or "upload.jpg"
    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file).name
    return getattr(uploaded_file, "name", "upload.jpg")


def _normalize_relative_path(image_path):
    path = Path(image_path)
    if path.is_absolute():
        return _relative_path(path)
    return path.as_posix()


def _relative_path(path):
    return Path(path).resolve().relative_to(BASE_DIR).as_posix()


def _get_wardrobe_item_by_path(image_path):
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, image_path, model_name, category, style, color, confidence, created_at
            FROM wardrobe_items
            WHERE image_path = ?
            """,
            (image_path,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)
