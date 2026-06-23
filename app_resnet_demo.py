import argparse
import html
import json
import mimetypes
import re
import sys
import time
import traceback
import urllib.parse
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.db import (
    add_wardrobe_item,
    delete_wardrobe_item,
    get_all_wardrobe_items,
    get_wardrobe_item_by_id,
    initialize_database,
    save_uploaded_image,
)
from utils.resnet.wardrobe_recommender import recommend_outfit_from_wardrobe

UPLOAD_DIR = ROOT / "demo_uploads"
DEFAULT_CHECKPOINT = ROOT / "weights" / "resnet" / "resnet_model.pth"
DEFAULT_LABELS = ROOT / "weights" / "resnet" / "labels_map.pth"
TEST_IMAGES_DIR = ROOT / "test_images"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ResNetDemoRunner:
    def __init__(self):
        self.model = None
        self.device = None
        self.idx_to_cat = None
        self.idx_to_style = None
        self.checkpoint = None
        self.labels = None
        self.embedding_cache = {}

    def load(self, checkpoint_path, labels_path):
        checkpoint_path = Path(checkpoint_path)
        labels_path = Path(labels_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels map: {labels_path}")

        if (
            self.model is not None
            and self.checkpoint == checkpoint_path
            and self.labels == labels_path
        ):
            return

        import torch

        from src.models.resnet import MultiTaskResNet

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        labels = torch.load(labels_path, weights_only=False, map_location=self.device)
        self.idx_to_cat = {int(idx): label for idx, label in labels["cat"].items()}
        self.idx_to_style = {int(idx): label for idx, label in labels["style"].items()}

        self.model = MultiTaskResNet(
            len(self.idx_to_cat),
            len(self.idx_to_style),
            weights=None,
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.checkpoint = checkpoint_path
        self.labels = labels_path
        self.embedding_cache = {}

    def predict(self, image_path, checkpoint_path=DEFAULT_CHECKPOINT, labels_path=DEFAULT_LABELS):
        import torch

        from src.preprocessing.resnetPP import data_transforms, load_resnet_image

        self.load(checkpoint_path, labels_path)

        image_path = Path(image_path)
        batch = data_transforms["val"](load_resnet_image(image_path)).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            cat_out, style_out = self.model(batch)
            cat_probs = torch.softmax(cat_out, dim=1)
            style_probs = torch.softmax(style_out, dim=1)
            cat_conf, cat_pred = torch.max(cat_probs, 1)
            style_conf, style_pred = torch.max(style_probs, 1)
        elapsed = time.perf_counter() - start

        top_categories = self._topk(cat_probs.squeeze(0), self.idx_to_cat)
        top_styles = self._topk(style_probs.squeeze(0), self.idx_to_style)

        return {
            "category": self.idx_to_cat[int(cat_pred.item())],
            "category_confidence": float(cat_conf.item()),
            "style": self.idx_to_style[int(style_pred.item())],
            "style_confidence": float(style_conf.item()),
            "top_categories": top_categories,
            "top_styles": top_styles,
            "checkpoint": str(self.checkpoint),
            "labels": str(self.labels),
            "device": str(self.device),
            "inference_time": elapsed,
        }

    @staticmethod
    def _topk(probabilities, labels, k=3):
        values, indices = probabilities.topk(min(k, len(labels)))
        return [
            {"label": labels[int(index.item())], "score": float(value.item())}
            for value, index in zip(values, indices)
        ]


RUNNER = ResNetDemoRunner()


def safe_filename(name):
    name = Path(name or "upload").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return stem or "upload"


def list_sample_images():
    if not TEST_IMAGES_DIR.exists():
        return []

    images = []
    for path in sorted(TEST_IMAGES_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
            images.append(
                {
                    "name": path.name,
                    "url": f"/sample/{urllib.parse.quote(path.name)}",
                    "size": path.stat().st_size,
                }
            )
    return images


def is_relative_to(path, base):
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


def path_to_url(path):
    if not path:
        return None

    resolved = Path(path).resolve()
    if not resolved.exists() or not is_relative_to(resolved, ROOT):
        return None

    relative = resolved.relative_to(ROOT)
    return "/file?path=" + urllib.parse.quote(str(relative).replace("\\", "/"))


def relative_path_to_url(relative_path):
    if not relative_path:
        return None
    return path_to_url(ROOT / relative_path)


def serialize_wardrobe_item(item):
    if not item:
        return None

    image_path = item.get("image_path")
    absolute_path = ROOT / image_path if image_path else None
    result = dict(item)
    result["image_url"] = relative_path_to_url(image_path)
    result["image_exists"] = bool(absolute_path and absolute_path.exists())
    return result


def serialize_wardrobe_items(items):
    return [serialize_wardrobe_item(item) for item in items]


def parse_multipart(headers, body):
    content_type = headers.get("Content-Type", "")
    raw = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8") + body
    message = BytesParser(policy=policy.default).parsebytes(raw)

    fields = {}
    files = {}

    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue

        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""

        if filename:
            files[name] = {
                "filename": filename,
                "content": payload,
                "content_type": part.get_content_type(),
            }
        elif name:
            fields[name] = payload.decode("utf-8", errors="replace")

    return fields, files


def save_upload(file_info, prefix):
    if not file_info or not file_info["content"]:
        return None

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(file_info["filename"])
    suffix = Path(filename).suffix.lower()
    if prefix == "image" and suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image type. Use JPG, PNG, WEBP, or BMP.")
    if prefix == "wardrobe" and suffix != ".csv":
        raise ValueError("Wardrobe file must be a CSV.")

    target = UPLOAD_DIR / f"{prefix}_{int(time.time() * 1000)}_{filename}"
    target.write_bytes(file_info["content"])
    return target


def json_response(handler, status, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def file_response(handler, path):
    path = Path(path).resolve()
    if not path.exists() or not path.is_file() or not is_relative_to(path, ROOT):
        handler.send_error(404)
        return

    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "ResNetDemo/1.0"

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/":
            self.send_html()
            return
        if path == "/api/status":
            self.send_status()
            return
        if path == "/api/wardrobe":
            self.send_wardrobe()
            return
        if path == "/api/recommend":
            self.send_recommendation(query)
            return
        if path.startswith("/sample/"):
            name = urllib.parse.unquote(path.removeprefix("/sample/"))
            file_response(self, TEST_IMAGES_DIR / safe_filename(name))
            return
        if path == "/file":
            relative = query.get("path", [""])[0]
            file_response(self, ROOT / relative)
            return

        self.send_error(404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/predict":
            self.handle_predict()
            return
        if path == "/api/save-wardrobe":
            self.handle_save_wardrobe()
            return
        if path == "/api/delete-wardrobe":
            self.handle_delete_wardrobe()
            return

        self.send_error(404)

    def handle_predict(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            fields, files = parse_multipart(self.headers, body)

            image_path = None
            image_upload = save_upload(files.get("image"), "image")
            if image_upload:
                image_path = image_upload
            else:
                raise ValueError("Upload an image before running prediction.")

            result = RUNNER.predict(image_path, DEFAULT_CHECKPOINT, DEFAULT_LABELS)

            payload = {
                "ok": True,
                "input": {
                    "name": image_path.name,
                    "path": str(image_path),
                    "url": path_to_url(image_path),
                },
                "result": result,
                "outfit": None,
            }
            json_response(self, 200, payload)
        except Exception as exc:
            json_response(
                self,
                400,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=4),
                },
            )

    def handle_save_wardrobe(self):
        try:
            payload = self.read_json_body()
            source_path = Path(payload["image_path"])
            if not source_path.is_absolute():
                source_path = ROOT / source_path
            if not source_path.exists() or not is_relative_to(source_path, ROOT):
                raise FileNotFoundError(f"Image file is not available: {source_path}")

            relative_image_path = save_uploaded_image(source_path)
            item = add_wardrobe_item(
                relative_image_path,
                category=payload["category"],
                style=payload.get("style"),
                color=payload.get("color"),
                model_name=payload.get("model_name", "resnet"),
                confidence=payload.get("confidence"),
            )

            json_response(
                self,
                200,
                {
                    "ok": True,
                    "item": serialize_wardrobe_item(item),
                    "items": serialize_wardrobe_items(get_all_wardrobe_items()),
                },
            )
        except Exception as exc:
            json_response(
                self,
                400,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=4),
                },
            )

    def handle_delete_wardrobe(self):
        try:
            payload = self.read_json_body()
            deleted = delete_wardrobe_item(int(payload["id"]))
            json_response(
                self,
                200,
                {
                    "ok": True,
                    "deleted": deleted,
                    "items": serialize_wardrobe_items(get_all_wardrobe_items()),
                },
            )
        except Exception as exc:
            json_response(
                self,
                400,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=4),
                },
            )

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def send_wardrobe(self):
        json_response(
            self,
            200,
            {
                "ok": True,
                "items": serialize_wardrobe_items(get_all_wardrobe_items()),
            },
        )

    def send_recommendation(self, query):
        try:
            item_id = int(query.get("item_id", ["0"])[0])
            selected_item = get_wardrobe_item_by_id(item_id)
            if not selected_item:
                raise ValueError(f"Wardrobe item not found: {item_id}")

            recommendation = recommend_outfit_from_wardrobe(selected_item)
            json_response(
                self,
                200,
                {
                    "ok": True,
                    "recommendation": {
                        slot: serialize_wardrobe_item(item)
                        for slot, item in recommendation.items()
                    },
                },
            )
        except Exception as exc:
            json_response(
                self,
                400,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=4),
                },
            )

    def send_status(self):
        payload = {
            "model_ready": DEFAULT_CHECKPOINT.exists() and DEFAULT_LABELS.exists(),
            "wardrobe_count": len(get_all_wardrobe_items()),
        }
        json_response(self, 200, payload)

    def send_html(self):
        data = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


INDEX_HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ResNet18 Wardrobe Demo</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --warn: #b45309;
      --error: #b42318;
      --soft: #ecfdf5;
      --shadow: 0 18px 50px rgba(24, 39, 75, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
    }

    aside {
      background: #ffffff;
      border-right: 1px solid var(--line);
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    main {
      padding: 24px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 18px;
    }

    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.15;
      font-weight: 760;
    }

    h2 {
      margin: 0 0 12px;
      font-size: 15px;
      font-weight: 740;
    }

    label {
      display: block;
      margin-bottom: 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    input, select, button {
      width: 100%;
      font: inherit;
    }

    input[type="text"], select, input[type="file"] {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 10px 11px;
      color: var(--ink);
    }

    input[type="file"] {
      font-size: 13px;
    }

    button {
      border: 0;
      background: var(--accent);
      color: #fff;
      border-radius: 8px;
      min-height: 44px;
      padding: 10px 14px;
      cursor: pointer;
      font-weight: 740;
    }

    button:hover { background: var(--accent-strong); }
    button:disabled { opacity: 0.58; cursor: progress; }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .control-panel {
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 15px;
      box-shadow: none;
    }

    .status {
      border-radius: 8px;
      padding: 11px 12px;
      font-size: 13px;
      line-height: 1.45;
      border: 1px solid var(--line);
      background: #f8fafc;
      color: var(--muted);
      word-break: break-word;
    }

    .status.ready {
      color: #065f46;
      background: var(--soft);
      border-color: #a7f3d0;
    }

    .status.missing {
      color: var(--warn);
      background: #fffbeb;
      border-color: #fde68a;
    }

    .stage {
      display: grid;
      grid-template-columns: minmax(280px, 420px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }

    .preview {
      padding: 14px;
    }

    .preview-frame {
      width: 100%;
      aspect-ratio: 1 / 1;
      background: #eef2f7;
      border: 1px solid var(--line);
      border-radius: 8px;
      display: grid;
      place-items: center;
      overflow: hidden;
    }

    .preview-frame img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #fff;
    }

    .placeholder {
      color: var(--muted);
      font-size: 14px;
    }

    .results {
      padding: 18px;
      min-height: 420px;
    }

    .result-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
      min-height: 96px;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    .metric strong {
      display: block;
      font-size: 22px;
      line-height: 1.15;
      overflow-wrap: anywhere;
    }

    .bar-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin: 16px 0;
    }

    .bar-row {
      display: grid;
      grid-template-columns: minmax(90px, 145px) minmax(0, 1fr) 52px;
      gap: 10px;
      align-items: center;
      margin: 9px 0;
      font-size: 13px;
    }

    .bar-label {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .bar-track {
      height: 8px;
      background: #edf1f6;
      border-radius: 999px;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      width: 0;
      background: var(--accent);
    }

    .outfit {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }

    .item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }

    .item.selectable {
      cursor: pointer;
    }

    .item.selected {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14);
    }

    .item-image {
      aspect-ratio: 1 / 1;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #f3f5f8;
      display: grid;
      place-items: center;
      overflow: hidden;
      margin-bottom: 8px;
    }

    .item-image img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #fff;
    }

    .item strong {
      display: block;
      font-size: 13px;
      margin-bottom: 4px;
    }

    .item small {
      display: block;
      color: var(--muted);
      overflow-wrap: anywhere;
    }

    .error {
      color: var(--error);
      background: #fef3f2;
      border: 1px solid #fecdca;
      border-radius: 8px;
      padding: 12px;
      white-space: pre-wrap;
      line-height: 1.45;
    }

    .meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }

    .hidden { display: none !important; }

    .tab-bar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .tab-button {
      width: auto;
      min-height: 38px;
      background: #ffffff;
      color: var(--ink);
      border: 1px solid var(--line);
      padding: 8px 12px;
    }

    .tab-button.active {
      background: var(--accent);
      color: #ffffff;
      border-color: var(--accent);
    }

    .tab-page {
      display: none;
    }

    .tab-page.active {
      display: block;
    }

    .secondary {
      background: #ffffff;
      color: var(--accent);
      border: 1px solid var(--accent);
    }

    .secondary:hover {
      background: var(--soft);
    }

    .danger {
      background: #ffffff;
      color: var(--error);
      border: 1px solid #fecdca;
    }

    .danger:hover {
      background: #fef3f2;
    }

    .actions {
      display: flex;
      gap: 10px;
      margin-top: 12px;
    }

    .wardrobe-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 12px;
    }

    .select-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      margin-bottom: 14px;
    }

    .notice {
      border: 1px solid #fde68a;
      background: #fffbeb;
      color: var(--warn);
      border-radius: 8px;
      padding: 10px;
      font-size: 13px;
      margin-bottom: 8px;
    }

    @media (max-width: 960px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .stage { grid-template-columns: 1fr; }
      .result-grid, .bar-list, .outfit { grid-template-columns: 1fr; }
      .select-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div>
        <h1>ResNet18 Wardrobe Demo</h1>
      </div>

      <div id="status" class="status">Checking model files...</div>

      <form id="predictForm" class="panel control-panel">
        <div>
          <label for="image">Upload Image</label>
          <input id="image" name="image" type="file" accept="image/*" />
        </div>
        <button id="runButton" type="submit">Run ResNet18</button>
        <button id="saveButton" class="secondary hidden" type="button">Save to Wardrobe</button>
      </form>
    </aside>

    <main>
      <nav class="tab-bar">
        <button class="tab-button active" type="button" data-tab="uploadTab">Upload & Predict</button>
        <button class="tab-button" type="button" data-tab="wardrobeTab">My Wardrobe</button>
        <button class="tab-button" type="button" data-tab="recommendTab">Outfit Recommendation</button>
      </nav>

      <section id="uploadTab" class="tab-page active">
      <div class="stage">
        <div class="panel preview">
          <h2>Input</h2>
          <div class="preview-frame">
            <img id="previewImage" class="hidden" alt="Input preview" />
            <div id="previewText" class="placeholder">No image selected</div>
          </div>
        </div>

        <div class="panel results">
          <h2>Prediction</h2>
          <div id="emptyState" class="meta">Results will appear here after inference.</div>
          <div id="errorBox" class="error hidden"></div>
          <div id="resultView" class="hidden">
            <div class="result-grid">
              <div class="metric">
                <span>Category</span>
                <strong id="categoryValue">-</strong>
                <div id="categoryConfidence" class="meta"></div>
              </div>
              <div class="metric">
                <span>Style</span>
                <strong id="styleValue">-</strong>
                <div id="styleConfidence" class="meta"></div>
              </div>
              <div class="metric">
                <span>Device</span>
                <strong id="deviceValue">-</strong>
              </div>
              <div class="metric">
                <span>Inference</span>
                <strong id="timeValue">-</strong>
              </div>
            </div>

            <div class="bar-list">
              <div>
                <h2>Top Categories</h2>
                <div id="categoryBars"></div>
              </div>
              <div>
                <h2>Top Styles</h2>
                <div id="styleBars"></div>
              </div>
            </div>

          </div>
        </div>
      </div>
      </section>

      <section id="wardrobeTab" class="tab-page">
        <div class="panel results">
          <h2>My Wardrobe</h2>
          <div id="wardrobeNotice" class="meta">Loading wardrobe...</div>
          <div id="wardrobeGrid" class="wardrobe-grid"></div>
        </div>
      </section>

      <section id="recommendTab" class="tab-page">
        <div class="panel results">
          <h2>Outfit Recommendation</h2>
          <div id="recommendNotice" class="meta">Choose an item from My Wardrobe.</div>
          <div id="recommendPicker" class="wardrobe-grid"></div>
          <div class="actions">
            <button id="recommendButton" type="button" disabled>Recommend Selected Item</button>
          </div>
          <div id="recommendGrid" class="outfit"></div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const form = document.getElementById("predictForm");
    const statusBox = document.getElementById("status");
    const imageInput = document.getElementById("image");
    const runButton = document.getElementById("runButton");
    const saveButton = document.getElementById("saveButton");
    const previewImage = document.getElementById("previewImage");
    const previewText = document.getElementById("previewText");
    const emptyState = document.getElementById("emptyState");
    const errorBox = document.getElementById("errorBox");
    const resultView = document.getElementById("resultView");
    const wardrobeGrid = document.getElementById("wardrobeGrid");
    const wardrobeNotice = document.getElementById("wardrobeNotice");
    const recommendPicker = document.getElementById("recommendPicker");
    const recommendButton = document.getElementById("recommendButton");
    const recommendGrid = document.getElementById("recommendGrid");
    const recommendNotice = document.getElementById("recommendNotice");

    let lastPrediction = null;
    let wardrobeItems = [];
    let selectedRecommendId = null;

    function pct(value) {
      return `${Math.round(value * 1000) / 10}%`;
    }

    function setPreview(url) {
      if (!url) {
        previewImage.classList.add("hidden");
        previewText.classList.remove("hidden");
        return;
      }
      previewImage.src = url;
      previewImage.classList.remove("hidden");
      previewText.classList.add("hidden");
    }

    function renderBars(targetId, rows) {
      const target = document.getElementById(targetId);
      target.innerHTML = "";
      rows.forEach((row) => {
        const div = document.createElement("div");
        div.className = "bar-row";
        div.innerHTML = `
          <div class="bar-label" title="${escapeHtml(row.label)}">${escapeHtml(row.label)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(1, row.score * 100)}%"></div></div>
          <div>${pct(row.score)}</div>
        `;
        target.appendChild(div);
      });
    }

    function renderResult(payload) {
      const result = payload.result;
      lastPrediction = payload;
      emptyState.classList.add("hidden");
      errorBox.classList.add("hidden");
      resultView.classList.remove("hidden");
      saveButton.classList.remove("hidden");

      document.getElementById("categoryValue").textContent = result.category;
      document.getElementById("styleValue").textContent = result.style;
      document.getElementById("categoryConfidence").textContent = pct(result.category_confidence);
      document.getElementById("styleConfidence").textContent = pct(result.style_confidence);
      document.getElementById("deviceValue").textContent = result.device;
      document.getElementById("timeValue").textContent = `${result.inference_time.toFixed(4)}s`;

      renderBars("categoryBars", result.top_categories);
      renderBars("styleBars", result.top_styles);
      setPreview(payload.input.url);
    }

    function renderError(message) {
      emptyState.classList.add("hidden");
      resultView.classList.add("hidden");
      saveButton.classList.add("hidden");
      errorBox.textContent = message;
      errorBox.classList.remove("hidden");
    }

    function renderMessage(message) {
      emptyState.classList.remove("hidden");
      emptyState.textContent = message;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function itemCard(item, options = {}) {
      const warning = item.image_exists ? "" : `<div class="notice">Image file is missing.</div>`;
      const image = item.image_url
        ? `<img src="${item.image_url}" alt="${escapeHtml(item.category)}" />`
        : `<span class="meta">No image</span>`;
      const confidence = item.confidence == null ? "" : ` · ${pct(item.confidence)}`;
      const deleteButton = options.deletable
        ? `<button class="danger" type="button" data-delete-id="${item.id}">Delete</button>`
        : "";
      const selectable = options.selectable ? " selectable" : "";
      const selected = options.selected ? " selected" : "";
      const selectAttr = options.selectable ? ` data-select-id="${item.id}"` : "";

      return `
        <div class="item${selectable}${selected}"${selectAttr}>
          ${warning}
          <div class="item-image">${image}</div>
          <strong>${escapeHtml(item.category || "-")}</strong>
          <small>${escapeHtml(item.style || "No style")}${confidence}</small>
          ${deleteButton ? `<div class="actions">${deleteButton}</div>` : ""}
        </div>
      `;
    }

    function renderWardrobe(items) {
      wardrobeItems = items || [];
      wardrobeGrid.innerHTML = "";
      recommendPicker.innerHTML = "";

      if (!wardrobeItems.length) {
        wardrobeNotice.textContent = "No saved wardrobe items yet.";
        recommendNotice.textContent = "Save at least one item before recommending outfits.";
        selectedRecommendId = null;
        recommendButton.disabled = true;
        recommendGrid.innerHTML = "";
        return;
      }

      wardrobeNotice.textContent = `${wardrobeItems.length} saved item(s).`;
      wardrobeGrid.innerHTML = wardrobeItems.map((item) => itemCard(item, { deletable: true })).join("");
      renderRecommendPicker();
      recommendNotice.textContent = selectedRecommendId
        ? "Selected item is ready for recommendation."
        : "Choose an item image below.";
    }

    function renderRecommendPicker() {
      recommendPicker.innerHTML = wardrobeItems
        .map((item) => itemCard(item, {
          selectable: true,
          selected: Number(item.id) === Number(selectedRecommendId)
        }))
        .join("");
      recommendButton.disabled = !selectedRecommendId;
    }

    async function loadWardrobe() {
      const response = await fetch("/api/wardrobe");
      const payload = await response.json();
      if (!payload.ok) throw new Error(payload.error || "Cannot load wardrobe.");
      renderWardrobe(payload.items);
    }

    async function saveCurrentPrediction() {
      if (!lastPrediction) {
        renderError("Run prediction before saving to wardrobe.");
        return;
      }

      saveButton.disabled = true;
      saveButton.textContent = "Saving...";
      try {
        const response = await fetch("/api/save-wardrobe", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_path: lastPrediction.input.path,
            model_name: "resnet",
            category: lastPrediction.result.category,
            style: lastPrediction.result.style,
            confidence: lastPrediction.result.category_confidence
          })
        });
        const payload = await response.json();
        if (!payload.ok) throw new Error(payload.error);

        renderWardrobe(payload.items);
        showTab("wardrobeTab");
      } catch (error) {
        renderError(error.message);
      } finally {
        saveButton.disabled = false;
        saveButton.textContent = "Save to Wardrobe";
      }
    }

    async function deleteWardrobeItem(id) {
      const response = await fetch("/api/delete-wardrobe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id })
      });
      const payload = await response.json();
      if (!payload.ok) throw new Error(payload.error);
      if (Number(selectedRecommendId) === Number(id)) {
        selectedRecommendId = null;
        recommendGrid.innerHTML = "";
      }
      renderWardrobe(payload.items);
    }

    function renderRecommendation(recommendation) {
      const slots = ["top", "bottom", "shoes"];
      recommendGrid.innerHTML = slots
        .filter((slot) => recommendation[slot])
        .map((slot) => {
          const item = recommendation[slot];
          return `
            <div>
              <h2>${escapeHtml(slot.toUpperCase())}</h2>
              ${itemCard(item)}
            </div>
          `;
        })
        .join("");

      if (!recommendGrid.innerHTML) {
        recommendNotice.textContent = "No compatible outfit item found.";
      } else {
        recommendNotice.textContent = "Recommendation uses saved wardrobe items only.";
      }
    }

    async function runRecommendation() {
      if (!selectedRecommendId) {
        recommendNotice.textContent = "Select an item image first.";
        return;
      }

      recommendButton.disabled = true;
      recommendButton.textContent = "Finding...";
      try {
        const response = await fetch(`/api/recommend?item_id=${encodeURIComponent(selectedRecommendId)}`);
        const payload = await response.json();
        if (!payload.ok) throw new Error(payload.error);
        renderRecommendation(payload.recommendation);
      } catch (error) {
        recommendNotice.textContent = error.message;
      } finally {
        recommendButton.disabled = false;
        recommendButton.textContent = "Recommend";
      }
    }

    function showTab(tabId) {
      document.querySelectorAll(".tab-page").forEach((page) => {
        page.classList.toggle("active", page.id === tabId);
      });
      document.querySelectorAll(".tab-button").forEach((button) => {
        button.classList.toggle("active", button.dataset.tab === tabId);
      });

      if (tabId === "wardrobeTab" || tabId === "recommendTab") {
        loadWardrobe().catch((error) => {
          wardrobeNotice.textContent = error.message;
          recommendNotice.textContent = error.message;
        });
      }
    }

    async function loadStatus() {
      const response = await fetch("/api/status");
      const status = await response.json();

      if (status.model_ready) {
        statusBox.className = "status ready";
        statusBox.textContent = `Ready · Wardrobe: ${status.wardrobe_count} item(s)`;
      } else {
        statusBox.className = "status missing";
        statusBox.textContent = "Model files are missing. Please add the trained ResNet weights before predicting.";
      }
    }

    document.querySelectorAll(".tab-button").forEach((button) => {
      button.addEventListener("click", () => showTab(button.dataset.tab));
    });

    imageInput.addEventListener("change", () => {
      const file = imageInput.files[0];
      if (file) {
        setPreview(URL.createObjectURL(file));
      }
    });

    saveButton.addEventListener("click", saveCurrentPrediction);

    wardrobeGrid.addEventListener("click", async (event) => {
      const button = event.target.closest("[data-delete-id]");
      if (!button) return;

      try {
        await deleteWardrobeItem(Number(button.dataset.deleteId));
      } catch (error) {
        wardrobeNotice.textContent = error.message;
      }
    });

    recommendPicker.addEventListener("click", (event) => {
      const card = event.target.closest("[data-select-id]");
      if (!card) return;

      selectedRecommendId = Number(card.dataset.selectId);
      recommendGrid.innerHTML = "";
      recommendNotice.textContent = "Selected item is ready for recommendation.";
      renderRecommendPicker();
    });

    recommendButton.addEventListener("click", runRecommendation);

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const data = new FormData(form);

      if (!imageInput.files.length) {
        renderError("Upload an image before running prediction.");
        return;
      }

      runButton.disabled = true;
      runButton.textContent = "Running...";

      try {
        const response = await fetch("/api/predict", { method: "POST", body: data });
        const payload = await response.json();
        if (!payload.ok) {
          renderError(payload.error);
        } else {
          renderResult(payload);
        }
      } catch (error) {
        renderError(error.message);
      } finally {
        runButton.disabled = false;
        runButton.textContent = "Run ResNet18";
      }
    });

    loadStatus().catch((error) => renderError(error.message));
    loadWardrobe().catch((error) => {
      wardrobeNotice.textContent = error.message;
    });
  </script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ResNet18 wardrobe demo UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()
    initialize_database()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"ResNet18 demo running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
