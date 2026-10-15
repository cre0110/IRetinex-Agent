import base64
import hashlib
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, render_template, request, send_file, url_for

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "static" / "captures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 方式 1：直接在这里填写你项目里已经上传好的视频路径
# Windows 示例：DEFAULT_VIDEO_PATH = r"D:\\MedicalAI\\视频\\case001.mp4"
# Linux 示例：  DEFAULT_VIDEO_PATH = "/mnt/data/videos/case001.mp4"
#
# 方式 2：启动时通过环境变量传入
# Windows PowerShell：$env:VIDEO_PATH="D:\\MedicalAI\\视频\\case001.mp4"; python app.py
# Linux / macOS：   VIDEO_PATH="/mnt/data/videos/case001.mp4" python app.py
#
# 方式 3：测试时在浏览器 URL 后面临时传入
# http://127.0.0.1:5000/?video_path=D:/MedicalAI/videos/case001.mp4
# ============================================================
DEFAULT_VIDEO_PATH = r"D:\MedicalAI\视频\colon_demo.mp4"

# 简单的内存映射：页面拿到 token，后端根据 token 找到真实视频路径。
# 这样前端不需要直接暴露完整本地路径。
VIDEO_REGISTRY = {}


def normalize_video_path(path_text: str) -> Optional[Path]:
    """把传入的视频路径转成绝对路径。支持绝对路径和相对项目根目录的路径。"""
    if not path_text:
        return None

    path_text = path_text.strip().strip('"').strip("'")
    if not path_text:
        return None

    video_path = Path(path_text)
    if not video_path.is_absolute():
        video_path = BASE_DIR / video_path

    return video_path.resolve()


def get_requested_video_path() -> Optional[Path]:
    """优先级：URL 参数 > 环境变量 VIDEO_PATH > app.py 中的 DEFAULT_VIDEO_PATH。"""
    url_video_path = request.args.get("video_path", "")
    env_video_path = os.environ.get("VIDEO_PATH", "")
    selected_path = url_video_path or env_video_path or DEFAULT_VIDEO_PATH
    return normalize_video_path(selected_path)


def register_video(video_path: Path) -> str:
    """为视频路径生成 token，供 /video/<token> 使用。"""
    token_source = f"{video_path}|{time.time()}"
    token = hashlib.sha256(token_source.encode("utf-8")).hexdigest()[:16]
    VIDEO_REGISTRY[token] = str(video_path)
    return token


@app.route("/")
def index():
    video_path = get_requested_video_path()
    video_exists = bool(video_path and video_path.exists() and video_path.is_file())

    video_url = ""
    video_name = ""
    video_path_display = "未设置视频路径"

    if video_path:
        video_path_display = str(video_path)
        video_name = video_path.name

    if video_exists:
        token = register_video(video_path)
        video_url = url_for("serve_video", token=token)

    return render_template(
        "index.html",
        video_url=video_url,
        video_name=video_name,
        video_path_display=video_path_display,
        video_exists=video_exists,
    )


@app.route("/video/<token>")
def serve_video(token):
    """把服务器上的既有视频文件喂给浏览器播放。conditional=True 支持拖动进度条/Range 请求。"""
    video_path_text = VIDEO_REGISTRY.get(token)
    if not video_path_text:
        abort(404)

    video_path = Path(video_path_text)
    if not video_path.exists() or not video_path.is_file():
        abort(404)

    mimetype = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    return send_file(video_path, mimetype=mimetype, conditional=True)


@app.route("/save_frame", methods=["POST"])
def save_frame():
    """接收前端 canvas 生成的 base64 图片，并保存为 PNG。"""
    data = request.get_json(silent=True) or {}
    image_data = data.get("image", "")
    video_time = data.get("video_time", 0)

    if not image_data:
        return jsonify({"success": False, "message": "没有接收到图像数据"}), 400

    match = re.match(r"^data:image/(png|jpeg|jpg);base64,(.+)$", image_data)
    if not match:
        return jsonify({"success": False, "message": "图像格式不正确"}), 400

    image_base64 = match.group(2)

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        return jsonify({"success": False, "message": "base64 解码失败"}), 400

    timestamp_ms = int(time.time() * 1000)
    try:
        video_time_float = float(video_time)
    except Exception:
        video_time_float = 0.0

    filename = f"frame_{video_time_float:.2f}s_{timestamp_ms}.png"
    save_path = CAPTURE_DIR / filename

    with open(save_path, "wb") as f:
        f.write(image_bytes)

    image_url = url_for("static", filename=f"captures/{filename}")

    return jsonify({
        "success": True,
        "filename": filename,
        "image_url": image_url,
        "video_time": round(video_time_float, 2),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
