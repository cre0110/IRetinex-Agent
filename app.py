from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
import shutil
import sys
import time
from uuid import uuid4

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import requests
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory, stream_with_context, url_for
from PIL import Image
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from agent_workflow import (
    EndoscopyAnalysisAgent,
    FollowupAgent,
    REPORT_REVISION_CONFIDENCE_THRESHOLD,
    REPORT_REVISION_SCORE_THRESHOLD,
    build_analysis_response,
    create_initial_state,
)

try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is optional at runtime.
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent


def configure_console_debug_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for logger_name in ("werkzeug", "urllib3", "PIL"):
        logging.getLogger(logger_name).setLevel(logging.INFO)


configure_console_debug_logging()


def load_env_file(env_path: Path, *, override: bool = False) -> bool:
    if not env_path.exists():
        return False

    loaded = False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded = True

    return loaded


if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR / ".env.local", override=True)
else:
    load_env_file(BASE_DIR / ".env")
    load_env_file(BASE_DIR / ".env.local", override=True)


UPLOAD_ROOT = BASE_DIR / "uploads"
OUTPUT_ROOT = BASE_DIR / "outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen3-vl-32b-instruct"
CONFIDENCE_CHECK_MODEL = DEFAULT_MODEL
CONFIDENCE_CHECK_DISPLAY_MODEL = "z-ai/glm-4.6v"
SERVICE_UNAVAILABLE_MESSAGE = "分析服务暂时不可用，请稍后重试。"
MANUAL_FRAME_MAX_WAIT_SECONDS = int(os.getenv("MEDICALAI_MANUAL_FRAME_MAX_WAIT_SECONDS", "600"))
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge_base.json"
LESION_TYPE_TO_DISEASE_TYPE = {
    "polyp": "colorectal_polyp",
    "barrett": "barrett_esophagus",
}
VALID_LESION_TYPES = set(LESION_TYPE_TO_DISEASE_TYPE)
LESION_TYPE_ALIASES = {
    "polyp": "polyp",
    "colon": "polyp",
    "colon_polyp": "polyp",
    "colorectal_polyp": "polyp",
    "barrett": "barrett",
    "be": "barrett",
    "upper_gi": "barrett",
    "upper_gi_be": "barrett",
    "barrett_esophagus": "barrett",
}

UNSUPPORTED_REPORT_METADATA_PREFIXES = (
    "分析完成时间",
    "报告生成时间",
    "生成时间",
    "系统版本",
    "处理设备",
    "模型来源",
    "数据来源",
    "安全等级",
    "安全级别",
)

UNSUPPORTED_REPORT_METADATA_PATTERN = re.compile(
    r"^\s*(?:" + "|".join(re.escape(item) for item in UNSUPPORTED_REPORT_METADATA_PREFIXES) + r")\s*[:：]"
)


def sanitize_generated_report(report: str) -> str:
    """Remove unsupported audit-style metadata that the LLM may invent."""
    if not report:
        return report

    cleaned_lines: list[str] = []
    for raw_line in report.splitlines():
        line = raw_line.strip()
        if UNSUPPORTED_REPORT_METADATA_PATTERN.match(line):
            continue
        if line in {"---", "——", "—"}:
            continue
        if line.startswith(("注：", "备注：")) and any(term in line for term in ("严格遵循", "符合医疗AI安全约束", "安全约束")):
            continue
        cleaned_lines.append(raw_line)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def ensure_required_report_sections(report: str) -> str:
    """Keep the final assistant report aligned with the required three clinical sections."""
    cleaned = sanitize_generated_report(report)
    if not cleaned:
        return cleaned

    required_headers = ("内镜所见：", "内镜诊断：", "医生复诊建议：")
    if all(header in cleaned for header in required_headers):
        return cleaned

    return "\n\n".join(
        [
            f"内镜所见：{cleaned}",
            "内镜诊断：疑似内镜异常改变？",
            "医生复诊建议：建议由内镜医生复核原始图像/视频及标注区域，必要时复查内镜并结合活检病理等检查。仅供辅助参考，不能替代医生诊断。",
        ]
    )


def build_brief_local_diagnosis(vision_result: dict | None = None, lesion_type: str | None = None) -> str:
    vision_result = vision_result or {}
    normalized_type = normalize_requested_lesion_type(lesion_type or vision_result.get("lesion_type"))
    polyp_detected = bool(vision_result.get("polyp_detected"))

    if normalized_type == "barrett":
        items = ["Barrett食管？", "食管下段黏膜异常改变？"]
    elif normalized_type == "polyp":
        items = ["结肠息肉？", "结直肠黏膜隆起性病变？"]
    else:
        items = ["疑似内镜异常改变？"]

    if not polyp_detected:
        items = ["未见明确疑似病灶？", *items[:1]]

    return "\n".join(f"{index}、{item}" for index, item in enumerate(items[:3], start=1))


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
app.logger.setLevel(logging.DEBUG)

# Local in-memory context is enough for the desktop demo. Restarting Flask clears sessions.
ANALYSIS_CONTEXTS: dict[str, dict] = {}
KNOWLEDGE_BASE_CACHE: list[dict] | None = None


def get_default_dashscope_model() -> str:
    return os.getenv("DASHSCOPE_MODEL", DEFAULT_MODEL)


def get_step_model(step_type: str, title: str) -> str | None:
    if step_type == "classification" or "预检" in title:
        return get_default_dashscope_model()
    if step_type == "analysis" or "结构化病灶分析" in title:
        return get_default_dashscope_model()
    if step_type == "report" or "报告生成" in title:
        return get_default_dashscope_model()
    if step_type == "reflection" or "自检" in title or "置信度" in title or "反思" in title:
        return CONFIDENCE_CHECK_DISPLAY_MODEL
    return None


def is_api_request() -> bool:
    return request.path in {"/analyze", "/analyze-stream", "/chat", "/manual-video-frame", "/manual-video-frames/finish"}


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(exc):
    if is_api_request():
        return jsonify(
            {
                "ok": False,
                "error": "上传图片总大小超过 80MB，请减少图片数量或压缩后再发送。",
                "steps": [make_step("error", "图片过大", "上传图片总大小超过 80MB，我无法接收这批图片。")],
            }
        ), 413
    return exc


@app.errorhandler(HTTPException)
def handle_http_exception(exc):
    if is_api_request():
        return jsonify({"ok": False, "error": exc.description or exc.name, "steps": []}), exc.code
    return exc


@app.errorhandler(Exception)
def handle_unexpected_exception(exc):
    if is_api_request():
        app.logger.exception("Unhandled API error: %s", exc)
        return jsonify({"ok": False, "error": SERVICE_UNAVAILABLE_MESSAGE, "steps": []}), 500
    raise exc


class DashScopeError(RuntimeError):
    def __init__(self, user_message: str, technical_message: str | None = None):
        super().__init__(technical_message or user_message)
        self.user_message = user_message
        self.technical_message = technical_message or user_message


def log_debug(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
    app.logger.debug(message)


def log_task(task_name: str) -> None:
    log_debug(f"========== 切换任务：{task_name} ==========")


def log_json(title: str, data) -> None:
    try:
        formatted = json.dumps(data, ensure_ascii=False, indent=2)
    except TypeError:
        formatted = str(data)
    log_debug(f"{title}:\n{formatted}")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def is_video_upload_mode() -> bool:
    return str(request.form.get("upload_mode") or "").strip().lower() == "video"


def normalize_requested_lesion_type(value: str | None) -> str:
    return LESION_TYPE_ALIASES.get(str(value or "polyp").strip().lower(), "polyp")


def validate_image(image_path: Path) -> bool:
    log_debug(f"开始校验图片文件：{image_path}")
    try:
        with Image.open(image_path) as image:
            image.verify()
        log_debug("图片文件校验通过")
        return True
    except Exception as exc:
        log_debug(f"图片文件校验失败：{exc}")
        return False


def encode_image_data_url(image_path: Path) -> str:
    log_debug(f"开始编码图片为 DashScope data URL：{image_path}")
    mime_type, _ = mimetypes.guess_type(str(image_path))
    supported_mime_types = {"image/png", "image/jpeg", "image/webp", "image/gif"}
    if mime_type in supported_mime_types:
        with image_path.open("rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
        log_debug(f"图片编码完成：MIME={mime_type}，base64 长度={len(encoded)}")
        return f"data:{mime_type};base64,{encoded}"

    buffer = BytesIO()
    with Image.open(image_path) as image:
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    log_debug(f"图片已转为 JPEG 后编码完成，原始 MIME={mime_type}，base64 长度={len(encoded)}")
    return f"data:image/jpeg;base64,{encoded}"


def build_dashscope_payload(
    messages: list[dict],
    *,
    max_tokens: int,
    json_mode: bool,
    model: str | None = None,
) -> dict:
    payload = {
        "model": model or get_default_dashscope_model(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def build_dashscope_request_kwargs(payload: dict, headers: dict) -> dict:
    request_kwargs = {
        "headers": headers,
        "json": payload,
        "timeout": 90,
    }
    http_proxy = os.getenv("DASHSCOPE_HTTP_PROXY")
    https_proxy = os.getenv("DASHSCOPE_HTTPS_PROXY")
    if http_proxy or https_proxy:
        request_kwargs["proxies"] = {
            "http": http_proxy,
            "https": https_proxy or http_proxy,
        }
    return request_kwargs


def extract_message_content(data: dict) -> str:
    content = data["choices"][0]["message"].get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(text_parts).strip()
    return str(content).strip()


def format_dashscope_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]

    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    if error:
        return str(error)
    return str(payload)[:500]


def dashscope_chat(
    messages: list[dict],
    *,
    max_tokens: int = 700,
    json_mode: bool = False,
    task_name: str = "DashScope 调用",
    model: str | None = None,
) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        log_debug(f"{task_name} 取消：未读取到 DASHSCOPE_API_KEY")
        raise DashScopeError(
            "未配置 DashScope API Key。请在项目根目录创建 .env，写入 DASHSCOPE_API_KEY=你的密钥，然后重启 Flask。",
            "DASHSCOPE_API_KEY is not configured",
        )

    payload = build_dashscope_payload(messages, max_tokens=max_tokens, json_mode=json_mode, model=model)
    model_name = payload["model"]
    log_task(f"开始调用 AI：{task_name}")
    log_debug(
        f"DashScope 请求准备完成：model={model_name}, json_mode={json_mode}, "
        f"max_tokens={max_tokens}, message_count={len(messages)}"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_kwargs = build_dashscope_request_kwargs(payload, headers)

    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(DASHSCOPE_URL, **request_kwargs)
    except requests.RequestException as exc:
        log_debug(f"{task_name} 网络请求失败：{exc}")
        raise DashScopeError("无法连接 DashScope，请检查网络或代理设置后重试。", str(exc)) from exc

    log_debug(f"{task_name} 收到 DashScope HTTP 状态码：{response.status_code}")
    if response.status_code in {400, 422} and json_mode:
        log_debug(f"{task_name} JSON response_format 可能不被当前模型支持，准备移除 response_format 后重试")
        log_debug(f"{task_name} 首次错误响应内容：{response.text[:2000]}")
        payload = build_dashscope_payload(messages, max_tokens=max_tokens, json_mode=False, model=model)
        request_kwargs = build_dashscope_request_kwargs(payload, headers)
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(DASHSCOPE_URL, **request_kwargs)
        except requests.RequestException as exc:
            log_debug(f"{task_name} 降级重试网络请求失败：{exc}")
            raise DashScopeError("无法连接 DashScope，请检查网络或代理设置后重试。", str(exc)) from exc
        log_debug(f"{task_name} 降级重试后 HTTP 状态码：{response.status_code}")

    if response.status_code == 401:
        log_debug(f"{task_name} AI 错误响应内容：{response.text[:2000]}")
        raise DashScopeError("DashScope API Key 无效或未被当前进程读取，请检查 .env 后重启 Flask。", response.text)
    if response.status_code == 402:
        log_debug(f"{task_name} AI 错误响应内容：{response.text[:2000]}")
        raise DashScopeError("DashScope 账户余额不足或未开通计费，请在 DashScope 控制台检查额度。", response.text)
    if response.status_code == 429:
        log_debug(f"{task_name} AI 错误响应内容：{response.text[:2000]}")
        raise DashScopeError("DashScope 请求过于频繁或达到额度限制，请稍后重试。", response.text)
    if response.status_code >= 400:
        detail = format_dashscope_error(response)
        log_debug(f"{task_name} AI 错误响应内容：{response.text[:2000]}")
        raise DashScopeError(f"DashScope 调用失败：{response.status_code}，{detail}", response.text)

    try:
        data = response.json()
        content = extract_message_content(data)
    except Exception as exc:
        raise DashScopeError("DashScope 返回内容无法解析，请稍后重试或更换 DASHSCOPE_MODEL。", str(exc)) from exc

    if not content:
        log_debug(f"{task_name} AI 原始响应为空或未提取到文本：{response.text[:2000]}")
        raise DashScopeError("DashScope 返回了空内容，请稍后重试或更换 DASHSCOPE_MODEL。", response.text)
    log_debug(f"AI 给出输出（{task_name}）：\n{content}")
    return content


def parse_json_object(text: str) -> dict:
    log_debug("开始解析 AI JSON 输出")
    try:
        parsed = json.loads(text)
        log_json("AI JSON 解析结果", parsed)
        return parsed
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
        log_json("AI JSON 截取解析结果", parsed)
        return parsed


def _as_text_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def normalize_structured_report(value) -> dict:
    if isinstance(value, str):
        value = parse_json_object(value)
    if not isinstance(value, dict):
        raise ValueError("report JSON must be an object")

    findings = str(value.get("endoscopic_findings") or value.get("内镜所见") or "").strip()
    diagnosis_items = _as_text_list(value.get("endoscopic_diagnosis") or value.get("内镜诊断"))
    followup = str(value.get("followup_advice") or value.get("医生复诊建议") or "").strip()

    if not findings:
        findings = "当前结构化报告未提供明确内镜所见。"
    if not diagnosis_items:
        diagnosis_items = ["疑似内镜异常改变？"]
    normalized_diagnosis = []
    for item in diagnosis_items[:3]:
        item = re.sub(r"^\s*\d+\s*[、.．]\s*", "", item).strip()
        if item and not item.endswith(("?", "？")):
            item += "？"
        if item:
            normalized_diagnosis.append(item)
    if not normalized_diagnosis:
        normalized_diagnosis = ["疑似内镜异常改变？"]
    if not followup:
        followup = "建议由内镜医生复核原始图像/视频及标注区域，必要时复查内镜并结合活检病理等检查。仅供辅助参考，不能替代医生诊断。"

    return {
        "endoscopic_findings": findings,
        "endoscopic_diagnosis": normalized_diagnosis,
        "followup_advice": followup,
    }


def format_structured_report(report_json: dict) -> str:
    report = normalize_structured_report(report_json)
    diagnosis_text = "\n".join(
        f"{index}、{item}"
        for index, item in enumerate(report["endoscopic_diagnosis"], start=1)
    )
    return (
        f"内镜所见：{report['endoscopic_findings']}\n\n"
        f"内镜诊断：{diagnosis_text}\n\n"
        f"医生复诊建议：{report['followup_advice']}"
    )


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_revision_score_value(value) -> int:
    try:
        return max(0, min(10, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是", "需要"}
    return bool(value)


def should_revise_report_after_reflection(reflection: dict | None) -> bool:
    reflection = reflection or {}
    confidence = reflection.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = None

    explicit_request = bool(
        reflection.get("revise_report")
        or reflection.get("needs_report_revision")
        or reflection.get("report_revision_required")
    )
    low_confidence = confidence_value is not None and confidence_value < REPORT_REVISION_CONFIDENCE_THRESHOLD
    revision_score = _coerce_revision_score_value(
        reflection.get("report_revision_score", reflection.get("revision_score", reflection.get("rrs")))
    )
    return explicit_request or low_confidence or revision_score >= REPORT_REVISION_SCORE_THRESHOLD


def _normalize_bbox(raw_bbox, fallback_bbox=None) -> dict:
    source = raw_bbox if raw_bbox is not None else fallback_bbox
    if isinstance(source, dict):
        return {
            "x1": _coerce_int(source.get("x1")),
            "y1": _coerce_int(source.get("y1")),
            "x2": _coerce_int(source.get("x2")),
            "y2": _coerce_int(source.get("y2")),
        }
    if isinstance(source, (list, tuple)) and len(source) >= 4:
        return {
            "x1": _coerce_int(source[0]),
            "y1": _coerce_int(source[1]),
            "x2": _coerce_int(source[2]),
            "y2": _coerce_int(source[3]),
        }
    return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}


def normalize_lesion_analysis(raw: dict, vision_result: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("lesion analysis must be a JSON object")

    requested_lesion_type = normalize_requested_lesion_type(vision_result.get("lesion_type"))
    lesion_type = str(raw.get("lesion_type", "")).strip().lower()
    if lesion_type not in VALID_LESION_TYPES:
        lesion_type = requested_lesion_type
    elif vision_result.get("lesion_type"):
        lesion_type = requested_lesion_type

    confidence_level = str(raw.get("confidence_level", "")).strip().lower()
    if not confidence_level:
        raise ValueError("lesion analysis missing confidence_level")

    lesions_raw = raw.get("lesions")
    if not isinstance(lesions_raw, list):
        raise ValueError("lesion analysis missing lesions list")

    normalized_lesions = []
    fallback_bbox = vision_result.get("bbox")
    for index, lesion in enumerate(lesions_raw, start=1):
        if not isinstance(lesion, dict):
            continue
        normalized_lesions.append(
            {
                "lesion_id": str(lesion.get("lesion_id") or f"lesion_{index:03d}"),
                "location_description": str(lesion.get("location_description", "")).strip(),
                "anatomical_location": str(lesion.get("anatomical_location", "")).strip(),
                "bbox": _normalize_bbox(lesion.get("bbox"), fallback_bbox),
                "area_ratio": _coerce_float(lesion.get("area_ratio")),
                "shape_description": str(lesion.get("shape_description", "")).strip(),
                "morphology_tags": _as_text_list(lesion.get("morphology_tags")),
                "boundary_description": str(lesion.get("boundary_description", "")).strip(),
                "surface_description": str(lesion.get("surface_description", "")).strip(),
                "color_description": str(lesion.get("color_description", "")).strip(),
                "associated_findings": _as_text_list(lesion.get("associated_findings")),
            }
        )

    knowledge_retrieval = raw.get("knowledge_retrieval")
    if not isinstance(knowledge_retrieval, dict):
        raise ValueError("lesion analysis missing knowledge_retrieval object")

    disease_type = str(knowledge_retrieval.get("disease_type", "")).strip()
    if disease_type not in LESION_TYPE_TO_DISEASE_TYPE.values():
        disease_type = LESION_TYPE_TO_DISEASE_TYPE[lesion_type]

    normalized = {
        "lesion_type": lesion_type,
        "confidence_level": confidence_level,
        "lesion_count": _coerce_int(raw.get("lesion_count"), len(normalized_lesions)),
        "lesions": normalized_lesions,
        "knowledge_retrieval": {
            "primary_query": str(knowledge_retrieval.get("primary_query", "")).strip(),
            "disease_type": disease_type,
            "required_knowledge": _as_text_list(knowledge_retrieval.get("required_knowledge")),
            "search_tags": _as_text_list(knowledge_retrieval.get("search_tags")),
            "safety_constraints": _as_text_list(knowledge_retrieval.get("safety_constraints")),
        },
    }
    log_json("结构化病灶分析规范化结果", normalized)
    return normalized


def generate_lesion_analysis(image_path: Path, vision_result: dict) -> dict:
    log_task("AI 结构化识别病灶并生成知识库检索线索")
    image_data_url = encode_image_data_url(image_path)
    local_context = {
        "selected_lesion_type": vision_result.get("lesion_type"),
        "selected_disease_type": vision_result.get("disease_type"),
        "model_name": vision_result.get("model_name") or (vision_result.get("lesion_result") or {}).get("model_name"),
        "polyp_detected": vision_result.get("polyp_detected"),
        "bbox": vision_result.get("bbox"),
        "annotation_bbox": vision_result.get("annotation_bbox"),
        "message": vision_result.get("message", ""),
    }

    content = dashscope_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是内镜图像结构化观察助手。你只负责根据输入图像和本地视觉模型结果提取结构化影像描述，"
                    "不得确诊，不得判断良恶性，不得替代医生诊断。必须只返回 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请基于这张经过 IRetinex 增强且已叠加 YOLO/MedSAM 病灶标注的内镜图像，"
                            "输出结构化 JSON。lesion_type 只能在 polyp 和 barrett 中二选一。"
                            "lesion_type 必须与本地模型结果中的 selected_lesion_type 保持一致。"
                            "请优先参考图中标注区域和本地模型结果，但不要把标注当作确诊依据。"
                            f"本地模型结果：{json.dumps(local_context, ensure_ascii=False)}。"
                            "必须返回如下字段：lesion_type, confidence_level, lesion_count, lesions, knowledge_retrieval。"
                            "lesions 中每个元素必须包含 lesion_id, location_description, anatomical_location, bbox, "
                            "area_ratio, shape_description, morphology_tags, boundary_description, surface_description, "
                            "color_description, associated_findings。"
                            "knowledge_retrieval 必须包含 primary_query, disease_type, required_knowledge, search_tags, "
                            "safety_constraints；其中 disease_type 应使用 colorectal_polyp 或 barrett_esophagus。"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        max_tokens=1000,
        json_mode=True,
        task_name="结构化病灶分析",
    )
    return normalize_lesion_analysis(parse_json_object(content), vision_result)


def _dedupe_knowledge_rules(rules: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        key = (
            str(rule.get("disease_type", "")),
            str(rule.get("rule_id", "")),
            str(rule.get("source_file", "")),
            str(rule.get("source_section_or_page", "")),
            str(rule.get("evidence_text_summary", ""))[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rule)
    return deduped


def load_knowledge_base() -> list[dict]:
    global KNOWLEDGE_BASE_CACHE
    if KNOWLEDGE_BASE_CACHE is not None:
        return KNOWLEDGE_BASE_CACHE

    if not KNOWLEDGE_BASE_PATH.exists():
        log_debug(f"本地知识库不存在：{KNOWLEDGE_BASE_PATH}")
        KNOWLEDGE_BASE_CACHE = []
        return KNOWLEDGE_BASE_CACHE

    try:
        data = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        log_debug(f"本地知识库读取或解析失败：{exc}")
        KNOWLEDGE_BASE_CACHE = []
        return KNOWLEDGE_BASE_CACHE

    if not isinstance(data, list):
        log_debug("本地知识库格式不是 JSON 数组，已忽略。")
        KNOWLEDGE_BASE_CACHE = []
        return KNOWLEDGE_BASE_CACHE

    KNOWLEDGE_BASE_CACHE = _dedupe_knowledge_rules(data)
    log_debug(f"本地知识库加载完成：{len(KNOWLEDGE_BASE_CACHE)} 条去重规则。")
    return KNOWLEDGE_BASE_CACHE


def _collect_query_terms(lesion_analysis: dict) -> list[str]:
    terms: list[str] = []
    retrieval = lesion_analysis.get("knowledge_retrieval") or {}
    for key in ("primary_query", "disease_type"):
        if retrieval.get(key):
            terms.append(str(retrieval[key]))
    for key in ("required_knowledge", "search_tags", "safety_constraints"):
        terms.extend(_as_text_list(retrieval.get(key)))

    for lesion in lesion_analysis.get("lesions") or []:
        if not isinstance(lesion, dict):
            continue
        for key in (
            "shape_description",
            "surface_description",
            "boundary_description",
            "color_description",
            "anatomical_location",
        ):
            if lesion.get(key):
                terms.append(str(lesion[key]))
        terms.extend(_as_text_list(lesion.get("morphology_tags")))
        terms.extend(_as_text_list(lesion.get("associated_findings")))

    normalized_terms = []
    seen = set()
    for raw in terms:
        text = str(raw).strip().lower()
        if not text:
            continue
        candidates = [text]
        if any(char.isascii() and char.isalnum() for char in text):
            candidates.extend(part for part in text.replace("/", " ").replace(",", " ").split() if len(part) >= 3)
        for candidate in candidates:
            if len(candidate) < 2 or candidate in seen:
                continue
            seen.add(candidate)
            normalized_terms.append(candidate)
    return normalized_terms


def _rule_search_text(rule: dict) -> str:
    fields = {
        key: rule.get(key)
        for key in (
            "rule_id",
            "disease_type",
            "cn_name",
            "applicable_scene",
            "endoscopic_features",
            "required_fields",
            "possible_meaning_safe",
            "recommendation_safe",
            "forbidden_outputs",
            "safety_note",
            "evidence_text_summary",
        )
    }
    return json.dumps(fields, ensure_ascii=False).lower()


def select_knowledge_for_report(lesion_analysis: dict) -> list[dict]:
    retrieval = lesion_analysis.get("knowledge_retrieval") or {}
    disease_type = str(retrieval.get("disease_type") or "").strip()
    lesion_type = str(lesion_analysis.get("lesion_type") or "").strip().lower()
    if disease_type not in LESION_TYPE_TO_DISEASE_TYPE.values():
        disease_type = LESION_TYPE_TO_DISEASE_TYPE.get(lesion_type, "")
    if not disease_type:
        return []

    query_terms = _collect_query_terms(lesion_analysis)
    scored: list[tuple[int, dict]] = []
    for rule in load_knowledge_base():
        if rule.get("disease_type") != disease_type:
            continue
        haystack = _rule_search_text(rule)
        score = 0
        for term in query_terms:
            if term in haystack:
                score += max(1, min(4, len(term) // 4))
        if score > 0:
            scored.append((score, rule))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [rule for _, rule in scored[:5]]
    log_debug(f"知识库检索完成：disease_type={disease_type}, 命中={len(selected)}")
    return selected


def format_knowledge_context(rules: list[dict]) -> str:
    if not rules:
        return ""

    formatted_rules = []
    for index, rule in enumerate(rules, start=1):
        formatted_rules.append(
            "\n".join(
                [
                    f"规则 {index}: {rule.get('cn_name') or rule.get('rule_id') or '未命名规则'}",
                    f"- disease_type: {rule.get('disease_type', '')}",
                    f"- applicable_scene: {rule.get('applicable_scene', '')}",
                    f"- endoscopic_features: {json.dumps(rule.get('endoscopic_features', []), ensure_ascii=False)}",
                    f"- possible_meaning_safe: {rule.get('possible_meaning_safe', '')}",
                    f"- recommendation_safe: {rule.get('recommendation_safe', '')}",
                    f"- forbidden_outputs: {json.dumps(rule.get('forbidden_outputs', []), ensure_ascii=False)}",
                    f"- safety_note: {rule.get('safety_note', '')}",
                    f"- evidence_text_summary: {rule.get('evidence_text_summary', '')}",
                    f"- source: {rule.get('source_file', '')} / {rule.get('source_section_or_page', '')}",
                ]
            )
        )
    return "\n\n".join(formatted_rules)


def classify_endoscopy(image_path: Path) -> dict:
    log_task("AI 判断图片是否为内镜图像")
    image_data_url = encode_image_data_url(image_path)
    content = dashscope_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是医学影像分流助手。只判断用户上传图片是否为胃肠镜、消化道内镜等内镜医学图像，"
                    "不要做诊断。必须只返回 JSON。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            '请判断这张图片是否为内镜医学影像。只返回形如 '
                            '{"is_endoscopy": true, "reason": "简短原因"} 的 JSON。'
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        max_tokens=180,
        json_mode=True,
        task_name="内镜图片预检",
    )
    result = parse_json_object(content)
    if not isinstance(result.get("is_endoscopy"), bool):
        raise ValueError("DashScope classification JSON missing is_endoscopy")
    classification = {
        "is_endoscopy": result["is_endoscopy"],
        "reason": str(result.get("reason", "")).strip(),
    }
    log_json("内镜图片预检结构化结果", classification)
    return classification


def generate_diagnostic_report(
    image_path: Path,
    vision_result: dict,
    lesion_analysis: dict | None = None,
    knowledge_context: list[dict] | None = None,
    revision_context: dict | None = None,
) -> dict:
    log_task("AI 生成辅助诊断报告")
    image_data_url = encode_image_data_url(image_path)
    polyp_detected = vision_result.get("polyp_detected")
    bbox = vision_result.get("bbox")
    local_message = vision_result.get("message", "")
    lesion_analysis_text = (
        json.dumps(lesion_analysis, ensure_ascii=False, indent=2)
        if lesion_analysis
        else "本轮结构化病灶分析不可用。"
    )
    knowledge_text = format_knowledge_context(knowledge_context or [])
    knowledge_prompt = (
        f"\n\n本地知识库参考：\n{knowledge_text}\n\n"
        "请严格遵守本地知识库中的 forbidden_outputs 和 safety_note；"
        "知识库只作为辅助参考，不能作为确诊依据。"
        if knowledge_text
        else ""
    )
    revision_prompt = (
        "\n\n反思修订要求：\n"
        f"{json.dumps(revision_context, ensure_ascii=False, indent=2)}\n"
        "请只根据上述反思中指出的事实冲突、遗漏、不确定性或决策影响调整辅助报告；"
        "不要重新解释视觉算法流程，不要引入未由图像、本地模型、结构化分析或知识库支持的新结论。"
        if revision_context
        else ""
    )

    content = dashscope_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是内镜影像辅助诊断报告助手。请用中文输出简洁、谨慎的辅助分析。"
                    "你需要综合图像、本地 YOLO/MedSAM 结果、结构化病灶 JSON 和可用的本地知识库参考。"
                    "不能声称确诊，不能判断良恶性，不能替代医生诊断。"
                    "最终报告必须使用结构化 JSON 输出，字段为 endoscopic_findings、endoscopic_diagnosis、followup_advice。"
                    "如果提供了本地知识库，必须遵守其中的 forbidden_outputs 和安全提示。"
                    "不要生成任何未由后端显式提供的审计类元信息，包括分析完成时间、报告生成时间、系统版本、"
                    "处理设备、模型来源、数据来源、安全等级或“严格遵循某指南/符合医疗AI安全约束”等背书式结论。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请结合这张经过 IRetinex 增强和病灶定位后的内镜结果图，输出一段辅助诊断报告和建议。"
                            f"本地模型结果：polyp_detected={polyp_detected}, bbox={bbox}, message={local_message}。"
                            f"\n\n结构化病灶分析 JSON：\n{lesion_analysis_text}"
                            f"{knowledge_prompt}"
                            f"{revision_prompt}"
                            "\n\n请只返回 JSON 对象，不要使用 Markdown，不要添加 JSON 以外的任何文字。JSON 格式必须为："
                            '{"endoscopic_findings": "图像详细描述", '
                            '"endoscopic_diagnosis": ["1-3条简短疑问式诊断结论，每条必须以？结尾"], '
                            '"followup_advice": "医生复检建议"}。'
                            "endoscopic_findings 需详细描述图像中可见的黏膜形态、颜色、隆起/凹陷、边界、表面结构、出血/糜烂/溃疡等可观察表现，并结合本地定位结果描述疑似区域；若证据不足需说明局限。"
                            "endoscopic_diagnosis 只给出简短疑问式结论，例如“Barrett食管？”“食管下段黏膜异常改变？”，不要解释依据，不要写长句。"
                            "followup_advice 需给出医生复检、随访或进一步检查建议，必须包含“仅供辅助参考，不能替代医生诊断”。"
                            "不要在报告结尾追加时间、版本、设备、模型文件名、数据来源、安全等级或指南背书。"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        max_tokens=900,
        json_mode=True,
        task_name="辅助诊断报告生成",
    )
    return normalize_structured_report(content)


def _build_multi_image_summary(processed_images: list[dict]) -> str:
    lines = []
    for item in processed_images:
        state = item.get("state") or {}
        vision_result = state.get("vision_result") or {}
        lesion_analysis = state.get("lesion_analysis")
        knowledge_context = state.get("knowledge_context") or []
        reflection = state.get("reflection") or {}
        lines.append(
            "\n".join(
                [
                    f"图片 {item.get('index')}: {item.get('filename')}",
                    f"- 预检: {json.dumps(state.get('classification') or {}, ensure_ascii=False)}",
                    f"- 本地视觉结果: {json.dumps(vision_result, ensure_ascii=False)}",
                    f"- 结构化病灶分析: {json.dumps(lesion_analysis, ensure_ascii=False)}",
                    f"- 知识库参考: {json.dumps(knowledge_context[:3], ensure_ascii=False)}",
                    f"- 单图报告摘要: {state.get('report') or '未生成'}",
                    f"- 自检: {json.dumps(reflection, ensure_ascii=False)}",
                ]
            )
        )
    return "\n\n".join(lines)


def generate_multi_image_diagnostic_report(
    processed_images: list[dict],
    lesion_type: str,
    revision_context: dict | None = None,
    source_type: str = "image",
) -> dict:
    log_task("AI 生成多图综合辅助诊断报告")
    summary_text = _build_multi_image_summary(processed_images)
    source_prompt = (
        "注意：用户本次上传的是视频。下面这些图片不是用户直接上传的多张静态图片，而是后端从该视频中抽取、再经大模型逐帧初筛为可能存在病灶的关键帧。报告必须明确体现“用户上传的是视频”，并说明分析基于视频关键帧，不能把关键帧描述成完整视频全过程。"
        if source_type == "video"
        else "下面是同一次上传中的多张内镜图片。"
    )
    content_parts = [
        {
            "type": "text",
            "text": (
                f"{source_prompt}\n"
                "这些内镜图像已经分别完成 YOLO/MedSAM 病灶定位和 IRetinex 清晰化/增强标注。"
                "请把它们作为同一个上下文进行综合分析，不要把任意一张图当成全部证据。"
                f"\n\n本轮图片类型: {lesion_type}\n\n本地处理结果汇总:\n{summary_text}\n\n"
                + (
                    "反思修订要求：\n"
                    f"{json.dumps(revision_context, ensure_ascii=False, indent=2)}\n\n"
                    "请只根据上述反思中指出的事实冲突、遗漏、不确定性或决策影响调整综合报告；"
                    "不要重新运行或假设新的视觉结果。\n\n"
                    if revision_context
                    else ""
                )
                +
                "请只返回 JSON 对象，不要使用 Markdown，不要添加 JSON 以外的任何文字。JSON 格式必须为："
                '{"endoscopic_findings": "图像详细描述", '
                '"endoscopic_diagnosis": ["1-3条简短疑问式诊断结论，每条必须以？结尾"], '
                '"followup_advice": "医生复检建议"}。'
                "endoscopic_findings 需综合描述图像/关键帧中可见的黏膜形态、颜色、隆起/凹陷、边界、表面结构、出血/糜烂/溃疡等表现；多图时概括逐图要点和一致性/差异；视频来源时说明分析基于关键帧，不能代表完整视频全过程。"
                "endoscopic_diagnosis 只给出简短疑问式结论，例如“Barrett食管？”“食管下段黏膜异常改变？”，不要解释依据，不要写长句。"
                "followup_advice 需给出医生复检、随访或进一步检查建议，必须包含“仅供辅助参考，不能替代医生诊断、病理检查或临床决策”。"
            ),
        }
    ]
    for item in processed_images:
        result_image_path = item.get("result_image_path")
        if result_image_path and Path(result_image_path).is_file():
            content_parts.append({"type": "image_url", "image_url": {"url": encode_image_data_url(Path(result_image_path))}})

    content = dashscope_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是谨慎的内镜影像辅助分析助手。你会综合多张已经本地增强、检测和分割后的结果图，"
                    "结合本地 YOLO/MedSAM/IRetinex 输出，给出同一病例上下文下的辅助分析。"
                    "不能声称确诊，不能判断良恶性，不能替代医生。"
                    "最终报告必须使用结构化 JSON 输出，字段为 endoscopic_findings、endoscopic_diagnosis、followup_advice。"
                    "不要生成任何未由后端显式提供的审计类元信息，包括分析完成时间、报告生成时间、系统版本、"
                    "处理设备、模型来源、数据来源、安全等级或“严格遵循某指南/符合医疗AI安全约束”等背书式结论。"
                ),
            },
            {"role": "user", "content": content_parts},
        ],
        max_tokens=1200,
        json_mode=True,
        task_name="多图综合辅助诊断报告生成",
    )
    return normalize_structured_report(content)


def build_local_multi_image_report(
    processed_images: list[dict],
    reason: str | None = None,
    source_type: str = "image",
    lesion_type: str | None = None,
) -> dict:
    video_source_note = source_type == "video"
    finding_lines = []
    if video_source_note:
        finding_lines.append("用户本次上传的是视频；以下所见基于从视频中抽取并经 AI 初筛后的疑似病灶关键帧，不能代表完整视频全过程。")
    finding_lines.append("本轮多张图片已分别完成本地 IRetinex 增强和 YOLO/MedSAM 病灶定位，外部 AI 综合报告暂时不可用。")
    detected_count = 0
    usable_count = 0
    for item in processed_images:
        state = item.get("state") or {}
        vision_result = state.get("vision_result") or {}
        detected = vision_result.get("polyp_detected")
        bbox = vision_result.get("bbox")
        if vision_result:
            usable_count += 1
        if detected:
            detected_count += 1
            finding = f"提示存在疑似病灶区域，检测框坐标为 {bbox}。"
        elif detected is False:
            finding = "未检测到明显疑似病灶区域。"
        else:
            finding = "本地视觉结果暂不可用。"
        finding_lines.append(f"图片 {item.get('index')}（{item.get('filename')}）：{finding}")

    representative_vision_result = {}
    for item in processed_images:
        state = item.get("state") or {}
        vision_result = state.get("vision_result") or {}
        if vision_result:
            representative_vision_result = vision_result
            if vision_result.get("polyp_detected"):
                break
    diagnosis = build_brief_local_diagnosis(representative_vision_result, lesion_type)

    advice = "建议由内镜医生复核原始图像/完整视频及标注区域；如症状持续、图像质量欠佳、医生认为有必要或存在高危因素，应安排复查内镜，并按医生意见完善活检病理等检查。仅供辅助参考，不能替代医生诊断、病理检查或临床决策。"
    if reason:
        advice += f" 外部 AI 综合报告暂时不可用，降级原因：{reason}"
    return normalize_structured_report(
        {
            "endoscopic_findings": " ".join(finding_lines),
            "endoscopic_diagnosis": diagnosis.splitlines(),
            "followup_advice": advice,
        }
    )


def build_local_multi_reflection(processed_images: list[dict], reason: str | None = None) -> dict:
    detected_count = 0
    usable_count = 0
    for item in processed_images:
        state = item.get("state") or {}
        vision_result = state.get("vision_result") or {}
        if vision_result:
            usable_count += 1
        if vision_result.get("polyp_detected"):
            detected_count += 1

    confidence = 62 if detected_count else 55
    uncertainties = [
        "多张静态截图不能替代完整内镜检查过程",
        "不同图片可能来自不同视角或不同清晰度，跨图综合判断仍需人工复核",
        "置信度仅基于本地处理结果和综合报告的一致性估计",
    ]
    if reason:
        uncertainties.insert(0, f"自检降级原因：{reason}")

    return {
        "confidence": confidence,
        "confidence_label": "中",
        "reflection": f"本轮共有 {usable_count} 张图片进入综合分析，其中 {detected_count} 张提示疑似病灶区域。综合报告可作为辅助线索，但仍需结合完整检查过程理解。",
        "uncertainties": uncertainties,
        "suggestion": "请由专业医生结合原始内镜视频、各帧位置关系、病史和必要病理检查进行最终判断。",
        "report_revision_score": 3 if reason else 0,
        "revise_report": False,
        "revision_focus": [],
    }


def generate_multi_result_reflection(processed_images: list[dict], report: str) -> dict:
    log_task("AI 多图结果自检与置信度评估")
    summary_text = _build_multi_image_summary(processed_images)
    content_parts = [
        {
            "type": "text",
            "text": (
                "请对下面这次多图综合辅助分析进行质量自检和置信度评估。"
                "这些图片已经分别经过 YOLO/MedSAM 病灶定位和 IRetinex 增强标注，随后生成了一份综合辅助报告。"
                f"\n\n逐图本地处理摘要:\n{summary_text}\n\n综合辅助报告:\n{report}\n\n"
                "请只返回 JSON，格式为："
                '{"confidence": 0-100, "confidence_label": "低/中/高", '
                '"reflection": "一段简短反思", "uncertainties": ["不确定因素"], '
                '"suggestion": "建议如何理解该结果", '
                '"report_revision_score": 0-10, "revise_report": true/false, '
                '"revision_focus": ["如需重写报告，应调整的重点"]}。'
                "置信度应评估这份多图综合辅助报告与图像证据、本地模型结果之间的一致性，不代表确诊概率。"
            ),
        }
    ]
    for item in processed_images:
        result_image_path = item.get("result_image_path")
        if result_image_path and Path(result_image_path).is_file():
            content_parts.append({"type": "image_url", "image_url": {"url": encode_image_data_url(Path(result_image_path))}})

    content = dashscope_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是医学影像辅助系统的质量自检助手。你要审查多图综合报告是否与所有输入图像和本地视觉模型结果一致，"
                    "给出校准后的置信度。不要给出确诊，不要替代医生。必须只返回 JSON。"
                ),
            },
            {"role": "user", "content": content_parts},
        ],
        max_tokens=700,
        json_mode=True,
        task_name="多图结果自检与置信度评估",
        model=CONFIDENCE_CHECK_MODEL,
    )
    result = parse_json_object(content)
    confidence = result.get("confidence")
    try:
        confidence = max(0, min(100, round(float(confidence))))
    except (TypeError, ValueError):
        confidence = None
    uncertainties = result.get("uncertainties", [])
    if isinstance(uncertainties, str):
        uncertainties = [uncertainties]
    elif not isinstance(uncertainties, list):
        uncertainties = []
    return {
        "confidence": confidence,
        "confidence_label": str(result.get("confidence_label", "")).strip(),
        "reflection": str(result.get("reflection", "")).strip(),
        "uncertainties": [str(item).strip() for item in uncertainties if str(item).strip()],
        "suggestion": str(result.get("suggestion", "")).strip(),
        "report_revision_score": _coerce_revision_score_value(result.get("report_revision_score")),
        "revise_report": _coerce_bool(result.get("revise_report")),
        "revision_focus": _as_text_list(result.get("revision_focus")),
    }


def build_local_classification_fallback(reason: str) -> dict:
    return {
        "is_endoscopy": True,
        "reason": f"AI 预检暂时不可用，已跳过预检并继续执行本地 IRetinex 分析。原因：{reason}",
        "fallback": True,
    }


def build_local_diagnostic_report(vision_result: dict, reason: str | None = None) -> dict:
    polyp_detected = bool(vision_result.get("polyp_detected"))
    bbox = vision_result.get("bbox")

    if polyp_detected:
        finding = (
            f"本地 IRetinex 增强与 YOLO/MedSAM 定位流程提示图像中存在疑似病灶区域，"
            f"检测框坐标为 {bbox}。该区域建议作为后续人工复核的重点观察位置；"
            "需结合原始内镜图像进一步观察黏膜颜色、形态、边界、表面结构及是否伴出血、糜烂或溃疡。"
        )
        diagnosis = build_brief_local_diagnosis(vision_result)
    else:
        finding = "本地 IRetinex 增强后，YOLO/MedSAM 流程未检测到明显疑似病灶区域；仍需结合图像清晰度、曝光、观察范围和完整内镜过程判断。"
        diagnosis = build_brief_local_diagnosis(vision_result)

    advice = (
        "建议由内镜医生复核原始图像及标注区域；如图像质量不佳、症状持续、存在高危因素，"
        "或医生认为有必要，应进一步复查内镜，并按医生意见完善活检病理等检查。"
        "仅供辅助参考，不能替代医生诊断。"
    )
    if reason:
        advice += f" 外部 AI 报告生成暂时不可用，降级原因：{reason}"
    return normalize_structured_report(
        {
            "endoscopic_findings": finding,
            "endoscopic_diagnosis": diagnosis.splitlines(),
            "followup_advice": advice,
        }
    )


def build_local_reflection(vision_result: dict, reason: str | None = None) -> dict:
    polyp_detected = bool(vision_result.get("polyp_detected"))
    confidence = 60 if polyp_detected else 50
    label = "中" if polyp_detected else "低"
    uncertainties = [
        "外部 AI 自检不可用，当前置信度仅基于本地模型输出估计",
        "单张图片无法替代完整内镜检查过程",
        "图像清晰度、曝光和病灶遮挡可能影响算法结果",
    ]
    if reason:
        uncertainties.insert(0, f"自检降级原因：{reason}")

    return {
        "confidence": confidence,
        "confidence_label": label,
        "reflection": "本轮结果已完成本地增强与定位，但缺少外部多模态模型复核，应按辅助线索理解。",
        "uncertainties": uncertainties,
        "suggestion": "请结合原始图像、本地定位结果和医生意见谨慎解读。",
        "report_revision_score": 3 if reason else 0,
        "revise_report": False,
        "revision_focus": [],
    }


def generate_result_reflection(image_path: Path, vision_result: dict, report: str) -> dict:
    log_task("AI 反思生成结果并评估置信度")
    image_data_url = encode_image_data_url(image_path)
    polyp_detected = vision_result.get("polyp_detected")
    bbox = vision_result.get("bbox")
    local_message = vision_result.get("message", "")

    content = dashscope_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是医学影像辅助系统的质量自检助手。请基于输入图像、本地视觉模型结果和已经生成的辅助报告，"
                    "反思该结果是否充分、是否存在不确定性，并给出大致置信度。"
                    "请尽量提高置信度评估的准确性与校准性，不要机械输出固定分数，也不要为了显得谨慎而一律压低。"
                    "若存在遮挡、模糊、边界不清或结论冲突，应主动下调。"
                    "不要给出确诊结论，不要替代医生判断。必须只返回 JSON。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请审阅下面的辅助诊断结果，并只返回 JSON，格式为："
                            '{"confidence": 0-100, "confidence_label": "低/中/高", '
                            '"reflection": "一段简短反思", "uncertainties": ["不确定因素"], '
                            '"suggestion": "建议如何理解该结果", '
                            '"report_revision_score": 0-10, "revise_report": true/false, '
                            '"revision_focus": ["如需重写报告，应调整的重点"]}。'
                            "请输出最符合证据强弱的具体置信度数字，优先追求准确校准。"
                            f"本地模型结果：polyp_detected={polyp_detected}, bbox={bbox}, message={local_message}。"
                            f"已生成报告：{report}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        max_tokens=650,
        json_mode=True,
        task_name="结果反思与置信度评估",
        model=CONFIDENCE_CHECK_MODEL,
    )

    result = parse_json_object(content)
    confidence = result.get("confidence")
    try:
        confidence = max(0, min(100, round(float(confidence))))
    except (TypeError, ValueError):
        confidence = None

    uncertainties = result.get("uncertainties", [])
    if isinstance(uncertainties, str):
        uncertainties = [uncertainties]
    elif not isinstance(uncertainties, list):
        uncertainties = []

    reflection = {
        "confidence": confidence,
        "confidence_label": str(result.get("confidence_label", "")).strip(),
        "reflection": str(result.get("reflection", "")).strip(),
        "uncertainties": [str(item).strip() for item in uncertainties if str(item).strip()],
        "suggestion": str(result.get("suggestion", "")).strip(),
        "report_revision_score": _coerce_revision_score_value(result.get("report_revision_score")),
        "revise_report": _coerce_bool(result.get("revise_report")),
        "revision_focus": _as_text_list(result.get("revision_focus")),
    }
    log_json("AI 结果反思结构化结果", reflection)
    return reflection


def save_upload(uploaded_file) -> tuple[str, Path]:
    log_task("保存上传图片")
    run_id = uuid4().hex
    upload_dir = UPLOAD_ROOT / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(uploaded_file.filename or "")
    if not safe_name:
        safe_name = "uploaded_image.jpg"
    upload_path = upload_dir / safe_name
    uploaded_file.save(upload_path)
    log_debug(f"上传图片已保存：run_id={run_id}, filename={safe_name}, path={upload_path}")
    return run_id, upload_path


def get_uploaded_images() -> list:
    files = request.files.getlist("image")
    if not files:
        single_file = request.files.get("image")
        files = [single_file] if single_file is not None else []
    return [file for file in files if file is not None and file.filename]


def save_uploads(uploaded_files: list) -> tuple[str, list[dict]]:
    log_task("保存多张上传图片")
    run_id = uuid4().hex
    upload_dir = UPLOAD_ROOT / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_items = []
    used_names: set[str] = set()
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        safe_name = secure_filename(uploaded_file.filename or "") or f"uploaded_image_{index}.jpg"
        stem = Path(safe_name).stem or f"uploaded_image_{index}"
        suffix = Path(safe_name).suffix or ".jpg"
        candidate = safe_name
        counter = 2
        while candidate.lower() in used_names or (upload_dir / candidate).exists():
            candidate = f"{stem}_{counter}{suffix}"
            counter += 1
        used_names.add(candidate.lower())

        upload_path = upload_dir / candidate
        uploaded_file.save(upload_path)
        saved_items.append(
            {
                "index": index,
                "filename": candidate,
                "upload_path": upload_path,
                "upload_url": url_for("upload_file", run_id=run_id, filename=candidate),
            }
        )

    log_debug(f"多图上传已保存：run_id={run_id}, count={len(saved_items)}")
    return run_id, saved_items


def save_video_upload(uploaded_file) -> tuple[str, Path, str]:
    log_task("保存上传视频")
    run_id = uuid4().hex
    upload_dir = UPLOAD_ROOT / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(uploaded_file.filename or "") or "uploaded_video.mp4"
    upload_path = upload_dir / safe_name
    uploaded_file.save(upload_path)
    upload_url = url_for("upload_file", run_id=run_id, filename=safe_name)
    log_debug(f"上传视频已保存：run_id={run_id}, filename={safe_name}, path={upload_path}")
    return run_id, upload_path, upload_url


def _find_endoscopy_crop_box(image_bgr) -> tuple[int, int, int, int] | None:
    import cv2
    import numpy as np

    height, width = image_bgr.shape[:2]
    if height < 20 or width < 20:
        return None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    threshold = int(os.getenv("MEDICALAI_BLACK_CROP_THRESHOLD", "25"))

    # Prefer the saturated endoscopic tissue area so white overlay text on black bars
    # does not keep the black margin inside the crop.
    tissue_mask = ((hsv[:, :, 2] > threshold) & (hsv[:, :, 1] > 18)).astype(np.uint8) * 255
    kernel_size = max(5, min(width, height) // 120)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        fallback_mask = (gray > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(fallback_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < width * height * 0.08:
        return None

    x, y, crop_w, crop_h = cv2.boundingRect(contour)
    if crop_w < width * 0.25 or crop_h < height * 0.25:
        return None

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + crop_w)
    y2 = min(height, y + crop_h)
    if (x2 - x1) >= width * 0.98 and (y2 - y1) >= height * 0.98:
        return None
    return x1, y1, x2, y2


def crop_black_video_frame_file(image_path: Path) -> dict:
    try:
        import cv2
        import numpy as np

        path = Path(image_path)
        image_bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return {"cropped": False, "reason": "unreadable"}

        crop_box = _find_endoscopy_crop_box(image_bgr)
        if not crop_box:
            return {"cropped": False, "reason": "no_crop"}

        x1, y1, x2, y2 = crop_box
        cropped = image_bgr[y1:y2, x1:x2]
        ext = path.suffix.lower() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else ".jpg"
        ok, encoded = cv2.imencode(ext, cropped)
        if not ok:
            return {"cropped": False, "reason": "encode_failed"}
        encoded.tofile(str(path))
        return {
            "cropped": True,
            "crop_box": [x1, y1, x2, y2],
            "original_size": [int(image_bgr.shape[1]), int(image_bgr.shape[0])],
            "cropped_size": [int(cropped.shape[1]), int(cropped.shape[0])],
        }
    except Exception as exc:
        return {"cropped": False, "reason": str(exc)}


def save_cropped_video_frame_bytes(image_bytes: bytes, save_path: Path) -> dict:
    save_path.write_bytes(image_bytes)
    return crop_black_video_frame_file(save_path)


def _load_keyframe_yolo_model(lesion_type: str):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))
    try:
        from ultralytics import YOLO
    except Exception as exc:
        log_debug(f"关键帧抽取 YOLO 导入失败，将跳过 YOLO 辅助：{exc}")
        return None

    model_path = BASE_DIR / "weights" / ("YOLO_upper_gi.pt" if lesion_type == "barrett" else "YOLO_colon.pt")
    if not model_path.exists():
        log_debug(f"关键帧抽取 YOLO 权重不存在，将跳过 YOLO 辅助：{model_path}")
        return None

    try:
        return YOLO(str(model_path))
    except Exception as exc:
        log_debug(f"关键帧抽取 YOLO 加载失败，将跳过 YOLO 辅助：{exc}")
        return None


def extract_video_keyframes(video_path: Path, lesion_type: str, output_root: Path) -> list[Path]:
    log_task("视频关键帧抽取")
    output_root.mkdir(parents=True, exist_ok=True)
    model = _load_keyframe_yolo_model(lesion_type)
    if lesion_type == "barrett":
        from algorithm.BE_keyframe_extraction import process_video
    else:
        from algorithm.colon_keyframe_extraction import process_video

    keyframe_paths = process_video(str(video_path), model=model, output_root=str(output_root))
    normalized_paths = [Path(path).resolve() for path in keyframe_paths if Path(path).is_file()]
    cropped_count = 0
    for frame_path in normalized_paths:
        crop_result = crop_black_video_frame_file(frame_path)
        if crop_result.get("cropped"):
            cropped_count += 1
            log_debug(f"自动关键帧已裁剪黑边：file={frame_path.name}, crop={crop_result.get('crop_box')}")
    log_debug(f"视频关键帧抽取完成：count={len(normalized_paths)}")
    if normalized_paths:
        log_debug(f"视频关键帧黑边裁剪完成：cropped={cropped_count}/{len(normalized_paths)}")
    return normalized_paths


def screen_keyframe_for_lesion(image_path: Path, lesion_type: str) -> bool:
    disease_name = "Barrett 食管相关病灶" if lesion_type == "barrett" else "结直肠息肉相关病灶"
    image_data_url = encode_image_data_url(image_path)
    content = dashscope_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是内镜视频关键帧初筛助手。判断图像中是否可能存在指定病种相关疑似病灶。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"请判断这张内镜视频关键帧中是否可能存在{disease_name}。"
                            "如果可能存在，输出：是。否则输出：否。然后给出理由。"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        max_tokens=32760,
        json_mode=False,
        task_name="视频关键帧疑似病灶初筛",
    )
    answer = str(content or "").strip()
    log_debug(f"关键帧初筛结果：file={image_path.name}, answer={answer}")
    return answer.startswith("是")


def _extract_keyframe_time_label(image_path: Path) -> str:
    match = re.search(r"_t([0-9]+(?:\.[0-9]+)?)s", image_path.name)
    if match:
        return f"{match.group(1)}s"

    match = re.search(r"_frame(\d+)_", image_path.name)
    if match:
        return f"frame {match.group(1)}"

    return "未知"


def screen_keyframes_for_lesion(keyframe_paths: list[Path], lesion_type: str) -> dict:
    if not keyframe_paths:
        return {
            "selected_paths": [],
            "positive_count": 0,
            "forced_path": None,
            "frame_decisions": [],
            "raw_result": {},
        }

    disease_name = "Barrett 食管相关病灶" if lesion_type == "barrett" else "结直肠息肉相关病灶"
    frame_map = {f"KF{index:03d}": Path(path).resolve() for index, path in enumerate(keyframe_paths, start=1)}
    content_parts = [
        {
            "type": "text",
            "text": (
                f"下面是同一段内镜视频自动抽取出的 {len(frame_map)} 张候选关键帧。任务病种：{disease_name}。\n"
                "请综合比较所有关键帧，判断哪些关键帧可能存在该病种相关疑似病灶。"
                "每张图前都有唯一 frame_id，例如 KF001。你的回答必须只引用这些 frame_id，不能使用“第一张/第二张”或文件名代替。\n"
                "如果存在可疑病灶，请最多选择 2 张最可疑关键帧放入 suspicious_frames，并给出 0-100 的 suspicion_score。"
                "如果你认为所有关键帧都没有明确可疑病灶，suspicious_frames 必须为空，同时必须在 most_suspicious_frame_id 中选择 1 张相对最可疑、最值得后续展示和复核的关键帧。\n"
                "必须只返回 JSON，格式如下："
                '{"has_suspicious_lesion": true, '
                '"suspicious_frames": [{"frame_id": "KF001", "suspicion_score": 90, "reason": "简短原因"}, {"frame_id": "KF002", "suspicion_score": 80, "reason": "简短原因"}], '
                '"most_suspicious_frame_id": "KF001", '
                '"overall_reason": "简短总体理由"}'
            ),
        }
    ]
    for frame_id, image_path in frame_map.items():
        content_parts.append(
            {
                "type": "text",
                "text": f"关键帧 {frame_id}\n文件名：{image_path.name}\n抽帧位置：{_extract_keyframe_time_label(image_path)}",
            }
        )
        content_parts.append({"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}})

    content = dashscope_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是内镜视频关键帧综合初筛助手。你需要一次性比较多张关键帧，"
                    "筛出最多 2 张可能存在指定病种相关疑似病灶的帧；如无明确阳性，也必须选择一张相对最可疑帧。"
                    "只返回 JSON。"
                ),
            },
            {"role": "user", "content": content_parts},
        ],
        max_tokens=1200,
        json_mode=True,
        task_name="视频关键帧多图疑似病灶初筛",
    )
    result = parse_json_object(content)
    valid_frame_ids = set(frame_map)

    suspicious_frames = result.get("suspicious_frames") or []
    if not isinstance(suspicious_frames, list):
        suspicious_frames = []

    selected_items = []
    normalized_decisions = []
    for item in suspicious_frames:
        if not isinstance(item, dict):
            continue
        frame_id = str(item.get("frame_id") or "").strip().upper()
        if frame_id not in valid_frame_ids:
            continue
        score = item.get("suspicion_score")
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = None
        decision = {
            "frame_id": frame_id,
            "filename": frame_map[frame_id].name,
            "suspicion_score": score,
            "reason": str(item.get("reason") or "").strip(),
        }
        selected_items.append(decision)
        normalized_decisions.append(decision)

    selected_items.sort(
        key=lambda item: item["suspicion_score"] if item["suspicion_score"] is not None else -1,
        reverse=True,
    )
    selected_ids = [item["frame_id"] for item in selected_items[:2]]

    has_suspicious_lesion = bool(result.get("has_suspicious_lesion"))
    most_suspicious_id = str(result.get("most_suspicious_frame_id") or "").strip().upper()
    if most_suspicious_id not in valid_frame_ids:
        if selected_ids:
            most_suspicious_id = selected_ids[0]
        else:
            raise ValueError("关键帧多图初筛 JSON 缺少有效 most_suspicious_frame_id")

    forced_path = None
    if not has_suspicious_lesion or not selected_ids:
        selected_ids = [most_suspicious_id]
        forced_path = frame_map[most_suspicious_id]

    selected_ids = list(dict.fromkeys(selected_ids))[:2]
    selected_paths = [frame_map[frame_id] for frame_id in selected_ids]
    log_json(
        "关键帧多图初筛结构化结果",
        {
            "has_suspicious_lesion": has_suspicious_lesion,
            "selected_ids": selected_ids,
            "most_suspicious_frame_id": most_suspicious_id,
            "positive_count": 0 if forced_path else len(selected_paths),
            "overall_reason": result.get("overall_reason"),
        },
    )
    return {
        "selected_paths": selected_paths,
        "positive_count": 0 if forced_path else len(selected_paths),
        "forced_path": forced_path,
        "frame_decisions": normalized_decisions,
        "raw_result": result,
    }


def select_required_keyframe_for_video(keyframe_paths: list[Path], lesion_type: str) -> Path | None:
    if not keyframe_paths:
        return None

    disease_name = "Barrett 食管相关病灶" if lesion_type == "barrett" else "结直肠息肉相关病灶"
    max_candidates = max(1, int(os.getenv("MEDICALAI_REQUIRED_KEYFRAME_CANDIDATES", "8")))
    if len(keyframe_paths) <= max_candidates:
        candidates = keyframe_paths
    else:
        step = (len(keyframe_paths) - 1) / (max_candidates - 1) if max_candidates > 1 else 0
        indexes = sorted({int(round(index * step)) for index in range(max_candidates)})
        candidates = [keyframe_paths[index] for index in indexes]

    content_parts = [
        {
            "type": "text",
            "text": (
                f"下面是同一段内镜视频抽取出的候选关键帧。任务病种：{disease_name}。\n"
                "请必须选择 1 张最适合进入后续展示、增强和 YOLO/MedSAM 识别流程的图片。"
                "不允许回答“没有合适图片”或全部否。"
                "如果没有明确病灶，也要选择相对最清晰、内镜视野最完整、最可能包含可疑黏膜变化的一张。\n"
                "只输出候选编号，例如：3。"
            ),
        }
    ]
    for index, image_path in enumerate(candidates, start=1):
        content_parts.append({"type": "text", "text": f"候选 {index}：{image_path.name}"})
        content_parts.append({"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}})

    content = dashscope_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是内镜视频关键帧强制筛选助手。你必须从候选图像中选出 1 张，"
                    "用于保证报告至少有一张自动关键帧可展示。"
                ),
            },
            {"role": "user", "content": content_parts},
        ],
        max_tokens=256,
        json_mode=False,
        task_name="视频关键帧强制保留",
    )
    answer = str(content or "").strip()
    match = re.search(r"\d+", answer)
    if match:
        selected_index = int(match.group(0))
        if 1 <= selected_index <= len(candidates):
            selected = candidates[selected_index - 1]
            log_debug(f"关键帧强制保留结果：selected={selected.name}, answer={answer}")
            return selected

    fallback = candidates[len(candidates) // 2]
    log_debug(f"关键帧强制保留解析失败，使用中间帧兜底：selected={fallback.name}, answer={answer}")
    return fallback


def build_keyframe_upload_items(run_id: str, upload_dir: Path, keyframe_paths: list[Path], video_path: Path) -> list[dict]:
    saved_items = []
    for index, frame_path in enumerate(keyframe_paths, start=1):
        suffix = frame_path.suffix or ".jpg"
        filename = f"video_keyframe_{index:02d}{suffix}"
        upload_path = upload_dir / filename
        if frame_path.resolve() != upload_path.resolve():
            shutil.copy2(frame_path, upload_path)
        saved_items.append(
            {
                "index": index,
                "filename": filename,
                "upload_path": upload_path,
                "upload_url": url_for("upload_file", run_id=run_id, filename=filename),
                "source_type": "video",
                "original_video_path": str(video_path),
                "source_keyframe_path": str(frame_path),
            }
        )
    return saved_items


def collect_manual_video_frame_items(context: dict, start_index: int = 1) -> list[dict]:
    items = []
    for offset, item in enumerate(context.get("manual_frames") or [], start=start_index):
        normalized = dict(item)
        normalized["index"] = offset
        items.append(normalized)
    return items


def wait_for_manual_video_frames(context: dict) -> None:
    start_time = time.monotonic()
    while not context.get("manual_frames_closed"):
        now = time.monotonic()
        if now - start_time >= MANUAL_FRAME_MAX_WAIT_SECONDS:
            log_debug(
                "手动视频选帧等待达到最大时长，自动继续后续分析",
            )
            break

        time.sleep(0.5)


def is_valid_run_id(run_id: str) -> bool:
    return len(run_id) == 32 and all(char in "0123456789abcdef" for char in run_id)


def make_step(step_type: str, title: str, body: str, *, meta: dict | None = None) -> dict:
    step_meta = dict(meta or {})
    step_model = get_step_model(step_type, title)
    if step_model and "model" not in step_meta:
        step_meta["model"] = step_model
    return {
        "type": step_type,
        "title": title,
        "body": body,
        "meta": step_meta,
    }


def make_stream_event(event_type: str, **payload) -> str:
    payload["event"] = event_type
    return json.dumps(payload, ensure_ascii=False) + "\n"


def summarize_context(context: dict) -> str:
    processed_images = context.get("processed_images") or []
    if processed_images:
        image_summaries = []
        for item in processed_images:
            state = item.get("state") or {}
            image_summaries.append(
                f"- 图片 {item.get('index')}（{item.get('filename')}）："
                f"vision={json.dumps(state.get('vision_result') or {}, ensure_ascii=False)}；"
                f"lesion_analysis={json.dumps(state.get('lesion_analysis'), ensure_ascii=False)}"
            )
        return (
            "本次多图分析上下文：\n"
            f"run_id: {context.get('run_id')}\n"
            f"图片数量: {len(processed_images)}\n"
            f"综合辅助报告: {context.get('report') or '未生成'}\n"
            f"逐图结果:\n" + "\n".join(image_summaries)
        )

    classification = context.get("classification") or {}
    vision_result = context.get("vision_result") or {}
    lesion_analysis = context.get("lesion_analysis") or {}
    knowledge_context = context.get("knowledge_context") or []
    reflection = context.get("reflection") or {}
    steps_text = "\n".join(f"- {step['title']}：{step['body']}" for step in context.get("steps", []))
    return (
        "本次分析上下文：\n"
        f"run_id: {context.get('run_id')}\n"
        f"预检结果: is_endoscopy={classification.get('is_endoscopy')}, reason={classification.get('reason')}\n"
        f"本地模型结果: {json.dumps(vision_result, ensure_ascii=False)}\n"
        f"结构化病灶分析: {json.dumps(lesion_analysis, ensure_ascii=False)}\n"
        f"知识库命中数量: {len(knowledge_context)}\n"
        f"辅助报告: {context.get('report') or '未生成'}\n"
        f"自检结果: {json.dumps(reflection, ensure_ascii=False)}\n"
        f"分析过程:\n{steps_text}"
    )


def build_followup_messages(context: dict, user_text: str) -> list[dict]:
    system_prompt = (
        "你是 IRetinex Agent，一个谨慎的内镜影像辅助分析助手。"
        "回答必须基于用户上传图片、系统分析过程、本地模型输出、结果图和前面对话。"
        "不要夸大能力，不要给出确诊结论；涉及医疗建议时提醒用户由专业医生结合临床判断。"
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for item in context.get("chat_history", [])[-8:]:
        messages.append({"role": item["role"], "content": item["content"]})

    content_parts = [{"type": "text", "text": f"{summarize_context(context)}\n\n用户追问：{user_text}"}]
    upload_path = context.get("upload_path")
    result_image_path = context.get("result_image_path")
    if upload_path and Path(upload_path).is_file():
        content_parts.append({"type": "image_url", "image_url": {"url": encode_image_data_url(Path(upload_path))}})
    if result_image_path and Path(result_image_path).is_file():
        content_parts.append({"type": "image_url", "image_url": {"url": encode_image_data_url(Path(result_image_path))}})
    for item in context.get("processed_images") or []:
        item_upload_path = item.get("upload_path")
        item_result_image_path = item.get("result_image_path")
        if item_upload_path and Path(item_upload_path).is_file():
            content_parts.append({"type": "image_url", "image_url": {"url": encode_image_data_url(Path(item_upload_path))}})
        if item_result_image_path and Path(item_result_image_path).is_file():
            content_parts.append({"type": "image_url", "image_url": {"url": encode_image_data_url(Path(item_result_image_path))}})
    messages.append({"role": "user", "content": content_parts})
    return messages


def build_local_followup_answer(context: dict, user_text: str, reason: str | None = None) -> str:
    processed_images = context.get("processed_images") or []
    if processed_images:
        report = context.get("report") or "本轮尚未生成综合报告。"
        image_lines = []
        for item in processed_images:
            state = item.get("state") or {}
            vision_result = state.get("vision_result") or {}
            detected = vision_result.get("polyp_detected")
            bbox = vision_result.get("bbox")
            if detected:
                summary = f"提示存在疑似病灶区域，检测框坐标为 {bbox}。"
            elif detected is False:
                summary = "未检测到明显疑似病灶区域。"
            else:
                summary = "本地视觉结果暂不可用。"
            image_lines.append(f"图片 {item.get('index')}（{item.get('filename')}）：{summary}")
        reason_text = f"\n\n当前外部 AI 追问服务不可用，已按多图上下文给出保底回答。原因：{reason}" if reason else ""
        return (
            f"你追问的是：{user_text}\n\n"
            f"多图本地处理摘要：\n" + "\n".join(image_lines) + "\n\n"
            f"本轮综合报告：\n{report}\n\n"
            "请将这些结果作为辅助线索，由专业医生结合完整内镜过程、临床资料和必要病理检查综合判断。"
            f"{reason_text}"
        )

    vision_result = context.get("vision_result") or {}
    report = context.get("report") or "本轮尚未生成报告。"
    reflection = context.get("reflection") or {}
    detected = vision_result.get("polyp_detected")
    bbox = vision_result.get("bbox")
    confidence = reflection.get("confidence")
    confidence_text = "未评估" if confidence is None else f"{confidence}%"
    reason_text = f"\n\n当前外部 AI 追问服务不可用，已按本轮分析上下文给出保底回答。原因：{reason}" if reason else ""

    if detected:
        local_summary = f"本地模型提示存在疑似病灶区域，检测框坐标为 {bbox}。"
    elif detected is False:
        local_summary = "本地模型未检测到明显疑似病灶区域。"
    else:
        local_summary = "本地模型结果暂不可用。"

    return (
        f"你追问的是：{user_text}\n\n"
        f"{local_summary}粗略置信度：{confidence_text}。\n\n"
        f"本轮报告摘要：\n{report}\n\n"
        "请把这些内容作为辅助线索，由专业医生结合原始内镜图像、完整检查过程和临床资料判断。"
        f"{reason_text}"
    )


class FlaskAgentRuntime:
    service_unavailable_message = SERVICE_UNAVAILABLE_MESSAGE

    def make_step(self, step_type: str, title: str, body: str, *, meta: dict | None = None) -> dict:
        return make_step(step_type, title, body, meta=meta)

    def validate_image(self, image_path: Path) -> bool:
        return validate_image(image_path)

    def classify_endoscopy(self, image_path: Path) -> dict:
        return classify_endoscopy(image_path)

    def build_local_classification_fallback(self, reason: str) -> dict:
        return build_local_classification_fallback(reason)

    def enhance_input_image(self, input_image_path: str, output_dir: str) -> dict:
        from vision_tool import enhance_input_image

        return enhance_input_image(input_image_path, output_dir=output_dir)

    def diagnose_original_image(self, input_image_path: str, output_dir: str, lesion_type: str | None = None) -> dict:
        from vision_tool import diagnose_original_image

        return diagnose_original_image(input_image_path, output_dir=output_dir, lesion_type=lesion_type)

    def annotate_enhanced_image(self, enhancement_result: dict, lesion_result: dict, output_dir: str) -> dict:
        from vision_tool import annotate_enhanced_image

        return annotate_enhanced_image(enhancement_result, lesion_result, output_dir=output_dir)

    def diagnose_enhanced_image(self, enhancement_result: dict, output_dir: str, lesion_type: str | None = None) -> dict:
        from vision_tool import diagnose_enhanced_image

        return diagnose_enhanced_image(enhancement_result, output_dir=output_dir, lesion_type=lesion_type)

    def process_image(self, input_image_path: str, output_dir: str, lesion_type: str | None = None) -> dict:
        from vision_tool import process_image

        return process_image(input_image_path, output_dir=output_dir, lesion_type=lesion_type)

    def build_result_image_url(self, run_id: str, upload_path: Path, result_image_path: Path) -> str:
        upload_dir_path = upload_path.parent.resolve()
        try:
            result_filename = result_image_path.relative_to(upload_dir_path).as_posix()
            return url_for("upload_file", run_id=run_id, filename=result_filename)
        except ValueError:
            return url_for("output_file", run_id=run_id, filename=result_image_path.name)

    def generate_lesion_analysis(self, image_path: Path, vision_result: dict) -> dict:
        return generate_lesion_analysis(image_path, vision_result)

    def select_report_knowledge(self, lesion_analysis: dict) -> list[dict]:
        return select_knowledge_for_report(lesion_analysis)

    def generate_diagnostic_report(
        self,
        image_path: Path,
        vision_result: dict,
        lesion_analysis: dict | None = None,
        knowledge_context: list[dict] | None = None,
        revision_context: dict | None = None,
    ) -> dict:
        return generate_diagnostic_report(image_path, vision_result, lesion_analysis, knowledge_context, revision_context)

    def build_local_diagnostic_report(self, vision_result: dict, reason: str | None = None) -> dict:
        return build_local_diagnostic_report(vision_result, reason)

    def format_structured_report(self, report_json: dict) -> str:
        return format_structured_report(report_json)

    def generate_result_reflection(self, image_path: Path, vision_result: dict, report: str) -> dict:
        return generate_result_reflection(image_path, vision_result, report)

    def build_local_reflection(self, vision_result: dict, reason: str | None = None) -> dict:
        return build_local_reflection(vision_result, reason)

    def generate_multi_image_diagnostic_report(
        self,
        processed_images: list[dict],
        lesion_type: str,
        revision_context: dict | None = None,
    ) -> dict:
        return generate_multi_image_diagnostic_report(processed_images, lesion_type, revision_context)

    def build_local_multi_image_report(self, processed_images: list[dict], reason: str | None = None) -> dict:
        return build_local_multi_image_report(processed_images, reason)

    def generate_multi_result_reflection(self, processed_images: list[dict], report: str) -> dict:
        return generate_multi_result_reflection(processed_images, report)

    def build_local_multi_reflection(self, processed_images: list[dict], reason: str | None = None) -> dict:
        return build_local_multi_reflection(processed_images, reason)

    def answer_followup(self, context: dict, user_text: str) -> str:
        return dashscope_chat(
            build_followup_messages(context, user_text),
            max_tokens=900,
            task_name="用户多轮追问",
        )

    def fallback_followup_answer(self, context: dict, user_text: str, reason: str | None = None) -> str:
        return build_local_followup_answer(context, user_text, reason)

    def log_debug(self, message: str) -> None:
        log_debug(message)

    def log_task(self, task_name: str) -> None:
        log_task(task_name)

    def log_json(self, title: str, data) -> None:
        log_json(title, data)


def create_analysis_agent() -> EndoscopyAnalysisAgent:
    return EndoscopyAnalysisAgent(FlaskAgentRuntime())


def create_followup_agent() -> FollowupAgent:
    return FollowupAgent(FlaskAgentRuntime())


def update_context_from_state(context: dict, state: dict) -> None:
    for key in (
        "result_image_path",
        "result_image_url",
        "upload_urls",
        "upload_paths",
        "lesion_type",
        "enhancement_result",
        "lesion_result",
        "classification",
        "vision_result",
        "lesion_analysis",
        "knowledge_context",
        "report",
        "report_json",
        "reflection",
        "processed_images",
    ):
        if key in state:
            context[key] = state.get(key)
    context["steps"] = state.get("events", context.get("steps", []))


def stream_uploaded_image_for_batch(run_id: str, item: dict, lesion_type: str):
    runtime = FlaskAgentRuntime()
    upload_path = Path(item["upload_path"])
    output_dir = upload_path.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    source_label = "医生手动选帧" if item.get("source_type") == "video_manual" else ("自动抽帧" if item.get("source_type") == "video" else "上传图片")

    def add_step(step_type: str, title: str, body: str, *, meta: dict | None = None) -> dict:
        step = make_step(step_type, title, body, meta=meta)
        events.append(step)
        return step

    if not runtime.validate_image(upload_path):
        step = add_step("error", "图片校验失败", f"{item['filename']} 不是有效图像，已跳过这张图片。")
        yield step
        return {
            "ok": False,
            "error": "上传文件不是有效图像。",
            "stop_reason": "invalid_image",
            "events": events,
        }

    yield add_step("check", "图片校验通过", f"{item['filename']} 文件格式与图像完整性检查通过。")

    try:
        classification = runtime.classify_endoscopy(upload_path)
        warning_event = None
    except Exception as exc:
        user_message = getattr(exc, "user_message", str(exc))
        technical_message = getattr(exc, "technical_message", str(exc))
        runtime.log_debug(f"stream image classification fallback: {technical_message}")
        classification = runtime.build_local_classification_fallback(user_message)
        warning_event = add_step("warning", "AI 预检降级", classification["reason"])

    if warning_event is not None:
        yield warning_event

    if not classification.get("is_endoscopy"):
        yield add_step(
            "classification",
            "AI 预检判断",
            f"{item['filename']} 未通过内镜医学图像预检，已跳过本地视觉处理。依据：{classification.get('reason') or '缺少典型内镜图像特征'}。",
        )
        return {
            "ok": False,
            "classification": classification,
            "error": "图片未通过内镜图像预检。",
            "stop_reason": "not_endoscopy",
            "events": events,
        }

    yield add_step(
        "classification",
        "AI 预检判断",
        f"{item['filename']} 通过内镜医学图像预检。依据：{classification.get('reason') or '图像特征符合内镜视野'}。",
    )

    try:
        lesion_result = runtime.diagnose_original_image(str(upload_path), str(output_dir), lesion_type)
    except Exception as exc:
        yield add_step("error", "病灶定位失败", f"{item['filename']} 的 YOLO/MedSAM 处理失败：{exc}")
        return {
            "ok": False,
            "classification": classification,
            "error": "本地病灶定位失败。",
            "stop_reason": "lesion_error",
            "events": events,
        }

    if lesion_result.get("status") != "success":
        yield add_step(
            "error",
            "本地模型返回失败",
            f"{item['filename']} 的病灶定位失败：{lesion_result.get('message', '未知错误')}",
        )
        return {
            "ok": False,
            "classification": classification,
            "lesion_result": lesion_result,
            "error": "本地病灶定位失败。",
            "stop_reason": "lesion_error",
            "events": events,
        }

    lesion_image_path = lesion_result.get("lesion_image_path")
    lesion_body = (
        f"YOLO/MedSAM 已在 {item['filename']} 上完成病灶定位，检测框坐标：{lesion_result.get('bbox')}。"
        if lesion_result.get("polyp_detected")
        else f"YOLO/MedSAM 已在 {item['filename']} 上完成检查，未检测到明显疑似病灶区域。"
    )
    yield add_step("vision", "病灶位置标注完成", lesion_body)

    try:
        enhancement_result = runtime.enhance_input_image(str(upload_path), str(output_dir))
    except Exception as exc:
        yield add_step("error", "IRetinex 增强失败", f"{item['filename']} 增强失败：{exc}")
        return {
            "ok": False,
            "classification": classification,
            "lesion_result": lesion_result,
            "error": "IRetinex 增强失败。",
            "stop_reason": "enhancement_error",
            "events": events,
        }

    fusion_path = Path(enhancement_result.get("fusion_path") or "")
    if enhancement_result.get("status") != "success" or not fusion_path.is_file():
        yield add_step(
            "error",
            "IRetinex 增强失败",
            f"{item['filename']} 未生成可读取增强图：{enhancement_result.get('message', '未知错误')}",
        )
        return {
            "ok": False,
            "classification": classification,
            "lesion_result": lesion_result,
            "enhancement_result": enhancement_result,
            "error": "IRetinex 增强失败。",
            "stop_reason": "enhancement_error",
            "events": events,
        }

    yield add_step(
        "enhancement",
        "IRetinex 增强完成",
        f"{item['filename']} 已完成 IRetinex 增强。",
    )

    try:
        vision_result = runtime.annotate_enhanced_image(enhancement_result, lesion_result, str(output_dir))
    except Exception as exc:
        yield add_step("error", "病灶标注失败", f"{item['filename']} 增强图标注失败：{exc}")
        return {
            "ok": False,
            "classification": classification,
            "lesion_result": lesion_result,
            "enhancement_result": enhancement_result,
            "error": "本地病灶标注失败。",
            "stop_reason": "annotation_error",
            "events": events,
        }

    result_image_path = Path(vision_result.get("result_image_path") or "").resolve()
    if vision_result.get("status") != "success" or not result_image_path.is_file():
        yield add_step(
            "error",
            "本地标注返回失败",
            f"{item['filename']} 未生成可读取结果图：{vision_result.get('message', '未知错误')}",
        )
        return {
            "ok": False,
            "classification": classification,
            "lesion_result": lesion_result,
            "enhancement_result": enhancement_result,
            "vision_result": vision_result,
            "error": "本地病灶标注失败。",
            "stop_reason": "annotation_error",
            "events": events,
        }

    result_image_url = runtime.build_result_image_url(run_id, upload_path, result_image_path)
    result_body = (
        f"已将 {item['filename']} 原图中的病灶坐标和分割区域标注到 IRetinex 增强图上。"
        if vision_result.get("polyp_detected")
        else f"已生成 {item['filename']} 的 IRetinex 增强图；未检测到明显疑似病灶区域。"
    )
    result_title = "增强图病灶标注完成" if vision_result.get("polyp_detected") else "增强图生成完成"
    result_caption = (
        f"图片 {item['index']}（{source_label}）增强图病灶标注结果"
        if vision_result.get("polyp_detected")
        else f"图片 {item['index']}（{source_label}）增强图（未检出明显病灶）"
    )
    comparison_images = [
        {
            "url": item.get("upload_url"),
            "caption": "原图",
        }
    ]
    if lesion_image_path and Path(lesion_image_path).is_file():
        comparison_images.append(
            {
                "url": runtime.build_result_image_url(run_id, upload_path, Path(lesion_image_path).resolve()),
                "caption": "病灶标记图",
            }
        )
    comparison_images.extend(
        [
            {
                "url": runtime.build_result_image_url(run_id, upload_path, fusion_path.resolve()),
                "caption": "IRetinex 增强图",
            },
            {
                "url": result_image_url,
                "caption": "增强后标记图",
            },
        ]
    )
    yield add_step(
        "vision",
        result_title,
        result_body,
        meta={
            "comparison_title": f"处理结果 - 图片 {item['index']}（{source_label}）",
            "comparison_images": comparison_images,
            "image_url": result_image_url,
            "image_caption": result_caption,
        },
    )

    try:
        lesion_analysis = runtime.generate_lesion_analysis(result_image_path, vision_result)
        lesion_type_text = lesion_analysis.get("lesion_type") or "未知"
        confidence = lesion_analysis.get("confidence_level") or "未标注"
        lesion_count = lesion_analysis.get("lesion_count")
        yield add_step(
            "analysis",
            "结构化病灶分析完成",
            f"{item['filename']} 已输出结构化病灶信息：lesion_type={lesion_type_text}，confidence_level={confidence}，lesion_count={lesion_count}。",
        )
    except Exception as exc:
        user_message = getattr(exc, "user_message", str(exc))
        technical_message = getattr(exc, "technical_message", str(exc))
        runtime.log_debug(f"stream image lesion analysis fallback: {technical_message}")
        lesion_analysis = None
        yield add_step(
            "warning",
            "结构化病灶分析降级",
            f"{item['filename']} 的结构化病灶分析暂时不可用，后续报告将仅基于本地视觉结果汇总。原因：{user_message}",
        )

    return {
        "ok": True,
        "classification": classification,
        "lesion_result": lesion_result,
        "enhancement_result": enhancement_result,
        "vision_result": vision_result,
        "lesion_analysis": lesion_analysis,
        "result_image_path": str(result_image_path),
        "result_image_url": result_image_url,
        "events": events,
    }


def process_uploaded_image_for_batch(run_id: str, item: dict, lesion_type: str) -> dict:
    runtime = FlaskAgentRuntime()
    upload_path = Path(item["upload_path"])
    output_dir = upload_path.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    source_label = "医生手动选帧" if item.get("source_type") == "video_manual" else ("自动抽帧" if item.get("source_type") == "video" else "上传图片")

    if not runtime.validate_image(upload_path):
        return {
            "ok": False,
            "error": "上传文件不是有效图像。",
            "stop_reason": "invalid_image",
            "events": [
                make_step("error", "图片校验失败", f"{item['filename']} 不是有效图像，已跳过这张图片。")
            ],
        }

    events.append(make_step("check", "图片校验通过", f"{item['filename']} 文件格式与图像完整性检查通过。"))

    try:
        lesion_result = runtime.diagnose_original_image(str(upload_path), str(output_dir), lesion_type)
    except Exception as exc:
        return {
            "ok": False,
            "error": "本地病灶定位失败。",
            "stop_reason": "vision_error",
            "events": events
            + [make_step("error", "病灶定位失败", f"{item['filename']} 的 YOLO/MedSAM 处理失败：{exc}")],
        }

    if lesion_result.get("status") != "success":
        return {
            "ok": False,
            "lesion_result": lesion_result,
            "error": "本地病灶定位失败。",
            "stop_reason": "vision_error",
            "events": events
            + [
                make_step(
                    "error",
                    "本地模型返回失败",
                    f"{item['filename']} 的病灶定位失败：{lesion_result.get('message', '未知错误')}",
                )
            ],
        }

    lesion_image_path = lesion_result.get("lesion_image_path")
    lesion_body = (
        f"YOLO/MedSAM 已在 {item['filename']} 上完成病灶定位，检测框坐标：{lesion_result.get('bbox')}。"
        if lesion_result.get("polyp_detected")
        else f"YOLO/MedSAM 已在 {item['filename']} 上完成检查，未检测到明显疑似病灶区域。"
    )
    events.append(make_step("vision", "病灶位置标注完成", lesion_body))

    try:
        enhancement_result = runtime.enhance_input_image(str(upload_path), str(output_dir))
    except Exception as exc:
        return {
            "ok": False,
            "lesion_result": lesion_result,
            "error": "IRetinex 增强失败。",
            "stop_reason": "vision_error",
            "events": events + [make_step("error", "IRetinex 增强失败", f"{item['filename']} 增强失败：{exc}")],
        }

    fusion_path = Path(enhancement_result.get("fusion_path") or "")
    if enhancement_result.get("status") != "success" or not fusion_path.is_file():
        return {
            "ok": False,
            "lesion_result": lesion_result,
            "enhancement_result": enhancement_result,
            "error": "IRetinex 增强失败。",
            "stop_reason": "vision_error",
            "events": events
            + [
                make_step(
                    "error",
                    "IRetinex 增强失败",
                    f"{item['filename']} 未生成可读取增强图：{enhancement_result.get('message', '未知错误')}",
                )
            ],
        }

    events.append(
        make_step(
            "enhancement",
            "IRetinex 增强完成",
            f"{item['filename']} 已完成 IRetinex 增强。",
        )
    )

    try:
        vision_result = runtime.annotate_enhanced_image(enhancement_result, lesion_result, str(output_dir))
    except Exception as exc:
        return {
            "ok": False,
            "lesion_result": lesion_result,
            "enhancement_result": enhancement_result,
            "error": "本地病灶标注失败。",
            "stop_reason": "vision_error",
            "events": events + [make_step("error", "病灶标注失败", f"{item['filename']} 增强图标注失败：{exc}")],
        }

    result_image_path = Path(vision_result.get("result_image_path") or "").resolve()
    if vision_result.get("status") != "success" or not result_image_path.is_file():
        return {
            "ok": False,
            "lesion_result": lesion_result,
            "enhancement_result": enhancement_result,
            "vision_result": vision_result,
            "error": "本地病灶标注失败。",
            "stop_reason": "vision_error",
            "events": events
            + [
                make_step(
                    "error",
                    "本地标注返回失败",
                    f"{item['filename']} 未生成可读取结果图：{vision_result.get('message', '未知错误')}",
                )
            ],
        }

    result_image_url = runtime.build_result_image_url(run_id, upload_path, result_image_path)
    result_body = (
        f"已将 {item['filename']} 原图中的病灶坐标和分割区域标注到 IRetinex 增强图上。"
        if vision_result.get("polyp_detected")
        else f"已生成 {item['filename']} 的 IRetinex 增强图；未检测到明显疑似病灶区域。"
    )
    result_title = "增强图病灶标注完成" if vision_result.get("polyp_detected") else "增强图生成完成"
    result_caption = (
        f"图片 {item['index']}（{source_label}）增强图病灶标注结果"
        if vision_result.get("polyp_detected")
        else f"图片 {item['index']}（{source_label}）增强图（未检出明显病灶）"
    )
    comparison_images = [
        {
            "url": item.get("upload_url"),
            "caption": "原图",
        }
    ]
    if lesion_image_path and Path(lesion_image_path).is_file():
        comparison_images.append(
            {
                "url": runtime.build_result_image_url(run_id, upload_path, Path(lesion_image_path).resolve()),
                "caption": "病灶标记图",
            }
        )
    comparison_images.extend(
        [
            {
                "url": runtime.build_result_image_url(run_id, upload_path, fusion_path.resolve()),
                "caption": "IRetinex 增强图",
            },
            {
                "url": result_image_url,
                "caption": "增强后标记图",
            },
        ]
    )
    events.append(
        make_step(
            "vision",
            result_title,
            result_body,
            meta={
                "comparison_title": f"处理结果 - 图片 {item['index']}（{source_label}）",
                "comparison_images": comparison_images,
                "image_url": result_image_url,
                "image_caption": result_caption,
            },
        )
    )

    return {
        "ok": True,
        "lesion_result": lesion_result,
        "enhancement_result": enhancement_result,
        "vision_result": vision_result,
        "result_image_path": str(result_image_path),
        "result_image_url": result_image_url,
        "events": events,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze-stream", methods=["POST"])
def analyze_stream():
    uploaded_files = get_uploaded_images()
    lesion_type = normalize_requested_lesion_type(request.form.get("lesion_type"))
    upload_mode = "video" if is_video_upload_mode() else "image"
    source_type = "video" if upload_mode == "video" else "image"
    pre_steps: list[dict] = []
    original_video_path = None
    original_video_url = None
    if not uploaded_files:
        return Response(
            make_stream_event("done", ok=False, error="请先选择至少一个文件。", steps=[]),
            status=400,
            mimetype="application/x-ndjson",
        )

    if upload_mode == "video":
        if len(uploaded_files) != 1:
            return Response(
                make_stream_event("done", ok=False, error="视频模式一次只能上传一个视频。", steps=[]),
                status=400,
                mimetype="application/x-ndjson",
            )
        video_file = uploaded_files[0]
        if not allowed_video_file(video_file.filename):
            return Response(
                make_stream_event("done", ok=False, error="仅支持 mp4、avi、mov、mkv、webm 格式的视频。", steps=[]),
                status=400,
                mimetype="application/x-ndjson",
            )
        try:
            run_id, video_path, video_url = save_video_upload(video_file)
            original_video_path = str(video_path)
            original_video_url = video_url
            pre_steps.append(make_step("upload", "收到视频", f"我已接收并保存用户上传的视频：{video_path.name}。"))
            saved_items = []
        except Exception as exc:
            app.logger.exception("Failed to prepare uploaded video before streaming: %s", exc)
            return Response(
                make_stream_event("done", ok=False, error=SERVICE_UNAVAILABLE_MESSAGE, steps=pre_steps),
                status=500,
                mimetype="application/x-ndjson",
            )
    else:
        invalid_files = [file.filename for file in uploaded_files if not allowed_file(file.filename)]
        if invalid_files:
            return Response(
                make_stream_event("done", ok=False, error="仅支持 png、jpg、jpeg、bmp、webp 格式的图像。", steps=[]),
                status=400,
                mimetype="application/x-ndjson",
            )
        try:
            run_id, saved_items = save_uploads(uploaded_files)
        except Exception as exc:
            app.logger.exception("Failed to save uploaded images before streaming: %s", exc)
            return Response(
                make_stream_event("done", ok=False, error=SERVICE_UNAVAILABLE_MESSAGE, steps=[]),
                status=500,
                mimetype="application/x-ndjson",
            )

    context = {
        "run_id": run_id,
        "upload_path": str(saved_items[0]["upload_path"]) if saved_items else original_video_path,
        "upload_url": saved_items[0]["upload_url"] if saved_items else original_video_url,
        "upload_paths": [str(item["upload_path"]) for item in saved_items],
        "upload_urls": [item["upload_url"] for item in saved_items],
        "lesion_type": lesion_type,
        "source_type": source_type,
        "original_video_path": original_video_path,
        "original_video_url": original_video_url,
        "processed_images": [],
        "result_image_path": None,
        "result_image_url": None,
        "classification": None,
        "enhancement_result": None,
        "lesion_result": None,
        "vision_result": None,
        "lesion_analysis": None,
        "knowledge_context": None,
        "report": None,
        "reflection": None,
        "manual_frames": [],
        "manual_frame_counter": 0,
        "manual_frames_closed": source_type != "video",
        "steps": list(pre_steps),
        "chat_history": [],
    }
    ANALYSIS_CONTEXTS[run_id] = context

    @stream_with_context
    def incremental_event_stream():
        log_task("收到新的逐步流式分析请求")
        emitted_count = 0
        all_steps: list[dict] = list(pre_steps)
        processed_images: list[dict] = []
        analysis_items: list[dict] = list(saved_items)
        framework = "incremental-stream"
        is_multi_image = len(analysis_items) > 1 or source_type == "video"

        yield make_stream_event(
            "context",
            run_id=run_id,
            upload_url=context["upload_url"],
            upload_urls=context["upload_urls"],
            lesion_type=lesion_type,
            framework=framework,
            source_type=source_type,
            original_video_url=original_video_url,
        )

        try:
            for step in pre_steps:
                yield make_stream_event("step", step=step, index=emitted_count)
                emitted_count += 1

            if source_type == "video":
                auto_items: list[dict] = []
                try:
                    video_path = Path(original_video_path or "")
                    upload_dir = video_path.parent.resolve()
                    keyframe_root = upload_dir / "keyframes"
                    keyframes = extract_video_keyframes(video_path, lesion_type, keyframe_root)
                    step = make_step("video", "视频关键帧抽取完成", f"已从视频中抽取 {len(keyframes)} 张候选关键帧。")
                    all_steps.append(step)
                    context["steps"] = all_steps
                    yield make_stream_event("step", step=step, index=emitted_count)
                    emitted_count += 1

                    selected_keyframes: list[Path] = []
                    positive_keyframe_count = 0
                    forced_keyframe = None
                    frame_screen_result = None
                    if keyframes:
                        try:
                            frame_screen_result = screen_keyframes_for_lesion(keyframes, lesion_type)
                            selected_keyframes = list(frame_screen_result.get("selected_paths") or [])
                            positive_keyframe_count = int(frame_screen_result.get("positive_count") or 0)
                            forced_keyframe = frame_screen_result.get("forced_path")
                        except Exception as exc:
                            user_message = getattr(exc, "user_message", str(exc))
                            log_debug(f"关键帧多图 AI 初筛失败，改用强制保留兜底：原因：{user_message}")
                            step = make_step("warning", "关键帧多图初筛降级", f"多图综合初筛暂时不可用，已改用兜底保留 1 张自动关键帧。原因：{user_message}")
                            all_steps.append(step)
                            context["steps"] = all_steps
                            yield make_stream_event("step", step=step, index=emitted_count)
                            emitted_count += 1
                            try:
                                forced_keyframe = select_required_keyframe_for_video(keyframes, lesion_type)
                            except Exception as fallback_exc:
                                log_debug(f"关键帧强制保留失败，使用中间帧兜底：原因：{getattr(fallback_exc, 'user_message', fallback_exc)}")
                                forced_keyframe = keyframes[len(keyframes) // 2]
                            if forced_keyframe:
                                selected_keyframes = [forced_keyframe]

                    step = make_step(
                        "video",
                        "关键帧疑似病灶初筛完成",
                        (
                            f"候选关键帧 {len(keyframes)} 张，经多图综合比较，其中 {positive_keyframe_count} 张被 AI 初筛为可能存在病灶。"
                            + ("已保留 1 张相对最可疑自动关键帧。" if forced_keyframe else "")
                        ),
                    )
                    all_steps.append(step)
                    context["steps"] = all_steps
                    yield make_stream_event("step", step=step, index=emitted_count)
                    emitted_count += 1

                    if forced_keyframe:
                        step = make_step(
                            "video",
                            "关键帧强制保留",
                            f"本轮自动初筛没有明确阳性关键帧，系统已按展示与复核需要保留 1 张相对最可疑的关键帧：{forced_keyframe.name}。",
                        )
                        all_steps.append(step)
                        context["steps"] = all_steps
                        yield make_stream_event("step", step=step, index=emitted_count)
                        emitted_count += 1

                    if selected_keyframes:
                        auto_items = build_keyframe_upload_items(run_id, upload_dir, selected_keyframes, video_path)
                except Exception as exc:
                    app.logger.exception("Video keyframe extraction failed during streaming: %s", exc)
                    step = make_step("warning", "视频自动抽帧降级", f"后台自动抽帧暂时不可用，将继续等待并处理医生手动截图。原因：{exc}")
                    all_steps.append(step)
                    context["steps"] = all_steps
                    yield make_stream_event("step", step=step, index=emitted_count)
                    emitted_count += 1

                step = make_step(
                    "video",
                    "等待医生手动选帧",
                    "后台自动抽帧已完成等待阶段。请在视频面板中按 Enter 保留可疑画面，点击“完成手动选帧”后，手动截图会与自动关键帧一起进入增强和病灶识别流程；手动截图不会再经过 LLM 初筛。",
                )
                all_steps.append(step)
                context["steps"] = all_steps

                wait_for_manual_video_frames(context)
                manual_items = collect_manual_video_frame_items(context, start_index=len(auto_items) + 1)
                step = make_step(
                    "video",
                    "视频分析队列已合并",
                    f"最终进入增强和病灶识别流程的图片共 {len(auto_items) + len(manual_items)} 张：自动抽帧经 LLM 初筛 {len(auto_items)} 张，医生手动选帧 {len(manual_items)} 张。",
                )
                all_steps.append(step)
                context["steps"] = all_steps
                if manual_items:
                    step = make_step("video", "手动截图已加入分析队列", f"医生手动保留了 {len(manual_items)} 张视频截图。")
                    all_steps.append(step)
                    context["steps"] = all_steps

                analysis_items = auto_items + manual_items
                for index, item in enumerate(analysis_items, start=1):
                    item["index"] = index
                context["upload_paths"] = [str(item["upload_path"]) for item in analysis_items]
                context["upload_urls"] = [item["upload_url"] for item in analysis_items]
                if analysis_items:
                    context["upload_path"] = str(analysis_items[0]["upload_path"])
                    context["upload_url"] = analysis_items[0]["upload_url"]
                is_multi_image = len(analysis_items) > 1 or source_type == "video"

            if not analysis_items:
                yield make_stream_event(
                    "done",
                    ok=False,
                    run_id=run_id,
                    upload_url=context.get("upload_url"),
                    upload_urls=context.get("upload_urls"),
                    error="本轮视频没有得到可分析的自动关键帧或手动截图。",
                    processed_images=[],
                    steps=all_steps,
                    framework=framework,
                    source_type=source_type,
                    original_video_url=original_video_url,
                )
                return

            for item in analysis_items:
                if is_multi_image:
                    source_label = "医生手动选帧" if item.get("source_type") == "video_manual" else ("自动抽帧" if item.get("source_type") == "video" else "上传图片")
                    header_step = make_step(
                        "batch",
                        f"开始处理图片 {item['index']}/{len(analysis_items)}",
                        f"正在单独处理：{item['filename']}（{source_label}）。YOLO/MedSAM 与 IRetinex 会在这张图上分步运行。",
                    )
                    all_steps.append(header_step)
                    context["steps"] = all_steps
                    yield make_stream_event("step", step=header_step, index=emitted_count)
                    emitted_count += 1

                image_stream = stream_uploaded_image_for_batch(run_id, item, lesion_type)
                while True:
                    try:
                        step = next(image_stream)
                    except StopIteration as stop:
                        latest_state = stop.value or {"ok": False, "error": "图像处理提前结束。", "events": []}
                        break

                    step = dict(step)
                    if is_multi_image:
                        step["title"] = f"图片 {item['index']} - {step.get('title', '分析步骤')}"
                    all_steps.append(step)
                    context["steps"] = all_steps
                    yield make_stream_event("step", step=step, index=emitted_count)
                    emitted_count += 1

                image_response = {
                    "ok": latest_state.get("ok", True),
                    "error": latest_state.get("error"),
                    "lesion_type": lesion_type,
                    "upload_url": item["upload_url"],
                    "classification": latest_state.get("classification"),
                    "enhancement_result": latest_state.get("enhancement_result"),
                    "lesion_result": latest_state.get("lesion_result"),
                    "vision_result": latest_state.get("vision_result"),
                    "lesion_analysis": latest_state.get("lesion_analysis"),
                    "result_image_url": latest_state.get("result_image_url"),
                    "source_type": item.get("source_type", source_type),
                }
                processed_item = {
                    "index": item["index"],
                    "filename": item["filename"],
                    "upload_path": str(item["upload_path"]),
                    "upload_url": item["upload_url"],
                    "result_image_path": latest_state.get("result_image_path"),
                    "result_image_url": latest_state.get("result_image_url"),
                    "source_type": item.get("source_type", source_type),
                    "original_video_path": original_video_path,
                    "original_video_url": original_video_url,
                    "ok": latest_state.get("ok", True),
                    "error": latest_state.get("error"),
                    "state": image_response,
                }
                processed_images.append(processed_item)
                context["processed_images"] = processed_images
                context["result_image_path"] = processed_item.get("result_image_path") or context.get("result_image_path")
                context["result_image_url"] = processed_item.get("result_image_url") or context.get("result_image_url")
                for key in ("classification", "enhancement_result", "lesion_result", "vision_result", "lesion_analysis"):
                    if latest_state.get(key) is not None:
                        context[key] = latest_state.get(key)

            ok_images = [
                item
                for item in processed_images
                if item.get("ok", True) and item.get("result_image_path")
            ]
            if not ok_images:
                yield make_stream_event(
                    "done",
                    ok=False,
                    run_id=run_id,
                    upload_url=context.get("upload_url"),
                    upload_urls=context.get("upload_urls"),
                    error="所有图片都未能完成本地处理。",
                    processed_images=processed_images,
                    steps=all_steps,
                    framework=framework,
                    source_type=source_type,
                    original_video_url=original_video_url,
                )
                return

            try:
                combined_report_json = generate_multi_image_diagnostic_report(ok_images, lesion_type, source_type=source_type)
            except Exception as exc:
                user_message = getattr(exc, "user_message", str(exc))
                technical_message = getattr(exc, "technical_message", str(exc))
                log_debug(f"逐步流式综合报告降级：{technical_message}")
                warning_step = make_step(
                    "warning",
                    "综合报告降级",
                    f"外部 AI 综合分析暂时不可用，已生成本地汇总。原因：{user_message}",
                )
                all_steps.append(warning_step)
                context["steps"] = all_steps
                yield make_stream_event("step", step=warning_step, index=emitted_count)
                emitted_count += 1
                combined_report_json = build_local_multi_image_report(ok_images, user_message, source_type=source_type, lesion_type=lesion_type)
            combined_report = format_structured_report(combined_report_json)

            report_title = "多图综合辅助报告生成完成" if is_multi_image else "辅助报告生成完成"
            report_step = make_step("report", report_title, combined_report, meta={"report_json": combined_report_json})
            all_steps.append(report_step)
            context["steps"] = all_steps
            context["report_json"] = combined_report_json
            context["report"] = combined_report
            yield make_stream_event("step", step=report_step, index=emitted_count)
            emitted_count += 1

            try:
                reflection = generate_multi_result_reflection(ok_images, combined_report)
            except Exception as exc:
                user_message = getattr(exc, "user_message", str(exc))
                technical_message = getattr(exc, "technical_message", str(exc))
                log_debug(f"逐步流式结果自检降级：{technical_message}")
                reflection = build_local_multi_reflection(ok_images, user_message)

            context["reflection"] = reflection
            confidence_text = "未评估" if reflection.get("confidence") is None else f"{reflection['confidence']}%"
            uncertainty_text = "；".join(reflection.get("uncertainties") or []) or "暂无额外列出"
            reflection_step = make_step(
                "reflection",
                "结果自检完成",
                (
                    f"粗略置信度：{confidence_text}（{reflection.get('confidence_label') or '未标注'}）。"
                    f"{reflection.get('reflection') or ''}\n不确定因素：{uncertainty_text}\n{reflection.get('suggestion') or ''}"
                ).strip(),
            )
            all_steps.append(reflection_step)
            context["steps"] = all_steps
            yield make_stream_event("step", step=reflection_step, index=emitted_count)
            emitted_count += 1

            if should_revise_report_after_reflection(reflection):
                try:
                    combined_report_json = generate_multi_image_diagnostic_report(
                        ok_images,
                        lesion_type,
                        reflection,
                        source_type=source_type,
                    )
                except Exception as exc:
                    user_message = getattr(exc, "user_message", str(exc))
                    technical_message = getattr(exc, "technical_message", str(exc))
                    log_debug(f"逐步流式反思修订报告降级：{technical_message}")
                    warning_step = make_step(
                        "warning",
                        "反思修订报告降级",
                        f"外部 AI 反思修订暂时不可用，已根据自检内容生成本地汇总。原因：{user_message}",
                    )
                    all_steps.append(warning_step)
                    context["steps"] = all_steps
                    yield make_stream_event("step", step=warning_step, index=emitted_count)
                    emitted_count += 1
                    combined_report_json = build_local_multi_image_report(ok_images, user_message, source_type=source_type, lesion_type=lesion_type)
                combined_report = format_structured_report(combined_report_json)

                revision_step = make_step("report", "辅助报告已根据结果自检修订", combined_report, meta={"report_json": combined_report_json})
                all_steps.append(revision_step)
                context["steps"] = all_steps
                context["report_json"] = combined_report_json
                context["report"] = combined_report
                yield make_stream_event("step", step=revision_step, index=emitted_count)
                emitted_count += 1

            last_result = next((item for item in reversed(ok_images) if item.get("result_image_url")), None)
            if last_result:
                context["result_image_path"] = last_result.get("result_image_path")
                context["result_image_url"] = last_result.get("result_image_url")

            yield make_stream_event(
                "done",
                ok=True,
                run_id=run_id,
                upload_url=context.get("upload_url"),
                upload_urls=context.get("upload_urls"),
                result_image_url=context.get("result_image_url"),
                processed_images=processed_images,
                report=combined_report,
                report_json=context.get("report_json"),
                reflection=reflection,
                steps=all_steps,
                framework=framework,
                source_type=source_type,
                original_video_url=original_video_url,
            )
        except Exception as exc:
            app.logger.exception("Unhandled incremental streaming analysis error: %s", exc)
            step = make_step("error", "分析服务异常", SERVICE_UNAVAILABLE_MESSAGE)
            all_steps.append(step)
            context["steps"] = all_steps
            yield make_stream_event("step", step=step, index=emitted_count)
            yield make_stream_event(
                "done",
                ok=False,
                run_id=run_id,
                upload_url=context.get("upload_url"),
                upload_urls=context.get("upload_urls"),
                error=SERVICE_UNAVAILABLE_MESSAGE,
                processed_images=processed_images,
                steps=all_steps,
                framework=framework,
                source_type=source_type,
                original_video_url=original_video_url,
            )

    return Response(
        incremental_event_stream(),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

    if len(uploaded_files) > 1:
        try:
            run_id, saved_items = save_uploads(uploaded_files)
        except Exception as exc:
            app.logger.exception("Failed to save uploaded images before streaming: %s", exc)
            return Response(
                make_stream_event("done", ok=False, error=SERVICE_UNAVAILABLE_MESSAGE, steps=[]),
                status=500,
                mimetype="application/x-ndjson",
            )

        context = {
            "run_id": run_id,
            "upload_path": str(saved_items[0]["upload_path"]) if saved_items else None,
            "upload_url": saved_items[0]["upload_url"] if saved_items else None,
            "upload_paths": [str(item["upload_path"]) for item in saved_items],
            "upload_urls": [item["upload_url"] for item in saved_items],
            "lesion_type": lesion_type,
            "processed_images": [],
            "result_image_path": None,
            "result_image_url": None,
            "report": None,
            "reflection": None,
            "steps": [],
            "chat_history": [],
        }
        ANALYSIS_CONTEXTS[run_id] = context

        @stream_with_context
        def langgraph_multi_image_event_stream():
            log_task("收到新的多图 LangGraph 智能体流式分析请求")
            agent = create_analysis_agent()
            emitted_count = 0
            latest_state = create_initial_state(
                run_id,
                saved_items[0]["upload_path"],
                saved_items[0]["upload_url"],
                lesion_type,
                upload_items=saved_items,
            )

            yield make_stream_event(
                "context",
                run_id=run_id,
                upload_urls=context["upload_urls"],
                lesion_type=lesion_type,
                framework=agent.framework,
            )

            try:
                for latest_state in agent.stream(latest_state):
                    update_context_from_state(context, latest_state)
                    events = latest_state.get("events", [])
                    for step in events[emitted_count:]:
                        yield make_stream_event("step", step=step, index=emitted_count)
                        emitted_count += 1

                update_context_from_state(context, latest_state)
                response = build_analysis_response(latest_state)
                response["framework"] = agent.framework
                yield make_stream_event("done", **response)
            except Exception as exc:
                app.logger.exception("Unhandled multi-image LangGraph streaming analysis error: %s", exc)
                step = make_step("error", "多图分析服务异常", SERVICE_UNAVAILABLE_MESSAGE)
                context.setdefault("steps", []).append(step)
                yield make_stream_event("step", step=step, index=emitted_count)
                yield make_stream_event(
                    "done",
                    ok=False,
                    run_id=run_id,
                    upload_urls=context["upload_urls"],
                    error=SERVICE_UNAVAILABLE_MESSAGE,
                    steps=context.get("steps", []),
                    framework=agent.framework,
                )

        return Response(
            langgraph_multi_image_event_stream(),
            mimetype="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

        @stream_with_context
        def multi_image_event_stream():
            log_task("收到新的多图 LangGraph 智能体流式分析请求")
            framework = "langgraph"
            emitted_count = 0
            all_steps: list[dict] = []
            processed_images: list[dict] = []

            yield make_stream_event(
                "context",
                run_id=run_id,
                upload_urls=context["upload_urls"],
                lesion_type=lesion_type,
                framework=framework,
            )

            try:
                for item in saved_items:
                    source_label = "医生手动选帧" if item.get("source_type") == "video_manual" else ("自动抽帧" if item.get("source_type") == "video" else "上传图片")
                    header_step = make_step(
                        "batch",
                        f"开始处理图片 {item['index']}/{len(saved_items)}",
                        f"正在单独处理：{item['filename']}（{source_label}）。YOLO/MedSAM 与 IRetinex 会分别在这张图上运行。",
                    )
                    all_steps.append(header_step)
                    context["steps"] = all_steps
                    yield make_stream_event("step", step=header_step, index=emitted_count)
                    emitted_count += 1

                    latest_state = process_uploaded_image_for_batch(run_id, item, lesion_type)
                    for step in latest_state.get("events", []):
                        step = dict(step)
                        step["title"] = f"图片 {item['index']} - {step.get('title', '分析步骤')}"
                        all_steps.append(step)
                        context["steps"] = all_steps
                        yield make_stream_event("step", step=step, index=emitted_count)
                        emitted_count += 1

                    image_response = {
                        "ok": latest_state.get("ok", True),
                        "error": latest_state.get("error"),
                        "lesion_type": lesion_type,
                        "upload_url": item["upload_url"],
                        "enhancement_result": latest_state.get("enhancement_result"),
                        "lesion_result": latest_state.get("lesion_result"),
                        "vision_result": latest_state.get("vision_result"),
                        "result_image_url": latest_state.get("result_image_url"),
                    }
                    processed_item = {
                        "index": item["index"],
                        "filename": item["filename"],
                        "upload_path": str(item["upload_path"]),
                        "upload_url": item["upload_url"],
                        "result_image_path": latest_state.get("result_image_path"),
                        "result_image_url": latest_state.get("result_image_url"),
                        "ok": latest_state.get("ok", True),
                        "error": latest_state.get("error"),
                        "state": image_response,
                    }
                    processed_images.append(processed_item)
                    context["processed_images"] = processed_images
                    context["result_image_path"] = processed_item.get("result_image_path")
                    context["result_image_url"] = processed_item.get("result_image_url")

                ok_images = [
                    item
                    for item in processed_images
                    if item.get("ok", True) and item.get("result_image_path")
                ]
                if not ok_images:
                    yield make_stream_event(
                        "done",
                        ok=False,
                        run_id=run_id,
                        upload_urls=context["upload_urls"],
                        error="所有图片都未能完成本地处理。",
                        processed_images=processed_images,
                        steps=all_steps,
                        framework=framework,
                    )
                    return

                try:
                    combined_report_json = generate_multi_image_diagnostic_report(ok_images, lesion_type)
                except Exception as exc:
                    user_message = getattr(exc, "user_message", str(exc))
                    technical_message = getattr(exc, "technical_message", str(exc))
                    log_debug(f"多图综合报告降级：{technical_message}")
                    warning_step = make_step("warning", "多图综合报告降级", f"外部 AI 多图综合分析暂时不可用，已生成本地汇总。原因：{user_message}")
                    all_steps.append(warning_step)
                    yield make_stream_event("step", step=warning_step, index=emitted_count)
                    emitted_count += 1
                    combined_report_json = build_local_multi_image_report(ok_images, user_message, lesion_type=lesion_type)
                combined_report = format_structured_report(combined_report_json)

                report_step = make_step("report", "多图综合辅助报告生成完成", combined_report, meta={"report_json": combined_report_json})
                all_steps.append(report_step)
                context["steps"] = all_steps
                context["report_json"] = combined_report_json
                context["report"] = combined_report
                yield make_stream_event("step", step=report_step, index=emitted_count)

                yield make_stream_event(
                    "done",
                    ok=True,
                    run_id=run_id,
                    upload_urls=context["upload_urls"],
                    result_image_url=context.get("result_image_url"),
                    processed_images=processed_images,
                    report=combined_report,
                    report_json=context.get("report_json"),
                    steps=all_steps,
                    framework=framework,
                )
            except Exception as exc:
                app.logger.exception("Unhandled multi-image streaming analysis error: %s", exc)
                step = make_step("error", "多图分析服务异常", SERVICE_UNAVAILABLE_MESSAGE)
                all_steps.append(step)
                context["steps"] = all_steps
                yield make_stream_event("step", step=step, index=emitted_count)
                yield make_stream_event(
                    "done",
                    ok=False,
                    run_id=run_id,
                    upload_urls=context["upload_urls"],
                    error=SERVICE_UNAVAILABLE_MESSAGE,
                    processed_images=processed_images,
                    steps=all_steps,
                    framework=framework,
                )

        return Response(
            multi_image_event_stream(),
            mimetype="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    uploaded_file = uploaded_files[0]
    try:
        run_id, upload_path = save_upload(uploaded_file)
        upload_url = url_for("upload_file", run_id=run_id, filename=upload_path.name)
    except Exception as exc:
        app.logger.exception("Failed to save uploaded image before streaming: %s", exc)
        return Response(
            make_stream_event("done", ok=False, error=SERVICE_UNAVAILABLE_MESSAGE, steps=[]),
            status=500,
            mimetype="application/x-ndjson",
        )

    context = {
        "run_id": run_id,
        "upload_path": str(upload_path),
        "upload_url": upload_url,
        "lesion_type": lesion_type,
        "enhancement_result": None,
        "lesion_result": None,
        "result_image_path": None,
        "result_image_url": None,
        "classification": None,
        "vision_result": None,
        "lesion_analysis": None,
        "knowledge_context": None,
        "report": None,
        "reflection": None,
        "steps": [],
        "chat_history": [],
    }
    ANALYSIS_CONTEXTS[run_id] = context

    @stream_with_context
    def agent_event_stream():
        log_task("收到新的 LangGraph 智能体流式分析请求")
        agent = create_analysis_agent()
        emitted_count = 0
        latest_state = create_initial_state(run_id, upload_path, upload_url, lesion_type)

        yield make_stream_event(
            "context",
            run_id=run_id,
            upload_url=upload_url,
            lesion_type=lesion_type,
            framework=agent.framework,
        )

        try:
            for latest_state in agent.stream(latest_state):
                update_context_from_state(context, latest_state)
                events = latest_state.get("events", [])
                for step in events[emitted_count:]:
                    yield make_stream_event("step", step=step, index=emitted_count)
                    emitted_count += 1

            update_context_from_state(context, latest_state)
            response = build_analysis_response(latest_state)
            response["framework"] = agent.framework
            yield make_stream_event("done", **response)
        except Exception as exc:
            app.logger.exception("Unhandled agent streaming analysis error: %s", exc)
            step = make_step("error", "分析服务异常", SERVICE_UNAVAILABLE_MESSAGE)
            context.setdefault("steps", []).append(step)
            yield make_stream_event("step", step=step, index=emitted_count)
            yield make_stream_event(
                "done",
                ok=False,
                run_id=run_id,
                upload_url=upload_url,
                error=SERVICE_UNAVAILABLE_MESSAGE,
                steps=context.get("steps", []),
                framework=agent.framework,
            )

    return Response(
        agent_event_stream(),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        return Response(
            make_stream_event("done", ok=False, error="请先选择一张图片。", steps=[]),
            status=400,
            mimetype="application/x-ndjson",
        )

    if not allowed_file(uploaded_file.filename):
        return Response(
            make_stream_event("done", ok=False, error="仅支持 png、jpg、jpeg、bmp、webp 格式的图像。", steps=[]),
            status=400,
            mimetype="application/x-ndjson",
        )

    try:
        run_id, upload_path = save_upload(uploaded_file)
        upload_url = url_for("upload_file", run_id=run_id, filename=upload_path.name)
    except Exception as exc:
        app.logger.exception("Failed to save uploaded image before streaming: %s", exc)
        return Response(
            make_stream_event("done", ok=False, error=SERVICE_UNAVAILABLE_MESSAGE, steps=[]),
            status=500,
            mimetype="application/x-ndjson",
        )

    @stream_with_context
    def event_stream():
        log_task("收到新的智能体流式分析请求")
        steps: list[dict] = []
        context: dict | None = None

        def emit_step(step_type: str, title: str, body: str, *, meta: dict | None = None) -> str:
            step = make_step(step_type, title, body, meta=meta)
            steps.append(step)
            if context is not None:
                context["steps"] = steps
            return make_stream_event("step", step=step, index=len(steps) - 1)

        try:
            yield emit_step("upload", "收到图片", f"我已接收并保存用户上传的图片：{upload_path.name}。")

            context = {
                "run_id": run_id,
                "upload_path": str(upload_path),
                "upload_url": upload_url,
                "enhancement_result": None,
                "lesion_result": None,
                "result_image_path": None,
                "result_image_url": None,
                "classification": None,
                "vision_result": None,
                "report": None,
                "reflection": None,
                "steps": steps,
                "chat_history": [],
            }
            ANALYSIS_CONTEXTS[run_id] = context
            yield make_stream_event("context", run_id=run_id, upload_url=upload_url)

            if not validate_image(upload_path):
                yield emit_step("error", "图片校验失败", "上传文件不是有效图像，我无法继续分析。")
                yield make_stream_event(
                    "done",
                    ok=False,
                    run_id=run_id,
                    upload_url=upload_url,
                    error="上传文件不是有效图像。",
                    steps=steps,
                )
                return

            yield emit_step("check", "图片校验通过", "文件格式与图像完整性检查通过，开始进行内镜图像预检。")

            try:
                classification = classify_endoscopy(upload_path)
            except DashScopeError as exc:
                log_debug(f"内镜预检不可用，降级继续本地分析：{exc.technical_message}")
                classification = build_local_classification_fallback(exc.user_message)
                yield emit_step("warning", "AI 预检降级", classification["reason"])
            except Exception as exc:
                log_debug(f"内镜预检发生未知异常，降级继续本地分析：{exc}")
                classification = build_local_classification_fallback(str(exc))
                yield emit_step("warning", "AI 预检降级", classification["reason"])

            context["classification"] = classification
            if classification["is_endoscopy"]:
                yield emit_step(
                    "classification",
                    "AI 预检判断",
                    f"我判断这是一张内镜医学影像。依据：{classification.get('reason') or '图像特征符合内镜视野'}。",
                )
            else:
                yield emit_step(
                    "classification",
                    "AI 预检判断",
                    f"我判断这不是内镜医学影像，因此停止本地病灶定位流程。依据：{classification.get('reason') or '缺少典型内镜图像特征'}。",
                )
                yield make_stream_event(
                    "done",
                    ok=True,
                    run_id=run_id,
                    upload_url=upload_url,
                    classification=classification,
                    steps=steps,
                    message="这张图片未通过内镜图像预检，未执行 IRetinex 处理。",
                )
                return

            output_dir = upload_path.parent / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                log_task("调用 vision_tool.process_image")
                log_debug(f"vision_tool 输入图片：{upload_path}")
                log_debug(f"vision_tool 输出目录：{output_dir}")
                from vision_tool import process_image

                vision_result = process_image(str(upload_path), output_dir=str(output_dir))
                log_json("vision_tool 返回结果", vision_result)
            except Exception as exc:
                log_debug(f"vision_tool 执行失败：{exc}")
                app.logger.exception("Local vision processing failed: %s", exc)
                yield emit_step("error", "本地模型执行失败", "IRetinex 增强或病灶定位流程执行失败，请稍后重试。")
                yield make_stream_event(
                    "done",
                    ok=False,
                    run_id=run_id,
                    upload_url=upload_url,
                    error="本地图像处理失败，请稍后重试。",
                    steps=steps,
                )
                return

            context["vision_result"] = vision_result
            if vision_result.get("status") != "success" or not vision_result.get("result_image_path"):
                yield emit_step(
                    "error",
                    "本地模型返回失败",
                    f"本地模型没有生成可展示的结果图。返回信息：{vision_result.get('message', '未知错误')}",
                )
                yield make_stream_event(
                    "done",
                    ok=False,
                    run_id=run_id,
                    upload_url=upload_url,
                    error="本地图像处理失败，请稍后重试。",
                    steps=steps,
                    vision_result=vision_result,
                )
                return

            result_image_path = Path(vision_result["result_image_path"]).resolve()
            if result_image_path.is_dir() or not result_image_path.is_file():
                yield emit_step("error", "结果图不可用", f"本地模型返回的结果图路径不可读取：{result_image_path}")
                yield make_stream_event(
                    "done",
                    ok=False,
                    run_id=run_id,
                    upload_url=upload_url,
                    error="本地处理结果图片不可读取。",
                    steps=steps,
                    vision_result=vision_result,
                )
                return

            upload_dir_path = upload_path.parent.resolve()
            try:
                result_filename = result_image_path.relative_to(upload_dir_path).as_posix()
                result_image_url = url_for("upload_file", run_id=run_id, filename=result_filename)
            except ValueError:
                result_image_url = url_for("output_file", run_id=run_id, filename=result_image_path.name)
            context["result_image_path"] = str(result_image_path)
            context["result_image_url"] = result_image_url

            if vision_result.get("polyp_detected"):
                yield emit_step(
                    "vision",
                    "IRetinex 与病灶定位完成",
                    f"我已生成病灶位置标注图，并标记到疑似病灶区域。检测框坐标：{vision_result.get('bbox')}。",
                )
            else:
                yield emit_step(
                    "vision",
                    "IRetinex 与病灶定位完成",
                    "我已生成增强后的结果图。本地模型未检测到明显病灶区域。",
                )

            try:
                report_json = generate_diagnostic_report(result_image_path, vision_result)
            except DashScopeError as exc:
                log_debug(f"辅助诊断报告不可用，使用本地保底报告：{exc.technical_message}")
                report_json = build_local_diagnostic_report(vision_result, exc.user_message)
                yield emit_step("warning", "报告生成降级", "外部 AI 报告生成暂时不可用，已根据本地模型结果生成保底报告。")
            except Exception as exc:
                log_debug(f"辅助诊断报告发生未知异常，使用本地保底报告：{exc}")
                report_json = build_local_diagnostic_report(vision_result, str(exc))
                yield emit_step("warning", "报告生成降级", "外部 AI 报告生成异常，已根据本地模型结果生成保底报告。")

            report = format_structured_report(report_json)
            context["report_json"] = report_json
            context["report"] = report
            yield emit_step("report", "辅助报告生成完成", report, meta={"report_json": report_json})

            try:
                reflection = generate_result_reflection(result_image_path, vision_result, report)
            except DashScopeError as exc:
                log_debug(f"结果自检不可用，使用本地保底自检：{exc.technical_message}")
                reflection = build_local_reflection(vision_result, exc.user_message)
            except Exception as exc:
                log_debug(f"结果反思发生未知异常：{exc}")
                reflection = build_local_reflection(vision_result, str(exc))

            context["reflection"] = reflection
            confidence_text = "未评估" if reflection.get("confidence") is None else f"{reflection['confidence']}%"
            uncertainty_text = "；".join(reflection.get("uncertainties") or []) or "暂无额外列出"
            yield emit_step(
                "reflection",
                "结果自检完成",
                (
                    f"粗略置信度：{confidence_text}（{reflection.get('confidence_label') or '未标注'}）。"
                    f"{reflection.get('reflection') or ''}\n不确定因素：{uncertainty_text}\n{reflection.get('suggestion') or ''}"
                ).strip(),
            )
            context["steps"] = steps
            log_task("智能体流式分析流程完成")
            yield make_stream_event(
                "done",
                ok=True,
                run_id=run_id,
                upload_url=upload_url,
                result_image_url=result_image_url,
                classification=classification,
                vision_result=vision_result,
                report=report,
                report_json=context.get("report_json"),
                reflection=reflection,
                steps=steps,
            )
        except Exception as exc:
            app.logger.exception("Unhandled streaming analysis error: %s", exc)
            if context is not None:
                yield emit_step("error", "分析服务异常", SERVICE_UNAVAILABLE_MESSAGE)
            yield make_stream_event(
                "done",
                ok=False,
                run_id=run_id,
                upload_url=upload_url,
                error=SERVICE_UNAVAILABLE_MESSAGE,
                steps=steps,
            )

    return Response(
        event_stream(),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    log_task("收到新的 LangGraph 智能体分析请求")
    uploaded_files = get_uploaded_images()
    lesion_type = normalize_requested_lesion_type(request.form.get("lesion_type"))
    if not uploaded_files:
        return jsonify({"ok": False, "error": "请先选择至少一张图片。", "steps": []}), 400

    if any(not allowed_file(file.filename) for file in uploaded_files):
        return jsonify({"ok": False, "error": "仅支持 png、jpg、jpeg、bmp、webp 格式的图像。", "steps": []}), 400

    if len(uploaded_files) > 1:
        run_id, saved_items = save_uploads(uploaded_files)
        context = {
            "run_id": run_id,
            "upload_path": str(saved_items[0]["upload_path"]) if saved_items else None,
            "upload_url": saved_items[0]["upload_url"] if saved_items else None,
            "upload_paths": [str(item["upload_path"]) for item in saved_items],
            "upload_urls": [item["upload_url"] for item in saved_items],
            "lesion_type": lesion_type,
            "processed_images": [],
            "result_image_path": None,
            "result_image_url": None,
            "report": None,
            "reflection": None,
            "steps": [],
            "chat_history": [],
        }
        ANALYSIS_CONTEXTS[run_id] = context

        agent = create_analysis_agent()
        state = agent.invoke(
            create_initial_state(
                run_id,
                saved_items[0]["upload_path"],
                saved_items[0]["upload_url"],
                lesion_type,
                upload_items=saved_items,
            )
        )
        update_context_from_state(context, state)
        response = build_analysis_response(state)
        response["framework"] = agent.framework
        if not response.get("ok", True):
            return jsonify(response), 500
        return jsonify(response)

        all_steps: list[dict] = []
        processed_images: list[dict] = []
        for item in saved_items:
            start_step = make_step(
                "batch",
                f"开始处理图片 {item['index']}/{len(saved_items)}",
                f"正在单独处理：{item['filename']}。YOLO/MedSAM 与 IRetinex 会分别在这张图上运行。",
            )
            all_steps.append(start_step)
            image_state = process_uploaded_image_for_batch(run_id, item, lesion_type)
            for step in image_state.get("events", []):
                step = dict(step)
                step["title"] = f"图片 {item['index']} - {step.get('title', '分析步骤')}"
                all_steps.append(step)
            processed_item = {
                "index": item["index"],
                "filename": item["filename"],
                "upload_path": str(item["upload_path"]),
                "upload_url": item["upload_url"],
                "result_image_path": image_state.get("result_image_path"),
                "result_image_url": image_state.get("result_image_url"),
                "ok": image_state.get("ok", True),
                "error": image_state.get("error"),
                "state": {
                    "ok": image_state.get("ok", True),
                    "error": image_state.get("error"),
                    "lesion_type": lesion_type,
                    "upload_url": item["upload_url"],
                    "enhancement_result": image_state.get("enhancement_result"),
                    "lesion_result": image_state.get("lesion_result"),
                    "vision_result": image_state.get("vision_result"),
                    "result_image_url": image_state.get("result_image_url"),
                },
            }
            processed_images.append(processed_item)

        ok_images = [item for item in processed_images if item.get("ok", True) and item.get("result_image_path")]
        if not ok_images:
            context["processed_images"] = processed_images
            context["steps"] = all_steps
            return jsonify(
                {
                    "ok": False,
                    "run_id": run_id,
                    "upload_urls": context["upload_urls"],
                    "error": "所有图片都未能完成本地处理。",
                    "processed_images": processed_images,
                    "steps": all_steps,
                    "framework": "langgraph",
                }
            ), 500

        try:
            combined_report_json = generate_multi_image_diagnostic_report(ok_images, lesion_type)
        except Exception as exc:
            user_message = getattr(exc, "user_message", str(exc))
            all_steps.append(make_step("warning", "多图综合报告降级", f"外部 AI 多图综合分析暂时不可用，已生成本地汇总。原因：{user_message}"))
            combined_report_json = build_local_multi_image_report(ok_images, user_message, lesion_type=lesion_type)
        combined_report = format_structured_report(combined_report_json)
        all_steps.append(make_step("report", "多图综合辅助报告生成完成", combined_report, meta={"report_json": combined_report_json}))

        last_result = next((item for item in reversed(ok_images) if item.get("result_image_url")), None)
        context.update(
            {
                "processed_images": processed_images,
                "result_image_path": last_result.get("result_image_path") if last_result else None,
                "result_image_url": last_result.get("result_image_url") if last_result else None,
                "report_json": combined_report_json,
                "report": combined_report,
                "steps": all_steps,
            }
        )
        return jsonify(
            {
                "ok": True,
                "run_id": run_id,
                "upload_urls": context["upload_urls"],
                "result_image_url": context.get("result_image_url"),
                "processed_images": processed_images,
                "report": combined_report,
                "report_json": combined_report_json,
                "steps": all_steps,
                "framework": "langgraph",
            }
        )

    uploaded_file = uploaded_files[0]

    run_id, upload_path = save_upload(uploaded_file)
    upload_url = url_for("upload_file", run_id=run_id, filename=upload_path.name)
    context = {
        "run_id": run_id,
        "upload_path": str(upload_path),
        "upload_url": upload_url,
        "lesion_type": lesion_type,
        "enhancement_result": None,
        "lesion_result": None,
        "result_image_path": None,
        "result_image_url": None,
        "classification": None,
        "vision_result": None,
        "lesion_analysis": None,
        "knowledge_context": None,
        "report": None,
        "reflection": None,
        "steps": [],
        "chat_history": [],
    }
    ANALYSIS_CONTEXTS[run_id] = context

    agent = create_analysis_agent()
    state = agent.invoke(create_initial_state(run_id, upload_path, upload_url, lesion_type))
    update_context_from_state(context, state)
    response = build_analysis_response(state)
    response["framework"] = agent.framework

    if not response.get("ok", True):
        status = 400 if state.get("stop_reason") == "invalid_image" else 500
        return jsonify(response), status
    return jsonify(response)
    log_task("收到新的智能体分析请求")
    steps: list[dict] = []
    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"ok": False, "error": "请先选择一张图片。", "steps": steps}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"ok": False, "error": "仅支持 png、jpg、jpeg、bmp、webp 格式的图像。", "steps": steps}), 400

    run_id, upload_path = save_upload(uploaded_file)
    upload_url = url_for("upload_file", run_id=run_id, filename=upload_path.name)
    steps.append(make_step("upload", "收到图片", f"我已接收并保存用户上传的图片：{upload_path.name}。"))

    context = {
        "run_id": run_id,
        "upload_path": str(upload_path),
        "upload_url": upload_url,
        "enhancement_result": None,
        "lesion_result": None,
        "result_image_path": None,
        "result_image_url": None,
        "classification": None,
        "vision_result": None,
        "report": None,
        "reflection": None,
        "steps": steps,
        "chat_history": [],
    }
    ANALYSIS_CONTEXTS[run_id] = context

    if not validate_image(upload_path):
        steps.append(make_step("error", "图片校验失败", "上传文件不是有效图像，我无法继续分析。"))
        return jsonify({"ok": False, "run_id": run_id, "upload_url": upload_url, "error": "上传文件不是有效图像。", "steps": steps}), 400

    steps.append(make_step("check", "图片校验通过", "文件格式与图像完整性检查通过，开始进行内镜图像预检。"))

    try:
        classification = classify_endoscopy(upload_path)
    except DashScopeError as exc:
        log_debug(f"内镜预检不可用，降级继续本地分析：{exc.technical_message}")
        classification = build_local_classification_fallback(exc.user_message)
        steps.append(make_step("warning", "AI 预检降级", classification["reason"]))
    except Exception as exc:
        log_debug(f"内镜预检发生未知异常，降级继续本地分析：{exc}")
        classification = build_local_classification_fallback(str(exc))
        steps.append(make_step("warning", "AI 预检降级", classification["reason"]))

    context["classification"] = classification
    if classification["is_endoscopy"]:
        steps.append(make_step("classification", "AI 预检判断", f"我判断这是一张内镜医学影像。依据：{classification.get('reason') or '图像特征符合内镜视野'}。"))
    else:
        steps.append(make_step("classification", "AI 预检判断", f"我判断这不是内镜医学影像，因此停止本地病灶定位流程。依据：{classification.get('reason') or '缺少典型内镜图像特征'}。"))
        context["steps"] = steps
        return jsonify(
            {
                "ok": True,
                "run_id": run_id,
                "upload_url": upload_url,
                "classification": classification,
                "steps": steps,
                "message": "这张图片未通过内镜图像预检，未执行 IRetinex 处理。",
            }
        )

    output_dir = upload_path.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        log_task("调用 vision_tool.process_image")
        log_debug(f"vision_tool 输入图片：{upload_path}")
        log_debug(f"vision_tool 输出目录：{output_dir}")
        from vision_tool import process_image

        vision_result = process_image(str(upload_path), output_dir=str(output_dir))
        log_json("vision_tool 返回结果", vision_result)
    except Exception as exc:
        log_debug(f"vision_tool 执行失败：{exc}")
        app.logger.exception("Local vision processing failed: %s", exc)
        steps.append(make_step("error", "本地模型执行失败", "IRetinex 增强或病灶定位流程执行失败，请稍后重试。"))
        return jsonify({"ok": False, "run_id": run_id, "upload_url": upload_url, "error": "本地图像处理失败，请稍后重试。", "steps": steps}), 500

    context["vision_result"] = vision_result
    if vision_result.get("status") != "success" or not vision_result.get("result_image_path"):
        steps.append(make_step("error", "本地模型返回失败", f"本地模型没有生成可展示的结果图。返回信息：{vision_result.get('message', '未知错误')}"))
        return jsonify({"ok": False, "run_id": run_id, "upload_url": upload_url, "error": "本地图像处理失败，请稍后重试。", "steps": steps, "vision_result": vision_result}), 500

    result_image_path = Path(vision_result["result_image_path"]).resolve()
    if result_image_path.is_dir() or not result_image_path.is_file():
        steps.append(make_step("error", "结果图不可用", f"本地模型返回的结果图路径不可读取：{result_image_path}"))
        return jsonify({"ok": False, "run_id": run_id, "upload_url": upload_url, "error": "本地处理结果图片不可读取。", "steps": steps, "vision_result": vision_result}), 500

    upload_dir_path = upload_path.parent.resolve()
    try:
        result_filename = result_image_path.relative_to(upload_dir_path).as_posix()
        result_image_url = url_for("upload_file", run_id=run_id, filename=result_filename)
    except ValueError:
        result_image_url = url_for("output_file", run_id=run_id, filename=result_image_path.name)
    context["result_image_path"] = str(result_image_path)
    context["result_image_url"] = result_image_url
    if vision_result.get("polyp_detected"):
        steps.append(make_step("vision", "IRetinex 与病灶定位完成", f"我已生成病灶位置标注图，并标记到疑似病灶区域。检测框坐标：{vision_result.get('bbox')}。"))
    else:
        steps.append(make_step("vision", "IRetinex 与病灶定位完成", "我已生成增强后的结果图。本地模型未检测到明显病灶区域。"))

    try:
        report_json = generate_diagnostic_report(result_image_path, vision_result)
    except DashScopeError as exc:
        log_debug(f"辅助诊断报告不可用，使用本地保底报告：{exc.technical_message}")
        report_json = build_local_diagnostic_report(vision_result, exc.user_message)
        steps.append(make_step("warning", "报告生成降级", "外部 AI 报告生成暂时不可用，已根据本地模型结果生成保底报告。"))
    except Exception as exc:
        log_debug(f"辅助诊断报告发生未知异常，使用本地保底报告：{exc}")
        report_json = build_local_diagnostic_report(vision_result, str(exc))
        steps.append(make_step("warning", "报告生成降级", "外部 AI 报告生成异常，已根据本地模型结果生成保底报告。"))

    report = format_structured_report(report_json)
    context["report_json"] = report_json
    context["report"] = report
    steps.append(make_step("report", "辅助报告生成完成", report, meta={"report_json": report_json}))

    try:
        reflection = generate_result_reflection(result_image_path, vision_result, report)
    except DashScopeError as exc:
        log_debug(f"结果自检不可用，使用本地保底自检：{exc.technical_message}")
        reflection = build_local_reflection(vision_result, exc.user_message)
    except Exception as exc:
        log_debug(f"结果反思发生未知异常：{exc}")
        reflection = build_local_reflection(vision_result, str(exc))

    context["reflection"] = reflection
    confidence_text = "未评估" if reflection.get("confidence") is None else f"{reflection['confidence']}%"
    uncertainty_text = "；".join(reflection.get("uncertainties") or []) or "暂无额外列出"
    steps.append(
        make_step(
            "reflection",
            "结果自检完成",
            (
                f"粗略置信度：{confidence_text}（{reflection.get('confidence_label') or '未标注'}）。"
                f"{reflection.get('reflection') or ''}\n不确定因素：{uncertainty_text}\n{reflection.get('suggestion') or ''}"
            ).strip(),
        )
    )
    context["steps"] = steps
    log_task("智能体分析流程完成")
    return jsonify(
        {
            "ok": True,
            "run_id": run_id,
            "upload_url": upload_url,
            "result_image_url": result_image_url,
            "classification": classification,
            "vision_result": vision_result,
            "report": report,
            "report_json": report_json,
            "reflection": reflection,
            "steps": steps,
        }
    )


@app.route("/reports/<run_id>/export", methods=["GET"])
def export_report(run_id: str):
    if not is_valid_run_id(run_id):
        abort(404)

    context = ANALYSIS_CONTEXTS.get(run_id)
    if context is None:
        return jsonify({"ok": False, "error": "未找到可导出的分析上下文，请先完成一次分析。"}), 404
    if not context.get("report") and not context.get("report_json"):
        return jsonify({"ok": False, "error": "本轮分析尚未生成报告，暂不能导出。"}), 400

    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"ai_endoscopy_report_{run_id[:8]}.pdf"
    output_pdf = output_dir / filename

    try:
        from report_export import generate_report_pdf_from_context

        generated_pdf = generate_report_pdf_from_context(context, output_pdf)
        context["report_pdf_path"] = str(generated_pdf)
        context["report_pdf_url"] = url_for("export_report", run_id=run_id)
    except Exception as exc:
        app.logger.exception("Failed to export report PDF for run_id=%s: %s", run_id, exc)
        return jsonify({"ok": False, "error": "报告导出失败，请稍后重试。"}), 500

    download_name = f"智影医镜_AI辅助报告_{run_id[:8]}.pdf"
    return send_from_directory(
        output_dir,
        filename,
        as_attachment=True,
        download_name=download_name,
        mimetype="application/pdf",
        max_age=0,
    )


@app.route("/manual-video-frame", methods=["POST"])
def save_manual_video_frame():
    payload = request.get_json(silent=True) or {}
    run_id = str(payload.get("run_id", "")).strip()
    image_data = str(payload.get("image", "")).strip()
    video_time = payload.get("video_time", 0)

    if not run_id or run_id not in ANALYSIS_CONTEXTS:
        return jsonify({"ok": False, "error": "未找到视频分析会话，请等待视频上传建立上下文后再截图。"}), 404
    context = ANALYSIS_CONTEXTS[run_id]
    if context.get("source_type") != "video":
        return jsonify({"ok": False, "error": "当前会话不是视频分析，不能保存手动截图。"}), 400
    if context.get("report"):
        return jsonify({"ok": False, "error": "本轮报告已经生成，新的截图不能再加入本轮分析。"}), 409
    if not image_data:
        return jsonify({"ok": False, "error": "没有接收到截图数据。"}), 400

    match = re.match(r"^data:image/(png|jpeg|jpg);base64,(.+)$", image_data)
    if not match:
        return jsonify({"ok": False, "error": "截图格式不正确。"}), 400

    try:
        image_bytes = base64.b64decode(match.group(2))
    except Exception:
        return jsonify({"ok": False, "error": "截图 base64 解码失败。"}), 400

    try:
        video_time_float = max(0.0, float(video_time))
    except (TypeError, ValueError):
        video_time_float = 0.0

    upload_dir = UPLOAD_ROOT / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    counter = int(context.get("manual_frame_counter") or 0) + 1
    context["manual_frame_counter"] = counter
    filename = f"manual_frame_{counter:02d}_t{video_time_float:.2f}s.png"
    save_path = upload_dir / filename
    crop_result = save_cropped_video_frame_bytes(image_bytes, save_path)
    if crop_result.get("cropped"):
        log_debug(f"手动视频截图已裁剪黑边：file={filename}, crop={crop_result.get('crop_box')}")

    item = {
        "id": uuid4().hex,
        "filename": filename,
        "upload_path": save_path,
        "upload_url": url_for("upload_file", run_id=run_id, filename=filename),
        "source_type": "video_manual",
        "original_video_path": context.get("original_video_path"),
        "original_video_url": context.get("original_video_url"),
        "manual_video_time": round(video_time_float, 2),
        "crop_result": crop_result,
    }
    context.setdefault("manual_frames", []).append(item)

    return jsonify(
        {
            "ok": True,
            "id": item["id"],
            "filename": filename,
            "image_url": item["upload_url"],
            "video_time": item["manual_video_time"],
            "crop_result": crop_result,
            "manual_frame_count": len(context.get("manual_frames") or []),
        }
    )


@app.route("/manual-video-frame/delete", methods=["POST"])
def delete_manual_video_frame():
    payload = request.get_json(silent=True) or {}
    run_id = str(payload.get("run_id", "")).strip()
    frame_id = str(payload.get("id", "")).strip()
    filename = str(payload.get("filename", "")).strip()

    if not run_id or run_id not in ANALYSIS_CONTEXTS:
        return jsonify({"ok": False, "error": "未找到视频分析会话。"}), 404
    context = ANALYSIS_CONTEXTS[run_id]
    if context.get("source_type") != "video":
        return jsonify({"ok": False, "error": "当前会话不是视频分析。"}), 400
    if context.get("manual_frames_closed") or context.get("report"):
        return jsonify({"ok": False, "error": "手动选帧已经提交，不能再删除本轮截图。"}), 409

    manual_frames = context.get("manual_frames") or []
    target = None
    remaining = []
    for item in manual_frames:
        is_match = (frame_id and item.get("id") == frame_id) or (filename and item.get("filename") == filename)
        if is_match and target is None:
            target = item
            continue
        remaining.append(item)

    if target is None:
        return jsonify({"ok": False, "error": "未找到要删除的手动截图。"}), 404

    context["manual_frames"] = remaining
    upload_path = target.get("upload_path")
    if upload_path:
        try:
            path = Path(upload_path).resolve()
            upload_root = (UPLOAD_ROOT / run_id).resolve()
            if path.is_file() and str(path).startswith(str(upload_root)):
                path.unlink()
        except Exception as exc:
            log_debug(f"删除手动截图文件失败，将仅从队列移除：{exc}")

    return jsonify({"ok": True, "manual_frame_count": len(remaining)})


@app.route("/manual-video-frames/finish", methods=["POST"])
def finish_manual_video_frames():
    payload = request.get_json(silent=True) or {}
    run_id = str(payload.get("run_id", "")).strip()
    if not run_id or run_id not in ANALYSIS_CONTEXTS:
        return jsonify({"ok": False, "error": "未找到视频分析会话。"}), 404

    context = ANALYSIS_CONTEXTS[run_id]
    if context.get("source_type") != "video":
        return jsonify({"ok": False, "error": "当前会话不是视频分析。"}), 400
    context["manual_frames_closed"] = True

    return jsonify({"ok": True, "manual_frame_count": len(context.get("manual_frames") or [])})


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    run_id = str(payload.get("run_id", "")).strip()
    user_text = str(payload.get("message", "")).strip()
    if not run_id or run_id not in ANALYSIS_CONTEXTS:
        return jsonify({"ok": False, "error": "未找到可追问的分析上下文，请先上传并完成一次分析。"}), 404
    if not user_text:
        return jsonify({"ok": False, "error": "请输入要追问的内容。"}), 400

    context = ANALYSIS_CONTEXTS[run_id]
    result = create_followup_agent().invoke(context, user_text)
    context.setdefault("chat_history", []).extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": result.answer},
        ]
    )
    return jsonify({"ok": True, "answer": result.answer, "fallback": result.fallback})

    payload = request.get_json(silent=True) or {}
    run_id = str(payload.get("run_id", "")).strip()
    user_text = str(payload.get("message", "")).strip()
    if not run_id or run_id not in ANALYSIS_CONTEXTS:
        return jsonify({"ok": False, "error": "未找到可追问的分析上下文，请先上传并完成一次分析。"}), 404
    if not user_text:
        return jsonify({"ok": False, "error": "请输入要追问的内容。"}), 400

    context = ANALYSIS_CONTEXTS[run_id]
    try:
        answer = dashscope_chat(
            build_followup_messages(context, user_text),
            max_tokens=900,
            task_name="用户多轮追问",
        )
    except DashScopeError as exc:
        log_debug(f"多轮追问 AI 不可用，使用本地上下文回答：{exc.technical_message}")
        answer = build_local_followup_answer(context, user_text, exc.user_message)
    except Exception as exc:
        log_debug(f"多轮追问发生未知异常，使用本地上下文回答：{exc}")
        answer = build_local_followup_answer(context, user_text, str(exc))

    context.setdefault("chat_history", []).extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]
    )
    return jsonify({"ok": True, "answer": answer})


@app.route("/uploads/<run_id>/<path:filename>", methods=["GET"])
def upload_file(run_id: str, filename: str):
    if not is_valid_run_id(run_id):
        abort(404)
    return send_from_directory(UPLOAD_ROOT / run_id, filename)


@app.route("/outputs/<run_id>/<path:filename>", methods=["GET"])
def output_file(run_id: str, filename: str):
    if not is_valid_run_id(run_id):
        abort(404)
    return send_from_directory(OUTPUT_ROOT / run_id, filename)


if __name__ == "__main__":
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5001, debug=True)