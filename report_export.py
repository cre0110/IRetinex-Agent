from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


FONT_NAME = "CNFont"

RISK_NOTICE = [
    "本报告由 AI 系统基于输入内镜图像进行辅助分析，仅供医生参考，不能替代内镜医生判断及病理结果。",
    "本系统不承担病理诊断功能，最终诊断及分级应以临床医生复核及病理学结果为准。",
]

FOLLOWUP_ADVICE = [
    "请由专业内镜医生结合完整内镜过程、病史、体征及既往检查资料综合判断。",
    "如发现可疑区域，建议结合临床需要进行靶向观察、复查或病理活检。",
    "若图像质量、暴露范围或病灶边界存在不确定性，应以人工复核和必要补充检查为准。",
]

LESION_DISPLAY_NAMES = {
    "polyp": "结肠息肉",
    "colon": "结肠息肉",
    "colon_polyp": "结肠息肉",
    "colorectal_polyp": "结肠息肉",
    "barrett": "Barrett 食管（BE）",
    "be": "Barrett 食管（BE）",
    "upper_gi": "Barrett 食管（BE）",
    "upper_gi_be": "Barrett 食管（BE）",
    "barrett_esophagus": "Barrett 食管（BE）",
}


def register_chinese_font(font_path: str | None = None) -> str:
    global FONT_NAME

    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(
        [
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\Deng.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]
    )

    for path in candidates:
        if Path(path).exists():
            try:
                FONT_NAME = "CNFont"
                pdfmetrics.registerFont(TTFont(FONT_NAME, path))
                return FONT_NAME
            except Exception:
                continue

    FONT_NAME = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))
    return FONT_NAME


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _existing_path(value: Any) -> str:
    text = _safe_text(value)
    if text and Path(text).is_file():
        return str(Path(text).resolve())
    return text


def _nested(source: dict, *keys: str) -> Any:
    current: Any = source
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _normalize_lesion_type(value: Any) -> str:
    raw = _safe_text(value)
    key = raw.lower().replace(" ", "_").replace("-", "_")
    if key in LESION_DISPLAY_NAMES:
        return LESION_DISPLAY_NAMES[key]
    if "barrett" in key or key == "be" or "上消化" in raw:
        return LESION_DISPLAY_NAMES["barrett"]
    if "polyp" in key or "息肉" in raw or "结肠" in raw:
        return LESION_DISPLAY_NAMES["polyp"]
    return raw or "内镜病灶"


def _pick_enhancement_result(state: dict) -> dict:
    vision_result = state.get("vision_result") or {}
    return state.get("enhancement_result") or vision_result.get("enhancement_result") or {}


def _pick_lesion_result(state: dict) -> dict:
    vision_result = state.get("vision_result") or {}
    return state.get("lesion_result") or vision_result.get("lesion_result") or {}


def _build_sample_from_processed_item(item: dict, index: int) -> dict:
    state = item.get("state") or {}
    vision_result = state.get("vision_result") or {}
    enhancement_result = _pick_enhancement_result(state)
    lesion_result = _pick_lesion_result(state)

    raw_image = (
        item.get("upload_path")
        or enhancement_result.get("original_path")
        or lesion_result.get("original_image_path")
    )
    enhanced_image = (
        enhancement_result.get("fusion_path")
        or enhancement_result.get("enhanced_image")
        or item.get("result_image_path")
        or vision_result.get("result_image_path")
        or raw_image
    )
    mask_image = (
        item.get("result_image_path")
        or vision_result.get("result_image_path")
        or lesion_result.get("lesion_image_path")
        or lesion_result.get("mask_path")
        or enhanced_image
    )

    filename = _safe_text(item.get("filename")) or f"image_{index}"
    return {
        "label": f"图像 {item.get('index') or index}",
        "filename": filename,
        "raw_image": _existing_path(raw_image),
        "enhanced_image": _existing_path(enhanced_image),
        "mask_image": _existing_path(mask_image),
        "vision_result": vision_result,
        "lesion_analysis": state.get("lesion_analysis"),
    }


def _build_single_sample(context: dict) -> dict:
    vision_result = context.get("vision_result") or {}
    enhancement_result = _pick_enhancement_result(context)
    lesion_result = _pick_lesion_result(context)
    raw_image = (
        context.get("upload_path")
        or enhancement_result.get("original_path")
        or lesion_result.get("original_image_path")
    )
    enhanced_image = (
        enhancement_result.get("fusion_path")
        or enhancement_result.get("enhanced_image")
        or context.get("result_image_path")
        or vision_result.get("result_image_path")
        or raw_image
    )
    mask_image = (
        context.get("result_image_path")
        or vision_result.get("result_image_path")
        or lesion_result.get("lesion_image_path")
        or lesion_result.get("mask_path")
        or enhanced_image
    )
    filename = Path(_safe_text(raw_image)).name if raw_image else "uploaded_image"
    return {
        "label": "图像 1",
        "filename": filename,
        "raw_image": _existing_path(raw_image),
        "enhanced_image": _existing_path(enhanced_image),
        "mask_image": _existing_path(mask_image),
        "vision_result": vision_result,
        "lesion_analysis": context.get("lesion_analysis"),
    }


def _build_samples(context: dict) -> list[dict]:
    processed_images = [
        item
        for item in (context.get("processed_images") or [])
        if item.get("ok", True) and (item.get("result_image_path") or _nested(item, "state", "vision_result", "result_image_path"))
    ]
    if processed_images:
        return [_build_sample_from_processed_item(item, index) for index, item in enumerate(processed_images, start=1)]
    if context.get("upload_path") or context.get("result_image_path"):
        return [_build_single_sample(context)]
    return []


def _build_diagnosis_items(samples: list[dict], lesion_type: str) -> list[str]:
    items: list[str] = []
    for sample in samples:
        vision_result = sample.get("vision_result") or {}
        detected = vision_result.get("polyp_detected")
        bbox = vision_result.get("bbox") or vision_result.get("annotation_bbox")
        label = sample.get("label") or "图像"
        if detected is True:
            bbox_text = f"，检测框坐标 {bbox}" if bbox else ""
            items.append(f"{label}：提示存在疑似 {lesion_type} 相关异常区域{bbox_text}，需结合医生复核？")
        elif detected is False:
            items.append(f"{label}：本地模型未检测到明显疑似病灶区域，仍需结合完整内镜过程判断？")

    if items:
        return items
    return [f"本轮图像提示需结合 {lesion_type} 专病场景进行人工复核？"]


def _build_reflection_text(reflection: dict | None) -> str:
    if not reflection:
        return ""
    confidence = reflection.get("confidence")
    confidence_text = "未评估" if confidence is None else f"{confidence}%"
    label = _safe_text(reflection.get("confidence_label")) or "未标注"
    uncertainties = reflection.get("uncertainties") or []
    if isinstance(uncertainties, str):
        uncertainties = [uncertainties]
    uncertainty_text = "；".join(_safe_text(item) for item in uncertainties if _safe_text(item)) or "暂无额外列出"
    parts = [
        f"粗略置信度：{confidence_text}（{label}）",
        _safe_text(reflection.get("reflection")),
        f"不确定因素：{uncertainty_text}",
        _safe_text(reflection.get("suggestion")),
    ]
    return "\n".join(part for part in parts if part)


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_report_json(value: Any) -> dict | None:
    if not value:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None

    findings = _safe_text(value.get("endoscopic_findings") or value.get("内镜所见"))
    diagnosis = _as_text_list(value.get("endoscopic_diagnosis") or value.get("内镜诊断"))
    followup = _safe_text(value.get("followup_advice") or value.get("医生复诊建议"))
    if not (findings or diagnosis or followup):
        return None
    return {
        "endoscopic_findings": findings,
        "endoscopic_diagnosis": diagnosis,
        "followup_advice": followup,
    }


def build_report_data_from_context(context: dict) -> dict:
    samples = _build_samples(context)
    lesion_type = _normalize_lesion_type(context.get("lesion_type") or _nested(context, "vision_result", "lesion_type"))
    is_video = context.get("source_type") == "video"
    report_json = _normalize_report_json(context.get("report_json"))
    report_text = _safe_text(context.get("report")) or "本轮尚未生成可导出的辅助报告正文。"

    if is_video:
        source_note = "用户本次上传的是视频；本报告基于后端抽取并初筛后的关键帧生成，不能代表完整视频全过程。"
    else:
        source_note = "用户本次上传的是静态内镜图像；本报告基于本轮已处理图像生成。"

    return {
        "title": "智影医镜 AI 辅助内镜专病分析报告",
        "subtitle": "医生审核版 / 供临床参考",
        "run_id": _safe_text(context.get("run_id")),
        "task_type": lesion_type,
        "source_note": source_note,
        "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "samples": samples,
        "report_json": report_json,
        "report_text": report_text,
        "diagnosis_items": _build_diagnosis_items(samples, lesion_type),
        "followup_advice": FOLLOWUP_ADVICE,
        "reflection_text": _build_reflection_text(context.get("reflection")),
        "risk_notice": RISK_NOTICE,
    }


def _escaped_paragraph_text(text: str) -> str:
    return html.escape(_safe_text(text)).replace("\n", "<br/>")


def make_paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_escaped_paragraph_text(text), style)


def _markdown_inline(text: str) -> str:
    escaped = html.escape(_safe_text(text))
    escaped = re.sub(r"`([^`]+)`", r'<font color="#374151">\1</font>', escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"__(.+?)__", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped.replace("\n", "<br/>")


def make_markdown_paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_markdown_inline(text), style)


def make_section_title(title: str, styles) -> Paragraph:
    return Paragraph(f"<b>{html.escape(title)}</b>", styles["report_section"])


def make_section_divider():
    return HRFlowable(width="100%", thickness=0.45, color=colors.HexColor("#9CA3AF"), spaceBefore=5, spaceAfter=6)


def make_numbered_list(items: list[str], styles) -> list[Paragraph]:
    story: list[Paragraph] = []
    for index, item in enumerate(items, start=1):
        clean_item = _safe_text(item)
        if clean_item:
            story.append(make_paragraph(f"{index}. {clean_item}", styles["report_body"]))
    return story


def make_text_blocks(text: str, styles) -> list:
    blocks = []
    code_lines = []
    in_code_block = False

    def flush_code_block() -> None:
        nonlocal code_lines
        if not code_lines:
            return
        code_text = html.escape("\n".join(code_lines)).replace("\n", "<br/>")
        blocks.append(Paragraph(code_text, styles["report_code"]))
        blocks.append(Spacer(1, 2))
        code_lines = []

    for raw_line in _safe_text(text).replace("\r\n", "\n").split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code_block:
                flush_code_block()
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if not stripped:
            if blocks and not isinstance(blocks[-1], Spacer):
                blocks.append(Spacer(1, 2))
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            blocks.append(make_markdown_paragraph(heading.group(2), styles[f"report_md_h{level}"]))
            continue

        checkbox = re.match(r"^\s*(?:[-*+]\s*)?(?:\[[ xX]\]|[□☐☑✓✔])\s*(.+)$", line)
        if checkbox:
            blocks.append(Paragraph(f"-&nbsp;{_markdown_inline(checkbox.group(1))}", styles["report_bullet"]))
            continue

        unordered = re.match(r"^\s*[-*+]\s+(.+)$", line)
        if unordered:
            blocks.append(Paragraph(f"-&nbsp;{_markdown_inline(unordered.group(1))}", styles["report_bullet"]))
            continue

        ordered = re.match(r"^\s*(\d+)\.\s+(.+)$", line)
        if ordered:
            blocks.append(
                Paragraph(
                    f"{html.escape(ordered.group(1))}.&nbsp;{_markdown_inline(ordered.group(2))}",
                    styles["report_bullet"],
                )
            )
            continue

        quote = re.match(r"^>\s+(.+)$", stripped)
        if quote:
            blocks.append(make_markdown_paragraph(quote.group(1), styles["report_quote"]))
            continue

        blocks.append(make_markdown_paragraph(stripped, styles["report_body"]))

    if in_code_block:
        flush_code_block()
    return blocks


def fit_image(image_path: str, max_w: float, max_h: float):
    path = Path(_safe_text(image_path))
    if not path.is_file():
        return Paragraph("图片未提供", ParagraphStyle(name=f"missing_{abs(hash(image_path))}", fontName=FONT_NAME, fontSize=8, alignment=TA_CENTER))

    try:
        with PILImage.open(path) as img:
            width, height = img.size
        scale = min(max_w / width, max_h / height)
        image = Image(str(path), width=width * scale, height=height * scale)
        image.hAlign = "CENTER"
        return image
    except Exception:
        return Paragraph("图片读取失败", ParagraphStyle(name=f"bad_{abs(hash(image_path))}", fontName=FONT_NAME, fontSize=8, alignment=TA_CENTER))


def build_title_block(data: dict, styles):
    table = Table(
        [
            [
                Paragraph("", styles["report_small"]),
                Paragraph(f"<b>{html.escape(data.get('title', '辅助报告'))}</b>", styles["report_title"]),
                Paragraph(html.escape(data.get("subtitle", "")), styles["report_subtitle"]),
            ]
        ],
        colWidths=[28 * mm, 104 * mm, 38 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_info_table(data: dict, styles):
    samples = data.get("samples") or []
    rows = [
        [
            f"任务类型：{data.get('task_type', '')}",
            f"图像数量：{len(samples)}",
            f"导出时间：{data.get('analysis_time', '')}",
        ],
        [
            f"会话编号：{data.get('run_id', '')[:12] or '未记录'}",
            data.get("source_note", ""),
            "",
        ],
    ]
    table_data = [[Paragraph(cell, styles["report_info"]) for cell in row] for row in rows]
    table = Table(table_data, colWidths=[53 * mm, 67 * mm, 50 * mm], rowHeights=[9 * mm, 12 * mm])
    table.setStyle(
        TableStyle(
            [
                ("SPAN", (1, 1), (2, 1)),
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.35, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def build_image_table(data: dict, styles):
    samples = data.get("samples") or []
    header = [
        Paragraph("", styles["report_img_header"]),
        Paragraph("<b>原图</b>", styles["report_img_header"]),
        Paragraph("<b>增强图</b>", styles["report_img_header"]),
        Paragraph("<b>病灶标注图</b>", styles["report_img_header"]),
    ]
    table_data = [header]
    max_img_w = 45 * mm
    max_img_h = 29 * mm
    for sample in samples:
        table_data.append(
            [
                Paragraph(html.escape(sample.get("label", "图像")), styles["report_img_label"]),
                fit_image(sample.get("raw_image", ""), max_img_w, max_img_h),
                fit_image(sample.get("enhanced_image", ""), max_img_w, max_img_h),
                fit_image(sample.get("mask_image", ""), max_img_w, max_img_h),
            ]
        )

    table = Table(
        table_data,
        colWidths=[18 * mm, 50 * mm, 50 * mm, 52 * mm],
        rowHeights=[7 * mm] + [34 * mm for _ in samples],
        repeatRows=1,
        splitByRow=1,
    )
    style = [
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("LINEAFTER", (0, 1), (0, -1), 0.35, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    for row in range(1, len(table_data)):
        style.append(("LINEBELOW", (0, row), (-1, row), 0.35, colors.black))
    table.setStyle(TableStyle(style))
    return table


REPORT_BODY_SECTION_TITLES = ("内镜所见", "内镜诊断", "医生复诊建议")


def split_required_report_sections(report_text: str) -> list[tuple[str, str]]:
    text = _safe_text(report_text)
    if not text:
        return []

    pattern = re.compile(r"(内镜所见|内镜诊断|医生复诊建议)\s*[:：]")
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    sections: list[tuple[str, str]] = []
    seen_titles: set[str] = set()
    for index, match in enumerate(matches):
        title = match.group(1)
        if title in seen_titles:
            continue
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : next_start].strip()
        sections.append((title, body))
        seen_titles.add(title)

    section_map = {title: body for title, body in sections}
    return [(title, section_map.get(title, "")) for title in REPORT_BODY_SECTION_TITLES if title in section_map]


def build_text_sections(data: dict, styles):
    story = [Spacer(1, 4)]
    report_json = _normalize_report_json(data.get("report_json"))
    if report_json:
        report_sections = [
            ("内镜所见", report_json.get("endoscopic_findings", "")),
            (
                "内镜诊断",
                "\n".join(
                    f"{index}、{item}"
                    for index, item in enumerate(report_json.get("endoscopic_diagnosis") or [], start=1)
                ),
            ),
            ("医生复诊建议", report_json.get("followup_advice", "")),
        ]
    else:
        report_sections = split_required_report_sections(data.get("report_text", ""))
    if report_sections:
        for index, (title, body) in enumerate(report_sections):
            if index:
                story.append(make_section_divider())
            story.append(make_section_title(f"{title}：", styles))
            story.extend(make_text_blocks(body, styles))
    else:
        story.extend(make_text_blocks(data.get("report_text", ""), styles))

    story.extend(
        [
            make_section_divider(),
            make_section_title("AI 风险提示：", styles),
            *make_numbered_list(data.get("risk_notice", RISK_NOTICE), styles),
        ]
    )
    return story


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_NAME, 8)
    canvas.drawRightString(200 * mm, 8 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_styles():
    styles = getSampleStyleSheet()
    base = {
        "fontName": FONT_NAME,
        "textColor": colors.black,
        "wordWrap": "CJK",
    }
    styles.add(ParagraphStyle(name="report_title", parent=styles["Normal"], fontSize=17, leading=22, alignment=TA_CENTER, **base))
    styles.add(ParagraphStyle(name="report_subtitle", parent=styles["Normal"], fontSize=8.5, leading=11, alignment=TA_RIGHT, **base))
    styles.add(ParagraphStyle(name="report_info", parent=styles["Normal"], fontSize=8.5, leading=10.5, alignment=TA_LEFT, **base))
    styles.add(ParagraphStyle(name="report_img_header", parent=styles["Normal"], fontSize=9, leading=11, alignment=TA_CENTER, **base))
    styles.add(ParagraphStyle(name="report_img_label", parent=styles["Normal"], fontSize=8.5, leading=10, alignment=TA_CENTER, **base))
    styles.add(ParagraphStyle(name="report_section", parent=styles["Normal"], fontSize=11.5, leading=14, alignment=TA_LEFT, spaceAfter=3, **base))
    styles.add(ParagraphStyle(name="report_body", parent=styles["Normal"], fontSize=9, leading=12.5, alignment=TA_LEFT, spaceAfter=2, **base))
    styles.add(ParagraphStyle(name="report_md_h1", parent=styles["Normal"], fontSize=12.5, leading=15, alignment=TA_LEFT, spaceBefore=4, spaceAfter=3, **base))
    styles.add(ParagraphStyle(name="report_md_h2", parent=styles["Normal"], fontSize=11, leading=13.5, alignment=TA_LEFT, spaceBefore=3, spaceAfter=2, **base))
    styles.add(ParagraphStyle(name="report_md_h3", parent=styles["Normal"], fontSize=10, leading=12.5, alignment=TA_LEFT, spaceBefore=2, spaceAfter=2, **base))
    styles.add(ParagraphStyle(name="report_bullet", parent=styles["Normal"], fontSize=9, leading=12.5, alignment=TA_LEFT, leftIndent=6 * mm, firstLineIndent=-4 * mm, spaceAfter=1.5, **base))
    styles.add(ParagraphStyle(name="report_quote", parent=styles["Normal"], fontSize=8.8, leading=12, alignment=TA_LEFT, leftIndent=5 * mm, textColor=colors.HexColor("#374151"), spaceAfter=2, wordWrap="CJK", fontName=FONT_NAME))
    styles.add(ParagraphStyle(name="report_code", parent=styles["Normal"], fontSize=8, leading=10.5, alignment=TA_LEFT, leftIndent=5 * mm, textColor=colors.HexColor("#374151"), wordWrap="CJK", fontName=FONT_NAME))
    styles.add(ParagraphStyle(name="report_small", parent=styles["Normal"], fontSize=8, leading=10, alignment=TA_LEFT, **base))
    return styles


def generate_report_pdf(report_data: dict, output_pdf: str | Path, font_path: str | None = None) -> Path:
    register_chinese_font(font_path)
    output_path = Path(output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=10 * mm,
        bottomMargin=14 * mm,
    )
    story = [
        build_title_block(report_data, styles),
        build_info_table(report_data, styles),
        Spacer(1, 5),
        build_image_table(report_data, styles),
        *build_text_sections(report_data, styles),
    ]
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return output_path


def generate_report_pdf_from_context(context: dict, output_pdf: str | Path, font_path: str | None = None) -> Path:
    report_data = build_report_data_from_context(context)
    return generate_report_pdf(report_data, output_pdf, font_path=font_path)
