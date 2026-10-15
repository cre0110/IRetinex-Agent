import time
import cv2
import math
import numpy as np
from tqdm import tqdm
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ============================================================
# Colon / Polyp 视频关键帧筛选：纯净输出版
#
# 只保存最终筛选出的原始关键帧图片。
# 不再保存 failed、top_candidates_debug、annotated、csv 报告。
#
# Q = 图像质量分
# Z = 息肉展示价值分
# M = YOLO模型可疑度
# score = 0.30 * Q + 0.30 * Z + 0.40 * M
# ============================================================


# =========================
# 1. 路径设置
# =========================

YOLO_MODEL_PATH = r"D:\MedicalAI\new weight\YOLO_colon_stable_cv.pt"
VIDEO_PATH = r"D:\MedicalAI\视频\colon_demo.mp4"
OUTPUT_DIR = r"D:\MedicalAI\colon_keyframe_final_only"


# =========================
# 2. 参数设置
# =========================

USE_YOLO = True
SAMPLE_EVERY_N_FRAMES = 5

VIDEO_START_SECONDS = None
VIDEO_END_SECONDS = None

YOLO_CONF_THRES = 0.25

MAX_KEYFRAMES_PER_VIDEO = 5
MIN_TIME_GAP_BETWEEN_KEYFRAMES = 2.5


# =========================
# 3. 硬过滤阈值
# =========================

MIN_VALID_VIEW_RATIO = 0.25
MIN_CLARITY_SCORE = 0.10

MAX_OVEREXPOSED_RATIO = 0.30
MAX_SPECULAR_RATIO = 0.28

MAX_OBSTRUCTION_SCORE = 0.88

MIN_BBOX_AREA_RATIO = 0.002
MAX_BBOX_AREA_RATIO = 0.55


# =========================
# 4. 工具函数
# =========================

def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def masked_ratio(binary_mask, valid_mask):
    valid_pixels = int(valid_mask.sum())
    if valid_pixels == 0:
        return 0.0
    return float(np.logical_and(binary_mask, valid_mask > 0).sum() / valid_pixels)


def score_tag(Q, Z, M, score):
    return f"Q{Q:.2f}_Z{Z:.2f}_M{M:.2f}_S{score:.2f}"


def get_valid_endoscopy_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rough = (gray > 20).astype(np.uint8) * 255

    kernel = np.ones((9, 9), np.uint8)
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, kernel, iterations=2)
    rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rough, connectivity=8)

    if num_labels <= 1:
        return np.ones(gray.shape, dtype=np.uint8)

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for label in range(2, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = label

    mask = (labels == largest_label).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)

    return mask


def crop_bbox(frame, bbox, expand_ratio=0.08):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    x1 = int(max(0, x1 - bw * expand_ratio))
    y1 = int(max(0, y1 - bh * expand_ratio))
    x2 = int(min(w - 1, x2 + bw * expand_ratio))
    y2 = int(min(h - 1, y2 + bh * expand_ratio))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


# =========================
# 5. Q：图像质量分
# =========================

def compute_quality_score(frame, valid_mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    valid_bool = valid_mask > 0
    if valid_bool.sum() == 0:
        valid_bool = np.ones_like(gray, dtype=bool)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(lap[valid_bool]))
    clarity_score = float(np.clip(math.log1p(lap_var) / math.log1p(1000), 0, 1))

    overexposed_mask = (V > 245)
    specular_mask = (V > 235) & (S < 110)

    overexposed_ratio = masked_ratio(overexposed_mask, valid_mask)
    specular_ratio = masked_ratio(specular_mask, valid_mask)

    glare_penalty = max(
        np.clip(overexposed_ratio / MAX_OVEREXPOSED_RATIO, 0, 1),
        np.clip(specular_ratio / MAX_SPECULAR_RATIO, 0, 1)
    )
    glare_score = float(1.0 - glare_penalty)

    dark_mask = (V < 35)
    yellow_brown_mask = (H >= 10) & (H <= 45) & (S > 50) & (V > 40)

    dark_ratio = masked_ratio(dark_mask, valid_mask)
    yellow_brown_ratio = masked_ratio(yellow_brown_mask, valid_mask)

    obstruction_score = (
        0.45 * np.clip(dark_ratio / 0.55, 0, 1)
        + 0.55 * np.clip(yellow_brown_ratio / 0.50, 0, 1)
    )
    obstruction_score = float(np.clip(obstruction_score, 0, 1))
    clean_score = float(1.0 - obstruction_score)

    gray_valid = gray[valid_bool]
    brightness = float(np.mean(gray_valid))
    brightness_score = float(np.clip(1.0 - abs(brightness - 130.0) / 130.0, 0, 1))

    Q = (
        0.45 * clarity_score
        + 0.25 * glare_score
        + 0.20 * clean_score
        + 0.10 * brightness_score
    )
    Q = float(np.clip(Q, 0, 1))

    metrics = {
        "clarity_score": clarity_score,
        "overexposed_ratio": overexposed_ratio,
        "specular_ratio": specular_ratio,
        "obstruction_score": obstruction_score,
    }

    return Q, metrics


# =========================
# 6. M：YOLO模型可疑度
# =========================

def run_yolo_detections(model, frame):
    if model is None:
        return 0.0, None

    results = model.predict(
        source=frame,
        conf=YOLO_CONF_THRES,
        verbose=False
    )

    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return 0.0, None

    boxes = results[0].boxes
    confs = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()

    best_idx = int(np.argmax(confs))
    raw_conf = float(confs[best_idx])
    bbox = xyxy[best_idx].astype(float).tolist()

    return raw_conf, bbox


# =========================
# 7. Z：息肉展示价值分
# =========================

def compute_colon_polyp_display_score(frame, valid_mask, bbox):
    if bbox is None:
        return 0.0

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]

    bbox_w = max(0.0, x2 - x1)
    bbox_h = max(0.0, y2 - y1)
    bbox_area = bbox_w * bbox_h

    valid_area = max(float(valid_mask.sum()), 1.0)
    bbox_area_ratio = float(bbox_area / valid_area)

    if bbox_area_ratio < MIN_BBOX_AREA_RATIO:
        bbox_area_score = 0.0
    elif bbox_area_ratio <= 0.03:
        bbox_area_score = float(np.clip(bbox_area_ratio / 0.03, 0, 1))
    elif bbox_area_ratio <= 0.25:
        bbox_area_score = 1.0
    elif bbox_area_ratio <= MAX_BBOX_AREA_RATIO:
        bbox_area_score = float(1.0 - 0.60 * (bbox_area_ratio - 0.25) / (MAX_BBOX_AREA_RATIO - 0.25))
    else:
        bbox_area_score = 0.15

    bbox_area_score = float(np.clip(bbox_area_score, 0, 1))

    x1i = int(max(0, min(w - 1, round(x1))))
    y1i = int(max(0, min(h - 1, round(y1))))
    x2i = int(max(0, min(w - 1, round(x2))))
    y2i = int(max(0, min(h - 1, round(y2))))

    if x2i <= x1i or y2i <= y1i:
        bbox_valid_ratio = 0.0
    else:
        bbox_mask = np.zeros(valid_mask.shape, dtype=np.uint8)
        bbox_mask[y1i:y2i, x1i:x2i] = 1
        bbox_pixels = max(float(bbox_mask.sum()), 1.0)
        bbox_valid_ratio = float(np.logical_and(bbox_mask > 0, valid_mask > 0).sum() / bbox_pixels)

    margin_x = min(x1, w - x2) / max(w, 1)
    margin_y = min(y1, h - y2) / max(h, 1)
    min_margin = min(margin_x, margin_y)

    bbox_edge_penalty = float(np.clip((0.015 - min_margin) / 0.015, 0, 1))
    bbox_position_score = float(np.clip(0.75 * bbox_valid_ratio + 0.25 * (1.0 - bbox_edge_penalty), 0, 1))

    roi = crop_bbox(frame, bbox, expand_ratio=0.08)

    if roi is None or roi.size == 0:
        roi_quality_score = 0.0
        mucosa_color_score = 0.0
    else:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(roi_hsv)

        roi_lap = cv2.Laplacian(roi_gray, cv2.CV_64F)
        roi_lap_var = float(np.var(roi_lap))
        roi_clarity_score = float(np.clip(math.log1p(roi_lap_var) / math.log1p(1000), 0, 1))

        roi_overexposed_ratio = float(np.mean(V > 245))
        roi_specular_ratio = float(np.mean((V > 235) & (S < 110)))

        roi_glare_penalty = max(
            np.clip(roi_overexposed_ratio / 0.30, 0, 1),
            np.clip(roi_specular_ratio / 0.28, 0, 1)
        )
        roi_glare_score = float(1.0 - roi_glare_penalty)

        roi_dark_ratio = float(np.mean(V < 35))
        roi_yellow_brown_ratio = float(np.mean((H >= 10) & (H <= 45) & (S > 50) & (V > 40)))

        roi_obstruction_score = (
            0.45 * np.clip(roi_dark_ratio / 0.55, 0, 1)
            + 0.55 * np.clip(roi_yellow_brown_ratio / 0.50, 0, 1)
        )
        roi_obstruction_score = float(np.clip(roi_obstruction_score, 0, 1))
        roi_clean_score = float(1.0 - roi_obstruction_score)

        roi_quality_score = float(np.clip(
            0.45 * roi_clarity_score
            + 0.30 * roi_glare_score
            + 0.25 * roi_clean_score,
            0,
            1
        ))

        mucosa_mask = (
            (V > 45)
            & (V < 245)
            & (S > 20)
            & ((H <= 28) | (H >= 165))
        )

        residue_mask = (H >= 18) & (H <= 45) & (S > 70) & (V > 40)

        roi_mucosa_ratio = float(np.mean(mucosa_mask))
        roi_residue_ratio = float(np.mean(residue_mask))

        mucosa_color_score = float(np.clip(roi_mucosa_ratio / 0.55, 0, 1))
        mucosa_color_score *= float(1.0 - 0.50 * np.clip(roi_residue_ratio / 0.45, 0, 1))
        mucosa_color_score = float(np.clip(mucosa_color_score, 0, 1))

    Z = (
        0.30 * bbox_area_score
        + 0.20 * bbox_position_score
        + 0.30 * roi_quality_score
        + 0.20 * mucosa_color_score
    )
    Z = float(np.clip(Z, 0, 1))

    return Z


# =========================
# 8. 过滤与排序
# =========================

def hard_reject(q_metrics, valid_view_ratio, M, bbox):
    if valid_view_ratio < MIN_VALID_VIEW_RATIO:
        return True

    if q_metrics["clarity_score"] < MIN_CLARITY_SCORE:
        return True

    if (
        q_metrics["overexposed_ratio"] > MAX_OVEREXPOSED_RATIO
        or q_metrics["specular_ratio"] > MAX_SPECULAR_RATIO
    ):
        return True

    if q_metrics["obstruction_score"] > MAX_OBSTRUCTION_SCORE:
        return True

    if bbox is None or M <= 0:
        return True

    return False


def compute_final_score(Q, Z, M):
    return float(np.clip(0.30 * Q + 0.30 * Z + 0.40 * M, 0, 1))


def select_keyframes(candidates):
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    selected = []

    for rec in candidates:
        too_close = False

        for chosen in selected:
            if abs(rec["time_s"] - chosen["time_s"]) < MIN_TIME_GAP_BETWEEN_KEYFRAMES:
                too_close = True
                break

        if too_close:
            continue

        selected.append(rec)

        if len(selected) >= MAX_KEYFRAMES_PER_VIDEO:
            break

    return sorted(selected, key=lambda x: x["time_s"])


def save_selected_frame(frame, out_dir, rank, frame_id, time_s, Q, Z, M, score):
    tag = score_tag(Q, Z, M, score)

    name = (
        f"selected_rank{rank:02d}_"
        f"{tag}_"
        f"frame{frame_id:06d}_"
        f"t{time_s:.2f}s.jpg"
    )

    save_path = Path(out_dir) / name
    cv2.imwrite(str(save_path), frame)
    return save_path


# =========================
# 9. 主流程
# =========================

def process_video(video_path, model, output_root):
    start_time = time.perf_counter()

    video_path = Path(video_path)
    video_out_dir = Path(output_root) / video_path.stem
    make_dir(video_out_dir)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0.0

    start_frame = 0
    end_frame = total_frames

    if VIDEO_START_SECONDS is not None:
        start_frame = int(VIDEO_START_SECONDS * fps)

    if VIDEO_END_SECONDS is not None:
        end_frame = min(total_frames, int(VIDEO_END_SECONDS * fps))

    start_frame = max(0, min(start_frame, total_frames))
    end_frame = max(start_frame, min(end_frame, total_frames))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"正在处理 Colon 视频：{video_path}")
    print(f"fps = {fps:.2f}")
    print(f"视频时长 = {duration_s:.2f} 秒")
    print(f"抽帧间隔 = 每 {SAMPLE_EVERY_N_FRAMES} 帧抽 1 帧")
    print(f"输出目录：{video_out_dir}")

    candidates = []
    sampled_count = 0

    frame_id = start_frame

    pbar = tqdm(total=end_frame - start_frame, desc=f"Colon 纯净抽帧 {video_path.stem}")

    while frame_id < end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % SAMPLE_EVERY_N_FRAMES == 0:
            sampled_count += 1
            time_s = frame_id / fps if fps > 0 else 0.0

            h, w = frame.shape[:2]
            valid_mask = get_valid_endoscopy_mask(frame)
            valid_view_ratio = float(valid_mask.sum() / (h * w))

            Q, q_metrics = compute_quality_score(frame, valid_mask)
            M, bbox = run_yolo_detections(model, frame)
            Z = compute_colon_polyp_display_score(frame, valid_mask, bbox)

            if not hard_reject(q_metrics, valid_view_ratio, M, bbox):
                score = compute_final_score(Q, Z, M)

                candidates.append({
                    "frame": frame.copy(),
                    "frame_id": int(frame_id),
                    "time_s": float(time_s),
                    "Q": float(Q),
                    "Z": float(Z),
                    "M": float(M),
                    "score": float(score),
                })

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    selected = select_keyframes(candidates)

    saved_paths = []

    for rank, rec in enumerate(selected, start=1):
        save_path = save_selected_frame(
            frame=rec["frame"],
            out_dir=video_out_dir,
            rank=rank,
            frame_id=rec["frame_id"],
            time_s=rec["time_s"],
            Q=rec["Q"],
            Z=rec["Z"],
            M=rec["M"],
            score=rec["score"],
        )
        saved_paths.append(save_path)

    runtime = time.perf_counter() - start_time

    print("\n处理完成")
    print(f"抽样帧数量：{sampled_count}")
    print(f"候选帧数量：{len(candidates)}")
    print(f"最终保存关键帧数量：{len(saved_paths)}")
    print(f"总运行时间：{runtime:.2f} 秒")
    print(f"输出目录：{video_out_dir}")

    return saved_paths


def main():
    make_dir(OUTPUT_DIR)

    model = None

    if USE_YOLO:
        if YOLO is None:
            print("没有成功导入 ultralytics，跳过 YOLO，M 分全部为 0。")
        elif not Path(YOLO_MODEL_PATH).exists():
            print(f"没有找到 YOLO 模型：{YOLO_MODEL_PATH}")
            print("跳过 YOLO，M 分全部为 0。")
        else:
            print("加载 YOLO 模型...")
            model = YOLO(YOLO_MODEL_PATH)
            print("YOLO 模型加载完成")

    process_video(
        video_path=VIDEO_PATH,
        model=model,
        output_root=OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
