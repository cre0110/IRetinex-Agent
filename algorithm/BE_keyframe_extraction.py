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
# BE 视频关键帧筛选：纯净输出版
#
# 只保存最终筛选出的原始关键帧图片。
# 不再保存 failed、top_candidates_debug、mask_debug、annotated、csv 报告。
#
# Q = 图像质量分
# Z = BE交界候选分
# M = YOLO辅助可疑度
# score = 0.35 * Q + 0.50 * Z + 0.15 * M
# ============================================================


# =========================
# 1. 路径设置
# =========================

YOLO_MODEL_PATH = r"D:\MedicalAI\new weight\YOLO_upper_gi_stable_cv.pt"
VIDEO_PATH = r"D:\MedicalAI\视频\BE_demo.mp4"
OUTPUT_DIR = r"D:\MedicalAI\BE_keyframe_final_only"


# =========================
# 2. 参数设置
# =========================

USE_YOLO = True
SAMPLE_EVERY_N_FRAMES = 5

VIDEO_START_SECONDS = None
VIDEO_END_SECONDS = None

YOLO_CONF_THRES = 0.30

MAX_KEYFRAMES_PER_VIDEO = 5
MIN_TIME_GAP_BETWEEN_KEYFRAMES = 2.5


# =========================
# 3. 硬过滤阈值
# =========================

MIN_VALID_VIEW_RATIO = 0.25
MIN_CLARITY_SCORE = 0.10
MAX_OVEREXPOSED_RATIO = 0.28
MAX_SPECULAR_RATIO = 0.25


# =========================
# 4. BE 颜色阈值
# =========================

PALE_S_MAX = 95
PALE_V_MIN = 80

SALMON_S_MIN = 40
SALMON_V_MIN = 45
SALMON_H_LOW_MAX = 24
SALMON_H_HIGH_MIN = 165


# =========================
# 5. 工具函数
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


def clean_binary_mask(mask, min_area=80):
    mask_u8 = mask.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    cleaned = np.zeros(mask_u8.shape, dtype=np.uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 1

    return cleaned.astype(bool)


# =========================
# 6. Q：图像质量分
# =========================

def compute_quality_score(frame, valid_mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

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

    gray_valid = gray[valid_bool]
    brightness = float(np.mean(gray_valid))
    brightness_score = float(np.clip(1.0 - abs(brightness - 130.0) / 130.0, 0, 1))

    Q = (
        0.55 * clarity_score
        + 0.35 * glare_score
        + 0.10 * brightness_score
    )
    Q = float(np.clip(Q, 0, 1))

    metrics = {
        "clarity_score": clarity_score,
        "overexposed_ratio": overexposed_ratio,
        "specular_ratio": specular_ratio,
    }

    return Q, metrics


# =========================
# 7. Z：BE交界候选分
# =========================

def compute_junction_score(frame, valid_mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    valid_bool = valid_mask > 0

    pale_raw = (
        valid_bool
        & (S <= PALE_S_MAX)
        & (V >= PALE_V_MIN)
        & (V <= 245)
    )

    salmon_hue = ((H <= SALMON_H_LOW_MAX) | (H >= SALMON_H_HIGH_MIN))
    salmon_raw = (
        valid_bool
        & salmon_hue
        & (S >= SALMON_S_MIN)
        & (V >= SALMON_V_MIN)
        & (V <= 245)
    )

    pale_mask = clean_binary_mask(pale_raw, min_area=120)
    salmon_mask = clean_binary_mask(salmon_raw, min_area=120)

    pale_ratio = masked_ratio(pale_mask, valid_mask)
    salmon_ratio = masked_ratio(salmon_mask, valid_mask)

    pale_presence = np.clip(pale_ratio / 0.10, 0, 1)
    salmon_presence = np.clip(salmon_ratio / 0.10, 0, 1)
    coexistence_score = float(min(pale_presence, salmon_presence))

    dilate_kernel = np.ones((9, 9), np.uint8)

    pale_u8 = pale_mask.astype(np.uint8)
    salmon_u8 = salmon_mask.astype(np.uint8)

    pale_dilated = cv2.dilate(pale_u8, dilate_kernel, iterations=1).astype(bool)
    salmon_dilated = cv2.dilate(salmon_u8, dilate_kernel, iterations=1).astype(bool)

    contact_region = (
        (pale_mask & salmon_dilated)
        | (salmon_mask & pale_dilated)
    )

    contact_ratio = masked_ratio(contact_region, valid_mask)
    junction_boundary_score = float(np.clip(contact_ratio / 0.025, 0, 1))

    if salmon_ratio < 0.03:
        salmon_shape_score = 0.0
    elif salmon_ratio <= 0.35:
        salmon_shape_score = float(np.clip(salmon_ratio / 0.18, 0, 1))
    elif salmon_ratio <= 0.75:
        salmon_shape_score = float(1.0 - 0.45 * (salmon_ratio - 0.35) / 0.40)
    else:
        salmon_shape_score = 0.35

    salmon_shape_score = float(np.clip(salmon_shape_score, 0, 1))

    dark_lumen_mask = valid_bool & (V < 70) & (S < 150)
    dark_lumen_ratio = masked_ratio(dark_lumen_mask, valid_mask)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = masked_ratio(edges > 0, valid_mask)

    lumen_score = float(np.clip(dark_lumen_ratio / 0.18, 0, 1))
    if dark_lumen_ratio > 0.55:
        lumen_score *= float(max(0.35, 1.0 - (dark_lumen_ratio - 0.55) / 0.35))

    edge_score = float(np.clip(edge_density / 0.035, 0, 1))
    context_score = float(np.clip(0.55 * lumen_score + 0.45 * edge_score, 0, 1))

    Z = (
        0.30 * coexistence_score
        + 0.35 * junction_boundary_score
        + 0.20 * salmon_shape_score
        + 0.15 * context_score
    )
    Z = float(np.clip(Z, 0, 1))

    return Z


# =========================
# 8. M：YOLO辅助可疑度
# =========================

def run_yolo_best_detection(model, frame):
    if model is None:
        return 0.0

    results = model.predict(
        source=frame,
        conf=YOLO_CONF_THRES,
        verbose=False
    )

    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return 0.0

    boxes = results[0].boxes
    confs = boxes.conf.detach().cpu().numpy()

    return float(np.max(confs))


# =========================
# 9. 过滤与排序
# =========================

def hard_reject(q_metrics, valid_view_ratio):
    if valid_view_ratio < MIN_VALID_VIEW_RATIO:
        return True

    if q_metrics["clarity_score"] < MIN_CLARITY_SCORE:
        return True

    if (
        q_metrics["overexposed_ratio"] > MAX_OVEREXPOSED_RATIO
        or q_metrics["specular_ratio"] > MAX_SPECULAR_RATIO
    ):
        return True

    return False


def compute_final_score(Q, Z, M):
    return float(np.clip(0.35 * Q + 0.50 * Z + 0.15 * M, 0, 1))


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
# 10. 主流程
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

    print(f"正在处理 BE 视频：{video_path}")
    print(f"fps = {fps:.2f}")
    print(f"视频时长 = {duration_s:.2f} 秒")
    print(f"抽帧间隔 = 每 {SAMPLE_EVERY_N_FRAMES} 帧抽 1 帧")
    print(f"输出目录：{video_out_dir}")

    candidates = []
    sampled_count = 0

    frame_id = start_frame

    pbar = tqdm(total=end_frame - start_frame, desc=f"BE 纯净抽帧 {video_path.stem}")

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

            if not hard_reject(q_metrics, valid_view_ratio):
                Z = compute_junction_score(frame, valid_mask)
                M = run_yolo_best_detection(model, frame)
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
