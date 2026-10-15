from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from debug_utils import debug_log, debug_timer
from .metrics import image_entropy, image_std, ssim


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}


@dataclass(frozen=True)
class EnhancementResult:
    original_path: str
    dark_path: str
    iretinex_path: str
    fusion_path: str
    transmission_path: str
    detail_path: str
    dark_weight: float
    iretinex_weight: float
    entropy_before: float
    entropy_after: float
    std_before: float
    std_after: float
    ssim: float | None
    processing_time_ms: int


def validate_image_file(path: str | Path) -> None:
    p = Path(path)
    debug_log("IRetinex", "Validating image file", path=p)
    ext = p.suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("仅支持 png、jpg、jpeg、bmp 格式的图像。")

    try:
        with Image.open(p) as img:
            img.verify()
    except Exception as exc:
        raise ValueError("上传文件不是有效图像。") from exc


def enhance_image(
    input_path: str | Path,
    output_dir: str | Path,
    weights: tuple[float, float] = (0.7, 0.3),
) -> EnhancementResult:
    validate_image_file(input_path)
    debug_log("IRetinex", "Enhancement started", input=input_path, output_dir=output_dir, weights=weights)
    start = perf_counter()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dark_weight, iretinex_weight = weights
    if not np.isclose(dark_weight + iretinex_weight, 1.0):
        raise ValueError("融合权重之和必须为 1。")

    with debug_timer("IRetinex", "Load RGB image", path=input_path):
        image = _load_rgb(input_path)
    debug_log("IRetinex", "Image loaded", shape=image.shape)
    run_id = Path(input_path).stem + "_" + uuid4().hex[:8]

    with debug_timer("IRetinex", "Dark channel dehaze"):
        dark, t_rough, atmosphere = _dark_channel_dehaze(image)
    debug_log("IRetinex", "Atmosphere estimated", atmosphere=np.round(atmosphere, 4).tolist())
    with debug_timer("IRetinex", "IRetinex layer recovery"):
        iretinex, t_final, detail = _iretinex_layer(image, t_rough, atmosphere)
    fusion = np.clip(dark_weight * dark + iretinex_weight * iretinex, 0.0, 1.0)
    debug_log("IRetinex", "Fusion created", dark_weight=dark_weight, iretinex_weight=iretinex_weight)

    dark_path = output / f"{run_id}_dark.png"
    iretinex_path = output / f"{run_id}_iretinex.png"
    fusion_path = output / f"{run_id}_fusion.png"
    transmission_path = output / f"{run_id}_transmission.png"
    detail_path = output / f"{run_id}_detail.png"

    with debug_timer("IRetinex", "Save enhancement artifacts", output_dir=output):
        _save_rgb(dark, dark_path)
        _save_rgb(iretinex, iretinex_path)
        _save_rgb(fusion, fusion_path)
        _save_gray(t_final, transmission_path)
        _save_gray(detail, detail_path)

    elapsed_ms = int(round((perf_counter() - start) * 1000))
    entropy_before = image_entropy(image)
    entropy_after = image_entropy(fusion)
    std_before = image_std(image)
    std_after = image_std(fusion)
    ssim_value = ssim(image, fusion)
    debug_log(
        "IRetinex",
        "Enhancement finished",
        elapsed_ms=elapsed_ms,
        entropy_before=f"{entropy_before:.4f}",
        entropy_after=f"{entropy_after:.4f}",
        std_before=f"{std_before:.4f}",
        std_after=f"{std_after:.4f}",
        ssim=f"{ssim_value:.4f}" if ssim_value is not None else None,
        fusion_path=fusion_path,
    )

    return EnhancementResult(
        original_path=str(input_path),
        dark_path=str(dark_path),
        iretinex_path=str(iretinex_path),
        fusion_path=str(fusion_path),
        transmission_path=str(transmission_path),
        detail_path=str(detail_path),
        dark_weight=dark_weight,
        iretinex_weight=iretinex_weight,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        std_before=std_before,
        std_after=std_after,
        ssim=ssim_value,
        processing_time_ms=elapsed_ms,
    )


def _load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _save_rgb(image: np.ndarray, path: str | Path) -> None:
    arr = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _save_gray(image: np.ndarray, path: str | Path) -> None:
    arr = np.clip(np.rint(_normalize01(image) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _dark_channel_dehaze(
    image: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.92,
    t0: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dark = _dark_channel(image, patch_size)
    atmosphere = _estimate_atmosphere(image, dark)
    normalized = image / np.maximum(atmosphere.reshape(1, 1, 3), 1e-6)
    transmission = 1.0 - omega * _dark_channel(normalized, patch_size)
    transmission = _smooth_map(transmission, radius=3)
    transmission = np.clip(transmission, t0, 0.95)
    radiance = _recover_radiance(image, atmosphere, transmission, t0=t0)
    return radiance, transmission, atmosphere


def _dark_channel(image: np.ndarray, patch_size: int) -> np.ndarray:
    min_channel = np.min(image, axis=2)
    gray = Image.fromarray(np.clip(min_channel * 255.0, 0, 255).astype(np.uint8), mode="L")
    size = patch_size if patch_size % 2 == 1 else patch_size + 1
    filtered = gray.filter(ImageFilter.MinFilter(size))
    return np.asarray(filtered, dtype=np.float32) / 255.0


def _estimate_atmosphere(image: np.ndarray, dark: np.ndarray) -> np.ndarray:
    flat_dark = dark.ravel()
    flat_image = image.reshape(-1, 3)
    count = max(int(flat_dark.size * 0.001), 1)
    indices = np.argpartition(flat_dark, -count)[-count:]
    candidates = flat_image[indices]
    scores = candidates.sum(axis=1)
    return candidates[int(np.argmax(scores))]


def _recover_radiance(
    image: np.ndarray,
    atmosphere: np.ndarray,
    transmission: np.ndarray,
    t0: float = 0.1,
) -> np.ndarray:
    safe_t = np.maximum(transmission[..., None], t0)
    return np.clip((image - atmosphere.reshape(1, 1, 3)) / safe_t + atmosphere.reshape(1, 1, 3), 0.0, 1.0)


def _iretinex_layer(
    image: np.ndarray,
    t_rough: np.ndarray,
    atmosphere: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = _rgb_to_gray(image)
    structure = _gaussian(gray, radius=0.5)
    haze_layer = _gaussian(gray, radius=15.0)

    for _ in range(3):
        haze_layer = haze_layer * (1.0 - structure / 2.0)
        structure = structure * (1.0 - haze_layer / 2.0)
        haze_layer = np.clip(haze_layer, 0.0, 1.0)
        structure = np.clip(structure, 0.0, 1.0)

    t_corrected = np.clip(t_rough * (1.0 - haze_layer) + 0.1, 0.1, 0.95)
    g3 = _gaussian(gray, radius=3.0)
    g7 = _gaussian(gray, radius=7.0)
    detail = _normalize01((gray - g3) + 2.0 * (g3 - g7))
    t_final = np.clip(t_corrected + 0.3 * detail, 0.1, 0.98)

    recovered = _recover_radiance(image, atmosphere, t_final, t0=0.1)
    enhanced = _clinical_postprocess(recovered)
    return enhanced, t_final, detail


def _clinical_postprocess(image: np.ndarray) -> np.ndarray:
    arr = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    channels = []
    for c in range(3):
        channel = Image.fromarray(arr[..., c], mode="L")
        equalized = ImageOps.autocontrast(channel, cutoff=1)
        blended = Image.blend(channel, equalized, alpha=0.28)
        channels.append(blended)
    rgb = Image.merge("RGB", channels)
    sharpened = rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=45, threshold=3))
    return np.asarray(sharpened, dtype=np.float32) / 255.0


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return (
        0.299 * image[..., 0]
        + 0.587 * image[..., 1]
        + 0.114 * image[..., 2]
    ).astype(np.float32)


def _gaussian(image: np.ndarray, radius: float) -> np.ndarray:
    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L")
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _smooth_map(image: np.ndarray, radius: int = 3) -> np.ndarray:
    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L")
    blurred = pil.filter(ImageFilter.BoxBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _normalize01(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    min_value = float(arr.min())
    max_value = float(arr.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_value) / (max_value - min_value)
