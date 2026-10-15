from __future__ import annotations

import math

import numpy as np


def to_float_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.max(initial=0) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    arr = to_float_image(image)
    if arr.ndim == 2:
        return arr
    return (
        0.299 * arr[..., 0]
        + 0.587 * arr[..., 1]
        + 0.114 * arr[..., 2]
    ).astype(np.float32)


def image_entropy(image: np.ndarray) -> float:
    gray = rgb_to_gray(image)
    pixels = np.clip(np.rint(gray * 255.0), 0, 255).astype(np.uint8)
    hist = np.bincount(pixels.ravel(), minlength=256).astype(np.float64)
    probs = hist / max(float(hist.sum()), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def image_std(image: np.ndarray) -> float:
    gray = rgb_to_gray(image)
    return float(np.std(gray))


def psnr(reference: np.ndarray, target: np.ndarray) -> float | None:
    ref = to_float_image(reference)
    tar = to_float_image(target)
    if ref.shape != tar.shape:
        return None
    mse = float(np.mean((ref - tar) ** 2))
    if mse <= 1e-12:
        return math.inf
    return float(10.0 * math.log10(1.0 / mse))


def ssim(reference: np.ndarray, target: np.ndarray) -> float | None:
    ref = rgb_to_gray(reference)
    tar = rgb_to_gray(target)
    if ref.shape != tar.shape:
        return None

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = float(ref.mean())
    mu_y = float(tar.mean())
    var_x = float(ref.var())
    var_y = float(tar.var())
    cov_xy = float(((ref - mu_x) * (tar - mu_y)).mean())

    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if denominator == 0:
        return None
    return float(max(min(numerator / denominator, 1.0), -1.0))

