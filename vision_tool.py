from __future__ import annotations

import gc
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from algorithm.iretinex import enhance_image
from debug_utils import debug_log, debug_timer
from medical_agent import MedicalAgent


CURRENT_DIR = Path(__file__).resolve().parent
LESION_MODEL_CONFIGS = {
    "polyp": {
        "label": "colorectal_polyp",
        "display_name": "colon polyp",
        "yolo_path": CURRENT_DIR / "weights" / "YOLO_colon.pt",
        "sam_path": CURRENT_DIR / "weights" / "MedSAM_colon.pt",
    },
    "barrett": {
        "label": "barrett_esophagus",
        "display_name": "upper GI Barrett esophagus",
        "yolo_path": CURRENT_DIR / "weights" / "YOLO_upper_gi.pt",
        "sam_path": CURRENT_DIR / "weights" / "MedSAM_upper_gi.pt",
    },
}

_global_agents = {}
_global_agent_devices = {}
_global_agent_errors = {}


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cuda out of memory" in message or "cublas_status_alloc_failed" in message


def _clear_agent_cache(lesion_type: str) -> None:
    _global_agents.pop(lesion_type, None)
    _global_agent_devices.pop(lesion_type, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_lesion_type(lesion_type: str | None) -> str:
    value = str(lesion_type or "polyp").strip().lower()
    aliases = {
        "colon": "polyp",
        "colon_polyp": "polyp",
        "colorectal_polyp": "polyp",
        "polyp": "polyp",
        "upper_gi": "barrett",
        "upper_gi_be": "barrett",
        "be": "barrett",
        "barrett": "barrett",
        "barrett_esophagus": "barrett",
    }
    return aliases.get(value, "polyp")


def _create_agent(lesion_type: str | None = None):
    lesion_type = normalize_lesion_type(lesion_type)
    model_config = LESION_MODEL_CONFIGS[lesion_type]

    if lesion_type in _global_agents:
        debug_log(
            "vision_tool",
            "Reusing cached MedicalAgent",
            lesion_type=lesion_type,
            device=_global_agent_devices.get(lesion_type),
        )
        return _global_agents[lesion_type], _global_agent_devices.get(lesion_type)

    init_errors = []
    for device in ("cuda", "cpu"):
        try:
            with debug_timer("vision_tool", "Initialize MedicalAgent", device=device):
                agent = MedicalAgent(
                    yolo_path=str(model_config["yolo_path"]),
                    sam_path=str(model_config["sam_path"]),
                    device=device,
                )
            actual_device = agent.device
            _global_agents[lesion_type] = agent
            _global_agent_devices[lesion_type] = actual_device
            _global_agent_errors[lesion_type] = None
            debug_log("vision_tool", "Models ready", lesion_type=lesion_type, requested_device=device, actual_device=actual_device)
            return agent, actual_device
        except Exception as exc:
            init_errors.append(f"{device}: {exc}")
            debug_log(
                "vision_tool",
                "Model initialization attempt failed",
                lesion_type=lesion_type,
                device=device,
                error=exc,
            )

    _global_agent_errors[lesion_type] = " | ".join(init_errors)
    debug_log("vision_tool", "All model initialization attempts failed", lesion_type=lesion_type, error=_global_agent_errors[lesion_type])
    raise RuntimeError(f"Failed to initialize MedicalAgent for {lesion_type}. {_global_agent_errors[lesion_type]}")


def get_runtime_info() -> dict:
    return {
        "devices": dict(_global_agent_devices),
        "init_errors": dict(_global_agent_errors),
    }


def _enhancement_payload(enhance_result) -> dict:
    return {
        "original_path": enhance_result.original_path,
        "dark_path": enhance_result.dark_path,
        "iretinex_path": enhance_result.iretinex_path,
        "fusion_path": enhance_result.fusion_path,
        "transmission_path": enhance_result.transmission_path,
        "detail_path": enhance_result.detail_path,
        "dark_weight": enhance_result.dark_weight,
        "iretinex_weight": enhance_result.iretinex_weight,
        "entropy_before": enhance_result.entropy_before,
        "entropy_after": enhance_result.entropy_after,
        "std_before": enhance_result.std_before,
        "std_after": enhance_result.std_after,
        "ssim": enhance_result.ssim,
        "processing_time_ms": enhance_result.processing_time_ms,
    }


def enhance_input_image(input_image_path: str, output_dir: str = "./outputs") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    debug_log(
        "vision_tool",
        "Enhancement request received",
        input=input_image_path,
        output_dir=output_dir,
    )

    try:
        with debug_timer("vision_tool", "IRetinex enhancement", input=input_image_path):
            enhance_result = enhance_image(input_image_path, output_dir=output_dir, weights=(0.7, 0.3))
        debug_log(
            "vision_tool",
            "Enhancement metrics",
            fusion_path=enhance_result.fusion_path,
            entropy_before=f"{enhance_result.entropy_before:.4f}",
            entropy_after=f"{enhance_result.entropy_after:.4f}",
            std_before=f"{enhance_result.std_before:.4f}",
            std_after=f"{enhance_result.std_after:.4f}",
            ssim=f"{enhance_result.ssim:.4f}" if enhance_result.ssim is not None else None,
            processing_time_ms=enhance_result.processing_time_ms,
        )
        payload = _enhancement_payload(enhance_result)
        payload.update(
            {
                "status": "success",
                "message": "IRetinex 增强已完成。",
            }
        )
        return payload
    except Exception as exc:
        debug_log("vision_tool", "Enhancement failed", error=exc)
        return {
            "status": "error",
            "device": None,
            "message": str(exc),
        }


def diagnose_original_image(input_image_path: str, output_dir: str = "./outputs", lesion_type: str | None = None) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_image_path)
    lesion_type = normalize_lesion_type(lesion_type)
    model_config = LESION_MODEL_CONFIGS[lesion_type]
    debug_log(
        "vision_tool",
        "Original-image diagnosis request received",
        input=input_image_path,
        output_dir=output_dir,
        lesion_type=lesion_type,
    )

    try:
        agent, agent_device = _create_agent(lesion_type)

        with debug_timer("vision_tool", "Load original image for YOLO + MedSAM", path=input_image_path):
            original_bgr = cv2.imdecode(
                np.fromfile(input_image_path, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        if original_bgr is None:
            raise RuntimeError(f"Failed to read original image: {input_image_path}")
        debug_log("vision_tool", "Original image loaded for diagnosis", shape=original_bgr.shape)

        try:
            with debug_timer("vision_tool", "YOLO + MedSAM diagnose original image", lesion_type=lesion_type, device=agent_device):
                pred_box, pred_mask = agent.diagnose(original_bgr)
        except RuntimeError as exc:
            if agent_device != "cuda" or not _is_cuda_oom(exc):
                raise
            debug_log(
                "vision_tool",
                "CUDA out of memory during diagnosis; retrying on CPU",
                lesion_type=lesion_type,
                error=exc,
            )
            _clear_agent_cache(lesion_type)
            with debug_timer("vision_tool", "Initialize CPU fallback MedicalAgent", lesion_type=lesion_type):
                agent = MedicalAgent(
                    yolo_path=str(model_config["yolo_path"]),
                    sam_path=str(model_config["sam_path"]),
                    device="cpu",
                )
            _global_agents[lesion_type] = agent
            _global_agent_devices[lesion_type] = "cpu"
            agent_device = "cpu"
            with debug_timer("vision_tool", "YOLO + MedSAM diagnose original image on CPU", lesion_type=lesion_type):
                pred_box, pred_mask = agent.diagnose(original_bgr)

        if pred_mask is None:
            debug_log("vision_tool", "No lesion detected on original image")
            return {
                "status": "success",
                "polyp_detected": False,
                "lesion_type": lesion_type,
                "disease_type": model_config["label"],
                "original_image_path": input_image_path,
                "original_shape": list(original_bgr.shape[:2]),
                "device": agent_device,
                "model_name": model_config["display_name"],
                "yolo_path": str(model_config["yolo_path"]),
                "sam_path": str(model_config["sam_path"]),
                "message": "YOLO/MedSAM 已检查原图，未检测到明显疑似病灶区域。",
            }

        x1, y1, x2, y2 = map(int, pred_box)
        mask_pixels = int(np.count_nonzero(pred_mask))
        mask_output_path = os.path.join(output_dir, f"mask_{Path(filename).stem}.png")
        cv2.imencode(".png", (pred_mask > 0).astype(np.uint8) * 255)[1].tofile(mask_output_path)

        lesion_image_path = os.path.join(output_dir, f"lesion_marked_{Path(filename).stem}.jpg")
        lesion_img = original_bgr.copy()
        overlay = original_bgr.copy()
        overlay[pred_mask > 0] = [0, 0, 255]
        lesion_img = cv2.addWeighted(overlay, 0.4, lesion_img, 0.6, 0)
        cv2.rectangle(lesion_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imencode(".jpg", lesion_img)[1].tofile(lesion_image_path)
        debug_log(
            "vision_tool",
            "原图病灶位置标注图已生成",
            output=lesion_image_path,
            exists=os.path.isfile(lesion_image_path),
        )

        debug_log(
            "vision_tool",
            "Lesion detected on original image",
            bbox=[x1, y1, x2, y2],
            mask_pixels=mask_pixels,
            mask_path=mask_output_path,
            lesion_image_path=lesion_image_path,
        )

        return {
            "status": "success",
            "polyp_detected": True,
            "lesion_type": lesion_type,
            "disease_type": model_config["label"],
            "bbox": [x1, y1, x2, y2],
            "mask_path": mask_output_path,
            "lesion_image_path": lesion_image_path,
            "mask_pixels": mask_pixels,
            "original_image_path": input_image_path,
            "original_shape": list(original_bgr.shape[:2]),
            "device": agent_device,
            "model_name": model_config["display_name"],
            "yolo_path": str(model_config["yolo_path"]),
            "sam_path": str(model_config["sam_path"]),
            "message": "已在原图上检测并标注疑似病灶区域。",
        }

    except Exception as exc:
        debug_log("vision_tool", "Original-image diagnosis failed", error=exc)
        return {
            "status": "error",
            "lesion_type": lesion_type,
            "disease_type": model_config["label"],
            "device": _global_agent_devices.get(lesion_type),
            "message": str(exc),
        }


def annotate_enhanced_image(
    enhancement_result: dict,
    lesion_result: dict,
    output_dir: str = "./outputs",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    input_image_path = enhancement_result.get("original_path") or lesion_result.get("original_image_path") or "image.jpg"
    filename = os.path.basename(input_image_path)
    output_image_path = os.path.join(output_dir, f"analyzed_{filename}")
    lesion_type = normalize_lesion_type(lesion_result.get("lesion_type"))
    agent_device = lesion_result.get("device") or _global_agent_devices.get(lesion_type)
    debug_log(
        "vision_tool",
        "Enhanced-image annotation request received",
        fusion_path=enhancement_result.get("fusion_path"),
        lesion_detected=lesion_result.get("polyp_detected"),
        lesion_type=lesion_type,
        output_dir=output_dir,
        output=output_image_path,
    )

    try:
        if lesion_result.get("status") != "success":
            raise RuntimeError(f"Lesion result is not successful: {lesion_result.get('message', 'unknown error')}")

        fusion_path = enhancement_result["fusion_path"]
        with debug_timer("vision_tool", "Load enhanced fusion image for annotation", path=fusion_path):
            enhanced_bgr = cv2.imdecode(
                np.fromfile(fusion_path, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        if enhanced_bgr is None:
            raise RuntimeError(f"Failed to read enhanced image: {fusion_path}")
        debug_log("vision_tool", "Enhanced image loaded for annotation", shape=enhanced_bgr.shape)

        if not lesion_result.get("polyp_detected"):
            debug_log("vision_tool", "No lesion detected, saving enhanced image", output=output_image_path)
            cv2.imencode(".jpg", enhanced_bgr)[1].tofile(output_image_path)
            debug_log("vision_tool", "Process image finished", detected=False, output=output_image_path)
            return {
                "status": "success",
                "polyp_detected": False,
                "enhancement_result": enhancement_result,
                "lesion_result": lesion_result,
                "result_image_path": output_image_path,
                "lesion_type": lesion_type,
                "disease_type": lesion_result.get("disease_type"),
                "device": agent_device,
                "message": "YOLO/MedSAM 在原图中未检测到明显疑似病灶区域。",
            }

        bbox = lesion_result.get("bbox") or []
        if len(bbox) != 4:
            raise RuntimeError("Lesion result is missing a valid bbox.")
        x1, y1, x2, y2 = map(int, bbox)
        mask_path = lesion_result.get("mask_path")
        if not mask_path:
            raise RuntimeError("Lesion result is missing mask_path.")
        pred_mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            raise RuntimeError(f"Failed to read lesion mask: {mask_path}")

        original_shape = lesion_result.get("original_shape") or list(pred_mask.shape[:2])
        original_h, original_w = map(int, original_shape[:2])
        enhanced_h, enhanced_w = enhanced_bgr.shape[:2]
        if (original_w, original_h) != (enhanced_w, enhanced_h):
            scale_x = enhanced_w / original_w
            scale_y = enhanced_h / original_h
            draw_box = [
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
                int(round(x2 * scale_x)),
                int(round(y2 * scale_y)),
            ]
            draw_mask = cv2.resize(
                pred_mask.astype(np.uint8),
                (enhanced_w, enhanced_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            debug_log(
                "vision_tool",
                "Mapped lesion coordinates from original image to enhanced image",
                original_shape=original_shape,
                enhanced_shape=enhanced_bgr.shape,
                original_bbox=[x1, y1, x2, y2],
                draw_bbox=draw_box,
            )
        else:
            draw_box = [x1, y1, x2, y2]
            draw_mask = pred_mask > 0

        mask_img = enhanced_bgr.copy()
        overlay = enhanced_bgr.copy()
        overlay[draw_mask] = [0, 0, 255]
        mask_img = cv2.addWeighted(overlay, 0.4, mask_img, 0.6, 0)
        dx1, dy1, dx2, dy2 = draw_box
        cv2.rectangle(mask_img, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)

        cv2.imencode(".jpg", mask_img)[1].tofile(output_image_path)
        debug_log("vision_tool", "Annotated result saved", output=output_image_path)
        debug_log("vision_tool", "Process image finished", detected=True, output=output_image_path)

        return {
            "status": "success",
            "polyp_detected": True,
            "bbox": [x1, y1, x2, y2],
            "annotation_bbox": draw_box,
            "enhancement_result": enhancement_result,
            "lesion_result": lesion_result,
            "result_image_path": output_image_path,
            "lesion_type": lesion_type,
            "disease_type": lesion_result.get("disease_type"),
            "device": agent_device,
            "message": "已将原图病灶定位结果标注到 iRetinex 增强图上。",
        }

    except Exception as exc:
        debug_log("vision_tool", "Enhanced-image annotation failed", error=exc)
        return {
            "status": "error",
            "lesion_type": lesion_type,
            "disease_type": lesion_result.get("disease_type"),
            "device": agent_device,
            "enhancement_result": enhancement_result,
            "lesion_result": lesion_result,
            "message": str(exc),
        }


def diagnose_enhanced_image(enhancement_result: dict, output_dir: str = "./outputs", lesion_type: str | None = None) -> dict:
    input_image_path = enhancement_result.get("original_path") or enhancement_result.get("fusion_path") or "image.jpg"
    lesion_result = diagnose_original_image(input_image_path, output_dir=output_dir, lesion_type=lesion_type)
    if lesion_result.get("status") != "success":
        lesion_result["enhancement_result"] = enhancement_result
        return lesion_result
    return annotate_enhanced_image(enhancement_result, lesion_result, output_dir=output_dir)


def process_image(input_image_path: str, output_dir: str = "./outputs", lesion_type: str | None = None) -> dict:
    lesion_type = normalize_lesion_type(lesion_type)
    debug_log(
        "vision_tool",
        "Process image request received",
        input=input_image_path,
        output_dir=output_dir,
        lesion_type=lesion_type,
    )
    lesion_result = diagnose_original_image(input_image_path, output_dir=output_dir, lesion_type=lesion_type)
    if lesion_result.get("status") != "success":
        return lesion_result

    enhancement_result = enhance_input_image(input_image_path, output_dir=output_dir)
    if enhancement_result.get("status") != "success":
        return enhancement_result
    return annotate_enhanced_image(enhancement_result, lesion_result, output_dir=output_dir)
