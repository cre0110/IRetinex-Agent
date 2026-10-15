import os
from pathlib import Path

import cv2

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))

import torch
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry

from debug_utils import debug_log, debug_timer


MIN_CUDA_MEMORY_GB = float(os.getenv("MEDICALAI_MIN_CUDA_MEMORY_GB", "6"))


class MedicalAgent:
    def __init__(self, yolo_path, sam_path, device="cuda"):
        debug_log("MedicalAgent", "Resolving device", requested_device=device)
        self.device = self._resolve_device(device)
        debug_log("MedicalAgent", "Device resolved", device=self.device)

        with debug_timer("MedicalAgent", "Load YOLO model", path=yolo_path, device=self.device):
            self.det_model = YOLO(yolo_path)
        if hasattr(self.det_model, "to"):
            debug_log("MedicalAgent", "Moving YOLO model to device", device=self.device)
            self.det_model.to(self.device)

        with debug_timer("MedicalAgent", "Load MedSAM checkpoint", path=sam_path, device=self.device):
            sam = sam_model_registry["vit_b"](checkpoint=None)
            state_dict = torch.load(sam_path, map_location=torch.device(self.device))
            sam.load_state_dict(state_dict)
            sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        debug_log("MedicalAgent", "Model stack ready", device=self.device)

    def diagnose(self, image_bgr):
        debug_log("MedicalAgent", "Diagnose started", image_shape=image_bgr.shape, device=self.device)
        with debug_timer("MedicalAgent", "YOLO detection", device=self.device):
            results = self.det_model.predict(
                image_bgr,
                conf=0.3,
                verbose=False,
                device=self.device,
            )
        box_count = len(results[0].boxes)
        debug_log("MedicalAgent", "YOLO detection finished", boxes=box_count)
        if box_count == 0:
            return None, None

        box = results[0].boxes.xyxy[0].cpu().numpy()
        debug_log("MedicalAgent", "Top detection selected", bbox=box.tolist())

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with debug_timer("MedicalAgent", "MedSAM segmentation", device=self.device):
            self.predictor.set_image(image_rgb)
            masks, scores, _ = self.predictor.predict(box=box[None, :], multimask_output=False)
        debug_log(
            "MedicalAgent",
            "MedSAM segmentation finished",
            mask_shape=masks[0].shape,
            score=float(scores[0]) if len(scores) else None,
        )

        return box, masks[0]

    @staticmethod
    def _resolve_device(device):
        if device != "cuda":
            debug_log("MedicalAgent", "Using CPU because requested device is not cuda", requested_device=device)
            return "cpu"

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Current environment does not have a usable CUDA device. Falling back to CPU is required."
            )

        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_memory_gb < MIN_CUDA_MEMORY_GB:
            debug_log(
                "MedicalAgent",
                "Using CPU because CUDA memory is below the safe threshold",
                total_memory_gb=f"{total_memory_gb:.2f}",
                min_cuda_memory_gb=MIN_CUDA_MEMORY_GB,
            )
            return "cpu"

        supported_arches = set(torch.cuda.get_arch_list())
        major, minor = torch.cuda.get_device_capability(0)
        current_arch = f"sm_{major}{minor}"

        if current_arch not in supported_arches:
            raise RuntimeError(
                "Current PyTorch build does not support this GPU architecture. "
                f"detected={current_arch}, supported={sorted(supported_arches)}"
            )

        return "cuda"
