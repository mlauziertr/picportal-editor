#!/usr/bin/env python3
"""Local-only Grounding DINO and MediaPipe Pose worker.

The Rust application owns the protocol and sends in-memory JPEG derivatives.
This worker never downloads weights, opens a network connection, or writes
photo data. DINO artifacts are loaded with local_files_only=True.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(target: torch.device) -> None:
    if target.type == "mps":
        torch.mps.synchronize()


def decode_image(value: str) -> Image.Image:
    raw = base64.b64decode(value, validate=True)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def dino_paths() -> Path:
    return Path(os.environ["PICPORTAL_CULLING_MODEL_DIR"]) / "dino"


def load_dino() -> tuple[Any, Any, torch.device]:
    model_dir = dino_paths()
    target = device()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_dir, local_files_only=True
    )
    model.eval().to(target)
    sync(target)
    return processor, model, target


def dino_detect(
    processor: Any, model: Any, target: torch.device, request: dict[str, Any]
) -> dict[str, Any]:
    image = decode_image(request["image"])
    phrase = request.get("phrase")
    if not isinstance(phrase, str) or not phrase.strip():
        raise ValueError("phrase is required")
    inputs = processor(images=image, text=phrase, return_tensors="pt").to(target)
    with torch.inference_mode():
        outputs = model(**inputs)
    sync(target)
    height, width = image.height, image.width
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.4,
        text_threshold=0.3,
        target_sizes=[(height, width)],
    )[0]
    labels = result.get("text_labels", ["subject"] * len(result["boxes"]))
    boxes = []
    for box, score in zip(result["boxes"], result["scores"]):
        x0, y0, x1, y1 = [float(value) for value in box.tolist()]
        boxes.append(
            {
                "x": max(0.0, x0),
                "y": max(0.0, y0),
                "width": max(0.0, min(float(width), x1) - max(0.0, x0)),
                "height": max(0.0, min(float(height), y1) - max(0.0, y0)),
                "score": float(score),
                "label": str(labels[len(boxes)]),
            }
        )
    return {"boxes": boxes, "width": width, "height": height, "phrase": phrase}


def pose_path() -> Path:
    return Path(os.environ["PICPORTAL_CULLING_MODEL_DIR"]) / "pose_landmarker_lite.task"


def load_pose() -> Any:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = pose_path()
    if not model_path.is_file():
        raise FileNotFoundError(str(model_path))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=8,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp, mp_vision.PoseLandmarker.create_from_options(options)


def pose_detect(runtime: tuple[Any, Any], request: dict[str, Any]) -> dict[str, Any]:
    mp, landmarker = runtime
    image = decode_image(request["image"])
    array = np.ascontiguousarray(np.asarray(image))
    height, width = array.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
    result = landmarker.detect(mp_image)
    poses = []
    for landmarks in result.pose_landmarks:
        def point(index: int) -> dict[str, float]:
            landmark = landmarks[index]
            return {"x": float(landmark.x * width), "y": float(landmark.y * height)}

        poses.append(
            {"nose": point(0), "torso": [point(11), point(12), point(23), point(24)]}
        )
    return {"poses": poses, "width": width, "height": height}


def run() -> None:
    dino: tuple[Any, Any, torch.device] | None = None
    pose: tuple[Any, Any] | None = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("operation")
            if operation == "detect":
                if dino is None:
                    dino = load_dino()
                response = dino_detect(*dino, request)
            elif operation == "pose":
                if pose is None:
                    pose = load_pose()
                response = pose_detect(pose, request)
            else:
                raise ValueError(f"unsupported operation: {operation}")
            print(json.dumps({"ok": True, **response}), flush=True)
        except Exception as error:
            print(
                json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}),
                flush=True,
            )


if __name__ == "__main__":
    run()
