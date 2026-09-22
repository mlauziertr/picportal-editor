#!/usr/bin/env python3
"""Local-only Grounding DINO and native VGG review worker.

The Rust application owns the protocol and sends in-memory JPEG derivatives.
This worker never downloads weights, opens a network connection, or writes
photo data.  Model loading is local_files_only and checkpoint loading is
weights_only=True.
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
RANGES = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))




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
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir, local_files_only=True)
    model.eval().to(target)
    sync(target)
    return processor, model, target


def dino_detect(processor: Any, model: Any, target: torch.device, request: dict[str, Any]) -> dict[str, Any]:
    image = decode_image(request["image"])
    phrase = request.get("phrase")
    if not isinstance(phrase, str) or not phrase.strip():
        raise ValueError("phrase is required")
    inputs = processor(images=image, text=phrase, return_tensors="pt")
    inputs = inputs.to(target)
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
                "label": str(result.get("text_labels", ["subject"] * len(result["boxes"]))[len(boxes)]),
            }
        )
    return {"boxes": boxes, "width": width, "height": height, "phrase": phrase, "device": str(target)}


def make_vgg_features() -> nn.Sequential:
    layers: list[nn.Module] = []
    in_ch = 3
    for value in VGG16:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_ch, int(value), kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_ch = int(value)
    return nn.Sequential(*layers)


class VGGFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = make_vgg_features()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        for index, (start, end) in enumerate(RANGES):
            for layer in range(start, end):
                x = self.features[layer](x)
            output[f"x{index + 1}"] = x
        return output


class SKSPP(nn.Module):
    def __init__(self, features: int, branches: int = 4) -> None:
        super().__init__()
        depth = max(int(features / 16), 32)
        convs: list[nn.Module] = []
        for index in range(1, branches):
            kernel = 1 + index * 2
            convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        features,
                        features,
                        kernel_size=kernel,
                        dilation=kernel,
                        padding=((kernel * (index * 2)) + 1) // 2,
                    ),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False),
                )
            )
        self.convs = nn.ModuleList(convs)
        self.fc = nn.Linear(features, depth)
        self.fcs = nn.ModuleList([nn.Linear(depth, features) for _ in range(branches)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
            features = torch.cat([features, x.unsqueeze(1)], dim=1)
        pooled = features.sum(dim=1).mean(-1).mean(-1)
        hidden = self.fc(pooled)
        vectors = [fc(hidden).unsqueeze(1) for fc in self.fcs]
        attention = self.softmax(torch.cat(vectors, dim=1)).unsqueeze(-1).unsqueeze(-1)
        return (features * attention).sum(dim=1)


class UpsampleSK(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid = in_ch // 4
        self.prev = nn.Conv2d(in_ch, mid, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(mid)
        self.next = nn.Conv2d(mid, out_ch, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.sk = SKSPP(mid)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2)
        x = self.bn(self.relu(self.prev(x)))
        x = self.sk(x)
        return self.bn2(self.relu(self.next(x)))


class SideHeads(nn.Module):
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.sides = nn.ModuleList([nn.Conv2d(in_ch, 1, kernel_size=1), nn.Conv2d(in_ch, 1, kernel_size=1)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sides[0](x), self.sides[1](x)


class MergeAtt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv2(self.bn(self.relu(self.conv1(x)))))


class DefocusVGG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrained_net = VGGFeatures()
        channels = ((512, 512), (512, 256), (256, 128), (128, 64), (64, 32))
        for index, (in_ch, out_ch) in enumerate(channels, start=1):
            setattr(self, f"deconv{index}", UpsampleSK(in_ch, out_ch))
            setattr(self, f"classifier{index}", SideHeads(out_ch))
        for index in range(1, 5):
            setattr(self, f"sideatt{index}", MergeAtt())
        self.fusion = nn.Conv2d(5, 1, kernel_size=1)
        self.fusiond = nn.Conv2d(5, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.pretrained_net(x)
        skips = [feats["x5"], feats["x4"], feats["x3"], feats["x2"], feats["x1"]]
        sides: list[torch.Tensor] = []
        score = skips[0]
        for index in range(1, 6):
            score = getattr(self, f"deconv{index}")(score)
            side, depth = getattr(self, f"classifier{index}")(score)
            sides.append(side)
            if index < 5:
                gate = getattr(self, f"sideatt{index}")(torch.cat([side, depth], 1))
                score = score * gate + skips[index]
        stacked = torch.cat([F.interpolate(side, size=x.shape[-2:], mode="bilinear") for side in sides], dim=1)
        return self.fusion(stacked)


def load_focus() -> tuple[DefocusVGG, torch.device]:
    model_path = Path(os.environ["PICPORTAL_CULLING_MODEL_DIR"]) / "vgg_best.pth"
    target = device()
    raw = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("checkpoint is not a tensor dictionary")
    model = DefocusVGG()
    model.load_state_dict(raw, strict=True)
    model.eval().to(target)
    sync(target)
    return model, target


def focus_score(model: DefocusVGG, target: torch.device, request: dict[str, Any]) -> dict[str, Any]:
    image = decode_image(request["image"])
    array = np.asarray(image)
    height, width = array.shape[:2]
    pad_h = (32 - height % 32) % 32
    pad_w = (32 - width % 32) % 32
    padded = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    normalized = (padded.astype(np.float32) / 255.0 - MEAN) / STD
    batch = torch.from_numpy(np.ascontiguousarray(normalized)).permute(2, 0, 1).unsqueeze(0).to(target)
    with torch.inference_mode():
        score = torch.sigmoid(model(batch))
    sync(target)
    value = float(score[..., :height, :width].mean().cpu())
    return {
        "score": value,
        "width": width,
        "height": height,
        "inputWidth": int(batch.shape[-1]),
        "inputHeight": int(batch.shape[-2]),
        "device": str(target),
    }


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

        poses.append({"nose": point(0), "torso": [point(11), point(12), point(23), point(24)]})
    return {"poses": poses, "width": width, "height": height}


def run() -> None:
    dino: tuple[Any, Any, torch.device] | None = None
    focus: tuple[DefocusVGG, torch.device] | None = None
    pose: tuple[Any, Any] | None = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("operation")
            if operation == "detect":
                if dino is None:
                    dino = load_dino()
                response = dino_detect(*dino, request)
            elif operation == "focus":
                if focus is None:
                    focus = load_focus()
                response = focus_score(*focus, request)
            elif operation == "pose":
                if pose is None:
                    pose = load_pose()
                response = pose_detect(pose, request)
            else:
                raise ValueError(f"unsupported operation: {operation}")
            print(json.dumps({"ok": True, **response}), flush=True)
        except Exception as error:
            print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}), flush=True)


if __name__ == "__main__":
    run()
