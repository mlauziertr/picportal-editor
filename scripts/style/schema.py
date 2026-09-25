"""Shared contract between the Python trainer and the Rust runtime (src-tauri/src/style_model.rs).

Any change here must bump FEATURE_VERSION and be mirrored in Rust; the parity fixture in
models/style/fixtures guards it.
"""

from __future__ import annotations

FEATURE_VERSION = "style-features-v1"
MANIFEST_VERSION = 1
PREVIEW_SIZE = 64
GRID = 4
MIN_TRAINING_IMAGES = 50

# Keys of the `Adjustments` type (src/utils/adjustments.ts) predicted by v1, with slider ranges.
OUTPUT_KEYS = (
    "exposure",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "temperature",
    "tint",
    "vibrance",
    "saturation",
)
OUTPUT_RANGES = {key: (-100.0, 100.0) for key in OUTPUT_KEYS}
OUTPUT_RANGES["exposure"] = (-5.0, 5.0)

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def _feature_names() -> tuple[str, ...]:
    names = ["mean_r", "mean_g", "mean_b", "mean_y", "std_y"]
    names += [f"y_p{p}" for p in PERCENTILES]
    names += ["clip_high", "clip_low", "warmth", "tint_axis", "saturation"]
    names += [f"grid_y_{row}{col}" for row in range(GRID) for col in range(GRID)]
    names += [f"grid_warmth_{row}{col}" for row in range(GRID) for col in range(GRID)]
    names += ["exif_present", "log2_iso", "log2_exposure_time", "log2_focal_length"]
    return tuple(names)


FEATURE_NAMES = _feature_names()
FEATURE_COUNT = len(FEATURE_NAMES)


def clamp_adjustments(values: dict[str, float]) -> dict[str, float]:
    """Clamp to slider ranges and round like the UI sliders (exposure 0.01, others 1)."""
    result = {}
    for key in OUTPUT_KEYS:
        low, high = OUTPUT_RANGES[key]
        value = min(max(float(values[key]), low), high)
        result[key] = round(value, 2) if key == "exposure" else float(round(value))
    return result
