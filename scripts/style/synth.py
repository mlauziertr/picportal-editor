"""Synthetic validation corpus: procedural photos, a known per-scene preset, capture errors.

    python3 -m scripts.style.synth --out <dir> [--count 400] [--seed 7]

Each image is rendered from a procedural scene (no downloaded image, no licence issue), then
"shot" with a random exposure error and white-balance cast. Its edit is the preset of its scene
type (scripts/style/fixtures/synthetic-presets.json) plus the correction of those errors plus
a little human noise. Half of the edits are written as `.rrdata`, half as Lightroom XMP, and a
few images are left unedited. `synthetic-truth.json` records what the model should recover.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from .schema import OUTPUT_KEYS

PRESETS_PATH = Path(__file__).with_name("fixtures") / "synthetic-presets.json"
STYLE_ONLY_KEYS = ["contrast", "highlights", "shadows", "whites", "blacks", "vibrance", "saturation"]
EXIF_BY_SCENE = {  # (ISO range, exposure time range in s, focal range in mm)
    "landscape": ((100, 200), (1 / 1000, 1 / 250), (16, 35)),
    "portrait": ((200, 800), (1 / 250, 1 / 125), (50, 135)),
    "night": ((1600, 6400), (1 / 60, 1 / 15), (24, 50)),
    "interior": ((800, 3200), (1 / 125, 1 / 30), (24, 50)),
}


def load_presets() -> dict[str, dict[str, float]]:
    return json.loads(PRESETS_PATH.read_text())["presets"]


def _coordinates(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    return np.mgrid[0:height, 0:width].astype(np.float64) / np.array([height, width])[:, None, None]


def _ellipse(y, x, cy, cx, ry, rx) -> np.ndarray:
    return (((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2) <= 1.0


def _texture(rng, height, width, scale=0.15) -> np.ndarray:
    coarse = rng.normal(0, 1, (height // 16 + 2, width // 16 + 2))
    image = Image.fromarray(((coarse - coarse.min()) / (np.ptp(coarse) + 1e-9) * 255).astype(np.uint8))
    smooth = np.asarray(image.resize((width, height), Image.BICUBIC), dtype=np.float64) / 255.0 - 0.5
    return 1.0 + scale * (smooth + 0.3 * rng.normal(0, 1, (height, width)))


def render_scene(kind: str, rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Linear-light RGB in [0, ~1.5]."""
    y, x = _coordinates(height, width)
    image = np.zeros((height, width, 3))
    jitter = lambda color, amount=0.15: np.clip(np.array(color) * rng.uniform(1 - amount, 1 + amount, 3), 0, 2)  # noqa: E731
    if kind == "landscape":
        horizon = rng.uniform(0.35, 0.65)
        top, bottom = jitter([0.18, 0.35, 0.8]), jitter([0.6, 0.72, 0.9])
        sky = top + (bottom - top) * (y / horizon)[..., None]
        ground = jitter(rng.choice([[0.12, 0.25, 0.06], [0.3, 0.22, 0.1], [0.35, 0.33, 0.28]]))
        image = np.where((y < horizon)[..., None], sky, ground * _texture(rng, height, width, 0.4)[..., None])
        if rng.random() < 0.5:
            sun = _ellipse(y, x, rng.uniform(0.05, horizon - 0.1), rng.uniform(0.1, 0.9), 0.05, 0.05 * height / width)
            image[sun] = [1.5, 1.4, 1.2]
    elif kind == "portrait":
        background = jitter(rng.choice([[0.3, 0.3, 0.32], [0.15, 0.25, 0.12], [0.45, 0.4, 0.35]]))
        image = background * _texture(rng, height, width, 0.3)[..., None]
        body = _ellipse(y, x, 0.95, 0.5, 0.35, 0.3)
        image[body] = jitter(rng.choice([[0.05, 0.05, 0.08], [0.5, 0.1, 0.1], [0.6, 0.6, 0.62]]))
        face = _ellipse(y, x, rng.uniform(0.35, 0.45), rng.uniform(0.42, 0.58), 0.2, 0.2 * height / width)
        image[face] = jitter(rng.choice([[0.62, 0.42, 0.32], [0.38, 0.24, 0.16], [0.75, 0.55, 0.45]]), 0.08)
    elif kind == "night":
        image = jitter([0.01, 0.015, 0.04], 0.5) * (1.3 - y)[..., None] * _texture(rng, height, width, 0.2)[..., None]
        for _ in range(rng.integers(3, 9)):
            lamp = _ellipse(y, x, rng.uniform(0.2, 0.8), rng.uniform(0.05, 0.95), 0.02, 0.02 * height / width)
            glow = np.exp(-(((y - y[lamp].mean()) ** 2 + (x - x[lamp].mean()) ** 2) / 0.01)) if lamp.any() else 0
            image += (glow * 0.25)[..., None] * jitter([1.0, 0.6, 0.25])
            image[lamp] = jitter([1.4, 1.0, 0.5])
    else:  # interior
        wall = jitter([0.35, 0.25, 0.15])
        image = wall * (0.6 + 0.4 * x)[..., None] * _texture(rng, height, width, 0.15)[..., None]
        window = (np.abs(x - rng.uniform(0.6, 0.85)) < 0.12) & (np.abs(y - 0.35) < 0.2)
        image[window] = [1.2, 1.25, 1.35]
        floor = y > rng.uniform(0.7, 0.8)
        image[floor] = jitter([0.2, 0.12, 0.06]) * _texture(rng, height, width, 0.3)[floor][:, None]
        for _ in range(rng.integers(1, 4)):
            top, left = rng.uniform(0.4, 0.7), rng.uniform(0.0, 0.6)
            block = (y > top) & (y < top + 0.25) & (x > left) & (x < left + 0.2)
            image[block] = jitter(rng.choice([[0.3, 0.1, 0.1], [0.1, 0.15, 0.3], [0.4, 0.35, 0.2]]))
    return np.clip(image, 0, None)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0, 1)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.round(encoded * 255).astype(np.uint8)


def _exif(kind: str, rng: np.random.Generator) -> Image.Exif:
    (iso_low, iso_high), (t_low, t_high), (f_low, f_high) = EXIF_BY_SCENE[kind]
    exif = Image.Exif()
    sub = exif.get_ifd(0x8769)
    sub[0x8827] = int(round(math.exp(rng.uniform(math.log(iso_low), math.log(iso_high)))))
    exposure_time = math.exp(rng.uniform(math.log(t_low), math.log(t_high)))
    sub[0x829A] = IFDRational(1, int(round(1 / exposure_time)))
    sub[0x920A] = IFDRational(int(round(rng.uniform(f_low, f_high))), 1)
    return exif


def _xmp(adjustments: dict[str, float]) -> str:
    """Inverse of the converter mapping (targets.adjustments_from_xmp / preset_converter.rs)."""
    as_shot = 5500
    temperature = 1_000_000.0 / (1_000_000.0 / as_shot - adjustments["temperature"] * 1.5)
    attributes = {
        "ProcessVersion": "15.4",
        "WhiteBalance": "Custom",
        "AsShotTemperature": str(as_shot),
        "Temperature": str(int(round(temperature))),
        "Tint": f"{round(adjustments['tint'] * 1.5):+d}",
        "Exposure2012": f"{adjustments['exposure']:+.2f}",
        "Contrast2012": f"{round(adjustments['contrast']):+d}",
        "Highlights2012": f"{round(adjustments['highlights']):+d}",
        "Shadows2012": f"{round(adjustments['shadows'] / 1.5):+d}",
        "Whites2012": f"{round(adjustments['whites']):+d}",
        "Blacks2012": f"{round(adjustments['blacks']):+d}",
        "Vibrance": f"{round(adjustments['vibrance']):+d}",
        "Saturation": f"{round(adjustments['saturation']):+d}",
    }
    body = "\n".join(f'    crs:{key}="{value}"' for key, value in attributes.items())
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"\n'
        f"{body}>\n  </rdf:Description>\n </rdf:RDF>\n</x:xmpmeta>\n"
    )


def generate(out: Path, count: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    presets = load_presets()
    kinds = sorted(presets)
    out.mkdir(parents=True, exist_ok=True)
    truth = {"seed": seed, "presets": presets, "styleOnlyKeys": STYLE_ONLY_KEYS, "images": {}}
    for index in range(count):
        kind = kinds[index % len(kinds)]
        landscape_frame = rng.random() < 0.7
        height, width = (256, 384) if landscape_frame else (384, 256)
        linear = render_scene(kind, rng, height, width)
        exposure_error = float(rng.uniform(-1.2, 1.2))
        warm_cast, green_cast = float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
        gains = np.array([1 + 0.25 * warm_cast, 1 + 0.12 * green_cast, 1 - 0.25 * warm_cast])
        shot = linear * (2.0**exposure_error) * gains
        name = f"{kind}-{index:04d}.jpg"
        Image.fromarray(srgb_encode(shot)).save(out / name, quality=90, exif=_exif(kind, rng))
        preset = presets[kind]
        edit = {key: preset[key] + rng.normal(0, 3.0) for key in OUTPUT_KEYS}
        edit["exposure"] = round(preset["exposure"] - exposure_error + rng.normal(0, 0.08), 2)
        edit["temperature"] = preset["temperature"] - 40 * warm_cast + rng.normal(0, 3.0)
        edit["tint"] = preset["tint"] + 30 * green_cast + rng.normal(0, 3.0)
        edit = {key: (value if key == "exposure" else float(round(value))) for key, value in edit.items()}
        record = {"preset": kind, "exposureError": round(exposure_error, 3), "warmCast": round(warm_cast, 3)}
        draw = rng.random()
        if draw < 0.05:
            record["edited"] = False  # left unedited: must be skipped by extraction
        elif draw < 0.525:
            sidecar = {"version": 1, "rating": 0, "adjustments": edit}
            (out / f"{name}.rrdata").write_text(json.dumps(sidecar, indent=2))
            record["sidecar"] = "rrdata"
        else:
            (out / name).with_suffix(".xmp").write_text(_xmp(edit))
            record["sidecar"] = "xmp"
        truth["images"][name] = record
    (out / "synthetic-truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    truth = generate(args.out, args.count, args.seed)
    edited = sum(1 for record in truth["images"].values() if record.get("edited", True))
    print(f"wrote {len(truth['images'])} images ({edited} edited) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
