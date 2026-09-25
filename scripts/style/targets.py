"""Training targets: the editor's `.rrdata` sidecars or Lightroom XMP develop settings.

The XMP mapping is the subset of `convert_xmp_to_preset` (src-tauri/src/preset_converter.rs)
covering the v1 keys; the shared fixture models/style/fixtures/lightroom-basic.xmp is checked
against the same expected values on both sides.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import OUTPUT_KEYS

_DIRECT = {
    "Exposure2012": "exposure",
    "Contrast2012": "contrast",
    "Highlights2012": "highlights",
    "Whites2012": "whites",
    "Blacks2012": "blacks",
    "Vibrance": "vibrance",
    "Saturation": "saturation",
}
_DEVELOP_MARKERS = ("ProcessVersion", "Exposure2012", "Contrast2012", "Temperature", "Tint")
_ATTRIBUTE = re.compile(r'crs:([A-Za-z0-9]+)="([^"]*)"')
_ELEMENT = re.compile(r"<crs:([A-Za-z0-9]+)>([^<]*)</crs:\1>")


def _number(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw.strip().lstrip("+"))
    except ValueError:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def xmp_attributes(content: str) -> dict[str, str]:
    one_line = " ".join(content.split("\n"))
    attributes = {name: value for name, value in _ELEMENT.findall(one_line)}
    # Attribute form wins, as in preset_converter.rs (which only reads attributes).
    attributes.update({name: value for name, value in _ATTRIBUTE.findall(one_line)})
    return attributes


def adjustments_from_xmp(content: str) -> dict[str, float] | None:
    """Return the v1 adjustments of a Lightroom XMP, or None if it carries no develop settings."""
    attributes = xmp_attributes(content)
    if not any(marker in attributes for marker in _DEVELOP_MARKERS):
        return None
    result = {key: 0.0 for key in OUTPUT_KEYS}
    for xmp_key, key in _DIRECT.items():
        value = _number(attributes.get(xmp_key))
        if value is not None:
            result[key] = value
    shadows = _number(attributes.get("Shadows2012"))
    if shadows is not None:
        result["shadows"] = min(shadows * 1.5, 100.0)
    temperature = _number(attributes.get("Temperature"))
    if temperature:
        as_shot = _number(attributes.get("AsShotTemperature")) or 5500.0
        mired_delta = 1_000_000.0 / temperature - 1_000_000.0 / as_shot
        result["temperature"] = _clamp(-mired_delta / 150.0 * 100.0, -100.0, 100.0)
    tint = _number(attributes.get("Tint"))
    if tint is not None:
        result["tint"] = _clamp(tint / 150.0 * 100.0, -100.0, 100.0)
    return result


def adjustments_from_rrdata(content: str) -> dict[str, float] | None:
    """Return the v1 adjustments of an editor sidecar, or None if no v1 slider was moved."""
    try:
        adjustments = json.loads(content).get("adjustments")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(adjustments, dict):
        return None
    result = {}
    for key in OUTPUT_KEYS:
        value = adjustments.get(key, 0.0)
        result[key] = float(value) if isinstance(value, (int, float)) else 0.0
    return result if any(result.values()) else None


def sidecar_paths(image: Path) -> tuple[Path, list[Path]]:
    """`.rrdata` path (same rule as parse_virtual_path) and XMP candidates (resolve_xmp_path)."""
    return image.with_name(image.name + ".rrdata"), [image.with_suffix(".xmp"), image.with_suffix(".XMP")]


def read_target(image: Path) -> tuple[dict[str, float], str] | None:
    rrdata, xmp_candidates = sidecar_paths(image)
    if rrdata.is_file():
        target = adjustments_from_rrdata(rrdata.read_text(encoding="utf-8", errors="replace"))
        if target is not None:
            return target, "rrdata"
    for xmp in xmp_candidates:
        if xmp.is_file():
            target = adjustments_from_xmp(xmp.read_text(encoding="utf-8", errors="replace"))
            if target is not None:
                return target, "xmp"
    return None
