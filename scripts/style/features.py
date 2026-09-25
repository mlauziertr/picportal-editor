"""Image features for the style model (`style-features-v1`).

Mirrors `src-tauri/src/style_model.rs`. The features are computed on a display-referred sRGB
preview: the decoded file for JPEG/PNG/TIFF, the largest embedded JPEG preview for RAW files.
"""

from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .schema import FEATURE_COUNT, GRID, PERCENTILES, PREVIEW_SIZE

# Copy of `RAW_EXTENSIONS` in src-tauri/src/formats.rs (a test keeps both in sync).
RAW_EXTENSIONS = frozenset(
    """3fr ari arw bay cr2 cr3 crw dcr dcs dng erf fff iiq k25 kdc mef mos mrw nef nrw orf pef pro
    ptx raf raw rw2 rwl sr2 srf srw x3f""".split()
)
DECODED_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "tif", "tiff", "webp"})

EXIF_IFD_POINTER = 0x8769
TAG_ISO = 0x8827
TAG_EXPOSURE_TIME = 0x829A
TAG_FOCAL_LENGTH = 0x920A
TAG_ORIENTATION = 0x0112


@dataclass(frozen=True)
class Exif:
    iso: float | None = None
    exposure_time: float | None = None
    focal_length: float | None = None


def extension(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def is_raw(path: Path) -> bool:
    return extension(path) in RAW_EXTENSIONS


def is_supported(path: Path) -> bool:
    return not path.name.startswith(".") and (is_raw(path) or extension(path) in DECODED_EXTENSIONS)


# --- TIFF helpers (RAW previews and RAW EXIF), same walk as `largest_tiff_jpeg_preview` -----


class _Tiff:
    def __init__(self, data: bytes):
        head = data[:4]
        if head == b"II*\x00":
            self.order = "<"
        elif head == b"MM\x00*":
            self.order = ">"
        else:
            raise ValueError("not a TIFF container")
        self.data = data

    def u16(self, offset: int) -> int | None:
        if offset < 0 or offset + 2 > len(self.data):
            return None
        return struct.unpack_from(self.order + "H", self.data, offset)[0]

    def u32(self, offset: int) -> int | None:
        if offset < 0 or offset + 4 > len(self.data):
            return None
        return struct.unpack_from(self.order + "I", self.data, offset)[0]

    def entries(self, ifd: int):
        count = self.u16(ifd)
        if count is None:
            return
        for index in range(count):
            entry = ifd + 2 + index * 12
            tag, kind, n, value = self.u16(entry), self.u16(entry + 2), self.u32(entry + 4), self.u32(entry + 8)
            if None in (tag, kind, n, value):
                continue
            yield tag, kind, n, value, entry + 8

    def next_ifd(self, ifd: int) -> int | None:
        count = self.u16(ifd)
        return None if count is None else self.u32(ifd + 2 + count * 12)

    def rational(self, offset: int) -> float | None:
        numerator, denominator = self.u32(offset), self.u32(offset + 4)
        if not denominator or numerator is None:
            return None
        return numerator / denominator


def _tiff_jpeg_candidates(data: bytes) -> list[tuple[int, int]]:
    tiff = _Tiff(data)
    candidates: list[tuple[int, int]] = []
    first = tiff.u32(4)
    queue = [first] if first else []
    seen: set[int] = set()
    while queue:
        ifd = queue.pop()
        if ifd in seen or len(seen) >= 64:
            continue
        seen.add(ifd)
        compression = 0
        strip = old_jpeg = None
        for tag, _kind, count, value, _ in tiff.entries(ifd):
            if tag == 259:
                compression = value
            elif tag == 273 and count == 1:
                strip = (value, strip[1] if strip else 0)
            elif tag == 279 and count == 1:
                strip = (strip[0], value) if strip else (0, value)
            elif tag == 513:
                old_jpeg = (value, old_jpeg[1] if old_jpeg else 0)
            elif tag == 514:
                old_jpeg = (old_jpeg[0], value) if old_jpeg else (0, value)
            elif tag == 330:
                if count == 1:
                    queue.append(value)
                else:
                    for j in range(min(count, 8)):
                        pointer = tiff.u32(value + j * 4)
                        if pointer is not None:
                            queue.append(pointer)
        if compression in (6, 7) and strip:
            candidates.append(strip)
        if old_jpeg:
            candidates.append(old_jpeg)
        following = tiff.next_ifd(ifd)
        if following:
            queue.append(following)
    return sorted(candidates, key=lambda item: -item[1])


def _scan_jpeg_candidates(data: bytes) -> list[tuple[int, int]]:
    """Fallback for non-TIFF RAW containers (CR3, RAF...): embedded JPEG streams by size."""
    candidates = []
    start = data.find(b"\xff\xd8\xff")
    while start != -1:
        end = data.find(b"\xff\xd9", start + 3)
        if end == -1:
            break
        candidates.append((start, end + 2 - start))
        start = data.find(b"\xff\xd8\xff", start + 3)
    return sorted(candidates, key=lambda item: -item[1])


def raw_preview(data: bytes) -> Image.Image:
    try:
        candidates = _tiff_jpeg_candidates(data)
    except ValueError:
        candidates = []
    candidates += _scan_jpeg_candidates(data)
    for offset, length in candidates:
        try:
            image = Image.open(io.BytesIO(data[offset : offset + length]))
            image.load()
            return image
        except Exception:
            continue
    raise ValueError("no decodable embedded JPEG preview")


def _raw_exif(data: bytes) -> tuple[Exif, int | None]:
    try:
        tiff = _Tiff(data)
    except ValueError:
        return Exif(), None
    ifd0 = tiff.u32(4)
    if not ifd0:
        return Exif(), None
    orientation = None
    exif_ifd = None
    for tag, _kind, _count, value, value_offset in tiff.entries(ifd0):
        if tag == TAG_ORIENTATION:
            orientation = tiff.u16(value_offset)
        elif tag == EXIF_IFD_POINTER:
            exif_ifd = value
    values: dict[int, float | None] = {}
    if exif_ifd:
        for tag, kind, _count, value, value_offset in tiff.entries(exif_ifd):
            if tag == TAG_ISO:
                values[tag] = float(tiff.u16(value_offset) if kind == 3 else value)
            elif tag in (TAG_EXPOSURE_TIME, TAG_FOCAL_LENGTH) and kind == 5:
                values[tag] = tiff.rational(value)
    return Exif(values.get(TAG_ISO), values.get(TAG_EXPOSURE_TIME), values.get(TAG_FOCAL_LENGTH)), orientation


def _pillow_exif(image: Image.Image) -> Exif:
    exif = image.getexif()
    sub = exif.get_ifd(EXIF_IFD_POINTER)

    def number(tag: int) -> float | None:
        value = sub.get(tag, exif.get(tag))
        if isinstance(value, tuple):
            value = value[0] if value else None
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    return Exif(number(TAG_ISO), number(TAG_EXPOSURE_TIME), number(TAG_FOCAL_LENGTH))


_ORIENTATION_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def load_source(path: Path) -> tuple[np.ndarray, Exif]:
    """Decode `path` into an oriented uint8 RGB array plus the EXIF fields used as features."""
    data = path.read_bytes()
    if is_raw(path):
        exif, orientation = _raw_exif(data)
        image = raw_preview(data)
        if orientation in _ORIENTATION_TRANSPOSE:
            image = image.transpose(_ORIENTATION_TRANSPOSE[orientation])
    else:
        image = Image.open(io.BytesIO(data))
        exif = _pillow_exif(image)
        image = ImageOps.exif_transpose(image)
    return np.asarray(image.convert("RGB"), dtype=np.uint8), exif


# --- Preview and features ------------------------------------------------------------------


def _bounds(length: int, cells: int) -> np.ndarray:
    starts = np.array([(i * length) // cells for i in range(cells)], dtype=np.int64)
    return np.minimum(starts, length - 1)


def box_preview(rgb: np.ndarray, size: int = PREVIEW_SIZE) -> np.ndarray:
    """Average `rgb` (H, W, 3 uint8) over a size x size grid of integer-bounded blocks.

    Block i spans [floor(i*L/size), max(start+1, floor((i+1)*L/size))). Returns float64 in [0, 1].
    """
    height, width = rgb.shape[:2]
    rows, cols = _bounds(height, size), _bounds(width, size)
    row_ends = np.maximum(np.append(rows[1:], height), rows + 1)
    col_ends = np.maximum(np.append(cols[1:], width), cols + 1)
    preview = np.empty((size, size, 3), dtype=np.float64)
    for i in range(size):
        band = rgb[rows[i] : row_ends[i]].sum(axis=0, dtype=np.uint64)
        cumulative = np.concatenate([np.zeros((1, 3), dtype=np.uint64), np.cumsum(band, axis=0)])
        sums = cumulative[col_ends] - cumulative[cols]
        counts = (row_ends[i] - rows[i]) * (col_ends - cols)
        preview[i] = sums / counts[:, None]
    return preview / 255.0


def _log2(value: float | None, default: float) -> float:
    return math.log2(value) if value and value > 0 else math.log2(default)


def compute_features(preview: np.ndarray, exif: Exif) -> np.ndarray:
    red, green, blue = preview[..., 0], preview[..., 1], preview[..., 2]
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    ordered = np.sort(luma.ravel())
    last = ordered.size - 1
    percentiles = [ordered[int(math.floor(p / 100.0 * last + 0.5))] for p in PERCENTILES]
    warmth = red - blue
    tint = green - (red + blue) / 2.0
    saturation = preview.max(axis=2) - preview.min(axis=2)
    cell = preview.shape[0] // GRID
    grid_luma = luma.reshape(GRID, cell, GRID, cell).mean(axis=(1, 3)).ravel()
    grid_warmth = warmth.reshape(GRID, cell, GRID, cell).mean(axis=(1, 3)).ravel()
    exif_present = float(None not in (exif.iso, exif.exposure_time, exif.focal_length))
    values = [
        red.mean(),
        green.mean(),
        blue.mean(),
        luma.mean(),
        luma.std(),
        *percentiles,
        float((luma >= 0.98).mean()),
        float((luma <= 0.02).mean()),
        warmth.mean(),
        tint.mean(),
        saturation.mean(),
        *grid_luma,
        *grid_warmth,
        exif_present,
        _log2(exif.iso, 100.0) - math.log2(100.0),
        _log2(exif.exposure_time, 1.0 / 125.0),
        _log2(exif.focal_length, 50.0),
    ]
    features = np.asarray(values, dtype=np.float64)
    assert features.shape == (FEATURE_COUNT,)
    return features


def features_for_path(path: Path) -> np.ndarray:
    rgb, exif = load_source(path)
    return compute_features(box_preview(rgb), exif)
