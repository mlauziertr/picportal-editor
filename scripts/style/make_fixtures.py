"""Regenerate models/style/fixtures (Python <-> Rust parity fixtures).

    python3 -m scripts.style.make_fixtures [--out models/style/fixtures]

Produces small procedural images (PNG, JPEG, and a minimal TIFF-based `.dng` holding only an
embedded JPEG preview plus EXIF), a Lightroom XMP with its expected conversion, a model trained
on a synthetic corpus, and `parity.expected.json` with the Python features and ONNX outputs that
`src-tauri/src/style_model.rs` must reproduce.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from . import synth, train
from .features import features_for_path
from .targets import adjustments_from_xmp

LIGHTROOM_XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    crs:ProcessVersion="15.4"
    crs:WhiteBalance="Custom"
    crs:AsShotTemperature="5200"
    crs:Temperature="5800"
    crs:Tint="+12"
    crs:Exposure2012="+0.35"
    crs:Contrast2012="+18"
    crs:Highlights2012="-42"
    crs:Shadows2012="+30"
    crs:Whites2012="+9"
    crs:Blacks2012="-14"
    crs:Vibrance="+22"
    crs:Saturation="-6">
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def _scene(seed: int, height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return synth.srgb_encode(synth.render_scene("landscape", rng, height, width) * 0.8)


def _exif(iso: int, exposure_denominator: int, focal: int, orientation: int = 1) -> Image.Exif:
    exif = Image.Exif()
    exif[0x0112] = orientation
    sub = exif.get_ifd(0x8769)
    sub[0x8827] = iso
    sub[0x829A] = IFDRational(1, exposure_denominator)
    sub[0x920A] = IFDRational(focal, 1)
    return exif


def _tiff_with_preview(jpeg: bytes, iso: int, exposure_denominator: int, focal: int, orientation: int) -> bytes:
    """Little-endian TIFF: IFD0 (JPEG strip, orientation, Exif pointer) + Exif IFD."""
    ifd0_entries = 6
    ifd0_offset = 8
    ifd0_size = 2 + ifd0_entries * 12 + 4
    exif_offset = ifd0_offset + ifd0_size
    exif_entries = 3
    exif_size = 2 + exif_entries * 12 + 4
    rationals_offset = exif_offset + exif_size
    jpeg_offset = rationals_offset + 16

    def entry(tag: int, kind: int, count: int, value: int) -> bytes:
        if kind == 3 and count == 1:
            return struct.pack("<HHIHH", tag, kind, count, value, 0)
        return struct.pack("<HHII", tag, kind, count, value)

    ifd0 = struct.pack("<H", ifd0_entries) + b"".join(
        [
            entry(0x00FE, 4, 1, 1),  # NewSubfileType: reduced-resolution preview
            entry(0x0103, 3, 1, 7),  # Compression: JPEG
            entry(0x0111, 4, 1, jpeg_offset),  # StripOffsets
            entry(0x0112, 3, 1, orientation),
            entry(0x0117, 4, 1, len(jpeg)),  # StripByteCounts
            entry(0x8769, 4, 1, exif_offset),
        ]
    ) + struct.pack("<I", 0)
    exif = struct.pack("<H", exif_entries) + b"".join(
        [
            entry(0x829A, 5, 1, rationals_offset),  # ExposureTime
            entry(0x8827, 3, 1, iso),
            entry(0x920A, 5, 1, rationals_offset + 8),  # FocalLength
        ]
    ) + struct.pack("<I", 0)
    rationals = struct.pack("<IIII", 1, exposure_denominator, focal, 1)
    return b"II*\x00" + struct.pack("<I", ifd0_offset) + ifd0 + exif + rationals + jpeg


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    images = []

    Image.fromarray(_scene(11, 150, 230)).save(out / "parity.png", exif=_exif(200, 500, 24))
    images.append(("parity.png", 1e-5))

    Image.fromarray(_scene(12, 300, 200)).save(out / "parity.jpg", quality=92, exif=_exif(3200, 30, 85, orientation=6))
    images.append(("parity.jpg", 4e-3))  # Pillow/libjpeg vs zune-jpeg decoding differ slightly

    preview = io.BytesIO()
    Image.fromarray(_scene(13, 160, 240)).save(preview, format="JPEG", quality=90)
    (out / "parity-preview.dng").write_bytes(_tiff_with_preview(preview.getvalue(), 800, 125, 50, orientation=8))
    images.append(("parity-preview.dng", 4e-3))

    (out / "lightroom-basic.xmp").write_text(LIGHTROOM_XMP)
    expected = adjustments_from_xmp(LIGHTROOM_XMP)
    (out / "lightroom-basic.expected.json").write_text(json.dumps(expected, indent=2) + "\n")

    with tempfile.TemporaryDirectory() as scratch:
        corpus = Path(scratch) / "corpus"
        synth.generate(corpus, count=120, seed=3)
        model_dir = out / "synthetic-model"
        shutil.rmtree(model_dir, ignore_errors=True)
        dataset = train.extract.build_dataset([corpus], workers=1)
        model, report = train.evaluate(dataset, seed=0, truth_path=corpus / "synthetic-truth.json")
        train.export(model, report, dataset, model_dir, seed=0)

    import onnxruntime  # only needed to record the reference ONNX outputs

    session = onnxruntime.InferenceSession(str(model_dir / train.MODEL_FILENAME))
    entries = []
    for name, tolerance in images:
        features = features_for_path(out / name)
        prediction = session.run(None, {"features": features[None].astype(np.float32)})[0][0]
        entries.append(
            {
                "file": name,
                "tolerance": tolerance,
                "features": [float(value) for value in features],
                "prediction": [float(value) for value in prediction],
            }
        )
    (out / "parity.expected.json").write_text(json.dumps({"images": entries}, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("models/style/fixtures"))
    build(parser.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
