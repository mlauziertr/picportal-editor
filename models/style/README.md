# Style model artifacts

`python3 -m scripts.style.train --corpus <dir>` writes the personal style model here by default
(`style-model.onnx`, `manifest.json` with its SHA-256, `report.json`). These files are ignored by
git: they are trained from private photos. See `docs/style-model.md`.

`fixtures/` holds the Python ↔ Rust parity fixtures (procedural images, a Lightroom XMP and a
model trained on the synthetic corpus), regenerated with `python3 -m scripts.style.make_fixtures`.
