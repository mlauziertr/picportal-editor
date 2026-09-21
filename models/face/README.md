# Canonical face model

PicPortal uses one versioned embedding space in this editor and on the server:
OpenCV YuNet 2023mar for detection and OpenCV SFace 2021dec for 128-dimensional
embeddings. `manifest.json` is the machine-readable source of truth for model
identity, preprocessing constants, immutable upstream revisions, byte sizes and
SHA-256 values.

The upstream model directories explicitly license **all files in each
directory**, including the ONNX weights: YuNet under MIT and SFace under
Apache-2.0. The fetch script also downloads the exact license texts and verifies
their checksums. This is deliberately different from InsightFace model-zoo
weights, which are not used or downloaded by this project.

The two ONNX files and PicPortal's internally generated CC0 parity fixture are intentionally committed so a
production app bundle can be built without downloading executable model data
at build time. Their combined size is about 39 MB, below GitHub's per-file
limit. `scripts/verify-face-models.py --require-tracked` proves that every
artifact is tracked and agrees with this manifest.

The parity portrait is fictional and was generated for this repository without
an input image. Its exact prompt, generation note, CC0 dedication, image, and
expected JSON are tracked and individually hashed by `manifest.json`. No Lena
or other third-party sample photograph is used.

`scripts/fetch-face-models.sh` remains the reproducible restoration path. It
downloads only immutable HTTPS URLs and replaces a destination only after size
and SHA-256 validation on macOS or Linux. The Tauri bundle copies the reviewed,
committed ONNX weights, licenses, and parity fixture as application resources.

The SFace threshold follows OpenCV's published same-identity cosine similarity
threshold of `0.363`; PicPortal stores cosine distance, therefore the threshold
is `1 - 0.363 = 0.637`. Those constants live in `manifest.json`.

Rust pins `ort` 2.0.0-rc.10 with dynamically loaded ONNX Runtime 1.22.0.
The manifest also records Python ONNX Runtime 1.24.4 and a `0.999`
cosine-parity floor; the local Rust fixture test enforces that floor.
