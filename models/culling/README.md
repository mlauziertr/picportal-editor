# PicPortal local culling models

This directory contains model contracts and licenses only; it does not contain
weights. Culling inference is local-only and runs only against already-present
artifacts whose hashes match `manifest.json`. This integration does not download
weights or Python dependencies. Unknown or unavailable signals remain explicitly
unknown.

- Grounding DINO tiny proposes subject boxes from the selected local session
  profile. Dance uses two separate prompts, with the person prompt only if the
  couple prompt returns no boxes.
- MediaPipe Pose Landmarker lite provides local pose points for the existing
  subject/face association rule. It does not assign artistic scores.
- YuNet uses the already-verified face detector in `models/face`; the 0.7
  fallback is attempted only when the primary 0.9 pass returns no face.
- Eye state remains a local heuristic and a review cue. It is never used to
  delete or reject an image. Existing culling categories and explicit unknowns
  are preserved.

The Python worker path is external/user-controlled through
`PICPORTAL_CULLING_PYTHON`. If Python, the sidecar, or verified model files are
not present, the corresponding result stays unavailable/unknown. No inference
request is sent to a remote service.

## Installing the optional subject/pose worker

Without the worker, assisted culling still runs (sharpness, similar photos,
faces and closed eyes). The pre-flight check (`culling_capabilities`) shows
"Subject and pose · Unavailable" before the run and the subject detector is
switched off. This is the supported v1 default.

To enable it (about 700 MB of weights plus the Python packages):

1. Python environment. Use Python 3.11 or 3.12: `mediapipe` and `torch` wheels
   may not exist yet for newer interpreters.

   ```sh
   python3.12 -m venv ~/.local/share/picportal-culling-venv
   ~/.local/share/picportal-culling-venv/bin/pip install torch transformers numpy Pillow mediapipe
   ```

2. Weights. The script checks exact size and SHA-256 against `manifest.json`,
   skips files that are already valid, and `--check` downloads nothing:

   ```sh
   python3 scripts/fetch-culling-models.py --check   # exit 1 while incomplete
   python3 scripts/fetch-culling-models.py           # ~689 MB Grounding DINO + 5.8 MB pose
   ```

   Use `--dest DIR` (or `PICPORTAL_CULLING_MODEL_DIR`) to keep the weights
   outside the repository. The manifest is copied there.

3. Launch the editor with the interpreter (and the model directory, if moved):

   ```sh
   PICPORTAL_CULLING_PYTHON=~/.local/share/picportal-culling-venv/bin/python \
   PICPORTAL_CULLING_MODEL_DIR=/path/to/models npm run tauri -- dev
   ```

The pre-flight result is computed once per app launch, because hashing the
weights is slow. Restart the app after installing.
