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
