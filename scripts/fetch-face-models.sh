#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="$repo_root/models/face"
mkdir -p "$model_dir"

fetch_verified() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"
  local expected_bytes="$4"
  local temporary="${destination}.download"

  rm -f "$temporary"
  curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 \
    "$url" -o "$temporary"
  local actual_bytes
  actual_bytes="$(wc -c < "$temporary" | tr -d ' ')"
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    rm -f "$temporary"
    echo "Size mismatch for $destination: expected $expected_bytes, got $actual_bytes" >&2
    return 1
  fi
  local actual_sha
  if command -v sha256sum >/dev/null 2>&1; then
    actual_sha="$(sha256sum "$temporary" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    actual_sha="$(shasum -a 256 "$temporary" | awk '{print $1}')"
  else
    rm -f "$temporary"
    echo "Neither sha256sum nor shasum is available." >&2
    return 1
  fi
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    rm -f "$temporary"
    echo "SHA-256 mismatch for $destination: expected $expected_sha, got $actual_sha" >&2
    return 1
  fi
  chmod 0644 "$temporary"
  mv -f "$temporary" "$destination"
}

fetch_verified \
  'https://media.githubusercontent.com/media/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx' \
  "$model_dir/face_detection_yunet_2023mar.onnx" \
  '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4' \
  '232589'
fetch_verified \
  'https://media.githubusercontent.com/media/opencv/opencv_zoo/ba91a3b91d00d76e86540d4013f944bd6b514e39/models/face_recognition_sface/face_recognition_sface_2021dec.onnx' \
  "$model_dir/face_recognition_sface_2021dec.onnx" \
  '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79' \
  '38696353'
fetch_verified \
  'https://raw.githubusercontent.com/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE' \
  "$model_dir/YUNET-LICENSE.txt" \
  'c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958' \
  '1085'
fetch_verified \
  'https://raw.githubusercontent.com/opencv/opencv_zoo/ba91a3b91d00d76e86540d4013f944bd6b514e39/models/face_recognition_sface/LICENSE' \
  "$model_dir/SFACE-LICENSE.txt" \
  'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30' \
  '11358'
python3 "$repo_root/scripts/verify-face-models.py" --model-dir "$model_dir"
printf '%s\n' 'Face model artifacts and the tracked CC0 parity fixture verified.'
