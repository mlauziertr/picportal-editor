#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
face_dir="$repo_root/models/face"
culling_dir="$repo_root/models/culling"
dino_dir="$culling_dir/dino"
mkdir -p "$face_dir" "$dino_dir"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

fetch_verified() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"
  local expected_bytes="$4"
  local temporary="${destination}.download"

  mkdir -p "$(dirname "$destination")"
  curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 "$url" -o "$temporary"
  local actual_bytes
  actual_bytes="$(wc -c < "$temporary" | tr -d ' ')"
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    rm -f "$temporary"
    printf 'Size mismatch for %s: expected %s, got %s\n' "$destination" "$expected_bytes" "$actual_bytes" >&2
    return 1
  fi
  local actual_sha
  actual_sha="$(sha256 "$temporary")"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    rm -f "$temporary"
    printf 'SHA-256 mismatch for %s: expected %s, got %s\n' "$destination" "$expected_sha" "$actual_sha" >&2
    return 1
  fi
  chmod 0644 "$temporary"
  mv -f "$temporary" "$destination"
}

revision='a2bb814dd30d776dcf7e30523b00659f4f141c71'
base_url="https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/${revision}"

fetch_verified \
  "${base_url}/model.safetensors?download=true" \
  "$dino_dir/model.safetensors" \
  '1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3' \
  '689359096'
fetch_verified \
  "${base_url}/tokenizer.json?download=true" \
  "$dino_dir/tokenizer.json" \
  'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66' \
  '711396'
fetch_verified \
  "${base_url}/vocab.txt?download=true" \
  "$dino_dir/vocab.txt" \
  '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3' \
  '231508'
fetch_verified \
  "${base_url}/config.json?download=true" \
  "$dino_dir/config.json" \
  'eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a' \
  '1644'
fetch_verified \
  "${base_url}/preprocessor_config.json?download=true" \
  "$dino_dir/preprocessor_config.json" \
  '8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60' \
  '457'
fetch_verified \
  "${base_url}/tokenizer_config.json?download=true" \
  "$dino_dir/tokenizer_config.json" \
  'd40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56' \
  '1237'
fetch_verified \
  "${base_url}/special_tokens_map.json?download=true" \
  "$dino_dir/special_tokens_map.json" \
  'b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3' \
  '125'
fetch_verified \
  "${base_url}/added_tokens.json?download=true" \
  "$dino_dir/added_tokens.json" \
  '909e96cb32d92ce728a01bc99850cbba26196d74115c17ebeb019275412588f2' \
  '82'
fetch_verified \
  'https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/LICENSE' \
  "$culling_dir/DINO-LICENSE.txt" \
  'b403c98ec1ffaccb0124632592a3d99642cddc3031ca412b402568e6c06c202a' \
  '11355'

fetch_verified \
  'https://media.githubusercontent.com/media/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx' \
  "$face_dir/face_detection_yunet_2023mar.onnx" \
  '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4' \
  '232589'
fetch_verified \
  'https://raw.githubusercontent.com/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE' \
  "$face_dir/YUNET-LICENSE.txt" \
  'c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958' \
  '1085'

fetch_verified \
  'https://raw.githubusercontent.com/vinthony/depth-distillation/master/LICENSE' \
  "$culling_dir/VGG-LICENSE.txt" \
  'fca4fb2b7af053895c597466a6dcfc7b40c38924cfc679ebd337d9c7a52ce8f9' \
  '1067'

fetch_verified \
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task' \
  "$culling_dir/pose_landmarker_lite.task" \
  '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a' \
  '5777746'

if [[ "${1:-}" == "--with-vgg" ]]; then
  : "${VGG_SOURCE_URL:?Set VGG_SOURCE_URL to the reviewed checkpoint source}"
  : "${VGG_SHA256:?Set VGG_SHA256 to the independently verified checkpoint digest}"
  fetch_verified "$VGG_SOURCE_URL" "$culling_dir/vgg_best.pth" "$VGG_SHA256" '77824617'
  printf '%s  %s\n' "$VGG_SHA256" "$culling_dir/vgg_best.pth.sha256"
  printf '%s\n' "$VGG_SHA256" > "$culling_dir/vgg_best.pth.sha256"
fi

printf '%s\n' 'Verified local culling model artifacts are ready.'
