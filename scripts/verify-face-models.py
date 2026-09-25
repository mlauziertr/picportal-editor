#!/usr/bin/env python3
"""Fail closed unless the canonical face bundle matches its checked-in manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


CANONICAL_ARTIFACTS = {
    "face_detection_yunet_2023mar.onnx": ("detector", "MIT"),
    "face_recognition_sface_2021dec.onnx": ("recognizer", "Apache-2.0"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require_file(path: Path, expected_bytes: int | None, expected_sha256: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing required face artifact: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise SystemExit(f"unexpected byte size for {path}")
    if digest(path) != expected_sha256:
        raise SystemExit(f"unexpected SHA-256 for {path}")


def require_tracked(repository: Path, path: Path) -> None:
    relative = path.resolve().relative_to(repository.resolve())
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"face artifact is not intentionally tracked by Git: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/face"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--require-tracked", action="store_true")
    parser.add_argument("--skip-fixture", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    manifest_path = model_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid face manifest {manifest_path}: {error}") from error

    if (
        manifest.get("version") != 1
        or manifest.get("modelId") != "opencv-yunet-sface-v1"
        or manifest.get("pipelineVersion") != "opencv-yunet-sface-canonical-1"
        or manifest.get("embeddingDimension") != 128
        or manifest.get("detectorInput") != 640
        or manifest.get("recognizerInput") != 112
        or manifest.get("detectorScoreThreshold") != 0.9
        or manifest.get("detectorNmsThreshold") != 0.3
        or manifest.get("cosineDistanceThreshold") != 0.637
        or manifest.get("maxFaces") != 16
        or manifest.get("faceThumbnail")
        != {"width": 220, "height": 220, "quality": 82, "cropScale": 1.85}
        or manifest.get("runtimeCompatibility")
        != {
            "rustCrateVersion": "2.0.0-rc.10",
            "rustRuntimeVersion": "1.22.0",
            "pythonPackageVersion": "1.24.4",
            "embeddingCosineMinimum": 0.999,
        }
    ):
        raise SystemExit("canonical face contract metadata is invalid")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or {
        artifact.get("filename") for artifact in artifacts if isinstance(artifact, dict)
    } != set(CANONICAL_ARTIFACTS):
        raise SystemExit("canonical face artifact set is invalid")
    if {path.name for path in model_dir.glob("*.onnx")} != set(CANONICAL_ARTIFACTS):
        raise SystemExit("model directory contains an unmanifested or missing ONNX artifact")

    descriptor = bytearray()
    tracked_paths = [manifest_path]
    for artifact in artifacts:
        filename = artifact["filename"]
        expected_role, expected_license = CANONICAL_ARTIFACTS[filename]
        if artifact.get("role") != expected_role or artifact.get("license") != expected_license:
            raise SystemExit(f"unexpected declared license for {filename}")
        source_url = str(artifact.get("sourceUrl", ""))
        license_url = str(artifact.get("licenseUrl", ""))
        if not source_url.startswith("https://") or not license_url.startswith("https://"):
            raise SystemExit(f"unpinned source URL for {filename}")
        revision = artifact.get("sourceRevision", "")
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise SystemExit(f"invalid source revision for {filename}")
        if revision not in source_url or revision not in license_url:
            raise SystemExit(f"source URLs do not contain the pinned revision for {filename}")
        artifact_path = model_dir / filename
        license_path = model_dir / artifact["licenseFilename"]
        require_file(artifact_path, artifact["bytes"], artifact["sha256"])
        require_file(license_path, None, artifact["licenseSha256"])
        descriptor.extend(f"{filename} {artifact['sha256']}\n".encode("ascii"))
        tracked_paths.extend((artifact_path, license_path))

    if hashlib.sha256(descriptor).hexdigest() != manifest.get("bundleSha256"):
        raise SystemExit("canonical face bundle digest does not match the manifest")

    if not args.skip_fixture:
        fixture = manifest.get("testFixture", {})
        fixture_path = model_dir / fixture.get("filename", "")
        require_file(fixture_path, fixture.get("bytes"), fixture.get("sha256", ""))
        expected_path = model_dir / fixture.get("expectedFilename", "")
        prompt_path = model_dir / fixture.get("promptFilename", "")
        license_path = model_dir / fixture.get("licenseFilename", "")
        require_file(expected_path, fixture.get("expectedBytes"), fixture.get("expectedSha256", ""))
        require_file(prompt_path, fixture.get("promptBytes"), fixture.get("promptSha256", ""))
        require_file(license_path, fixture.get("licenseBytes"), fixture.get("licenseSha256", ""))
        if (
            fixture.get("license") != "CC0-1.0"
            or fixture.get("sourceType") != "internally-generated"
            or fixture.get("generator") != "OpenAI image generation"
            or not fixture.get("generatedAt")
        ):
            raise SystemExit("unexpected fixture provenance or license")
        tracked_paths.extend((fixture_path, expected_path, prompt_path, license_path))

    if args.require_tracked:
        for path in tracked_paths:
            require_tracked(args.repository, path)

    print(
        f"verified {manifest['modelId']} ({manifest['bundleSha256']}), "
        f"{len(artifacts)} licensed ONNX artifacts"
    )


if __name__ == "__main__":
    main()
