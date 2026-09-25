#!/usr/bin/env python3
"""Download the optional subject/pose culling models listed in models/culling/manifest.json.

Every artifact is checked for exact size and SHA-256 before it is moved into place;
files already present with the right hash are skipped. Grounding DINO weights are
about 689 MB. The destination defaults to models/culling (or PICPORTAL_CULLING_MODEL_DIR).

Usage: python3 scripts/fetch-culling-models.py [--dest DIR] [--check]
  --check  only verify what is present, download nothing (exit 1 if incomplete).
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "models" / "culling" / "manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path, expected_sha, expected_bytes):
    return path.is_file() and path.stat().st_size == expected_bytes and sha256(path) == expected_sha


def fetch(url, destination, expected_sha, expected_bytes):
    if not url.startswith("https://"):
        raise SystemExit(f"refusing non-HTTPS URL for {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    with urllib.request.urlopen(url) as response, open(temporary, "wb") as output:
        shutil.copyfileobj(response, output, 1 << 20)
    if not is_valid(temporary, expected_sha, expected_bytes):
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"size or SHA-256 mismatch for {destination}")
    os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default=os.environ.get("PICPORTAL_CULLING_MODEL_DIR") or MANIFEST.parent)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = pathlib.Path(args.dest)
    manifest = json.loads(MANIFEST.read_text())
    subject, pose = manifest["subjectModel"], manifest["poseModel"]
    entries = [
        (subject["licenseUrl"], subject["licenseFilename"], subject["licenseSha256"], subject["licenseBytes"]),
        *[(a["url"], a["filename"], a["sha256"], a["bytes"]) for a in subject["artifacts"]],
        (pose["sourceUrl"], pose["filename"], pose["sha256"], pose["bytes"]),
    ]
    if destination.resolve() != MANIFEST.parent.resolve():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MANIFEST, destination / "manifest.json")
        for name in ("POSE-LICENSE.txt",):
            if (MANIFEST.parent / name).is_file():
                shutil.copy2(MANIFEST.parent / name, destination / name)
    missing = 0
    for url, filename, expected_sha, expected_bytes in entries:
        target = destination / filename
        if is_valid(target, expected_sha, expected_bytes):
            print(f"ok       {filename}")
            continue
        if args.check:
            print(f"missing  {filename}")
            missing += 1
            continue
        print(f"download {filename} ({expected_bytes / 1e6:.1f} MB)")
        fetch(url, target, expected_sha, expected_bytes)
    if missing:
        return 1
    print(f"Verified local culling model artifacts in {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
