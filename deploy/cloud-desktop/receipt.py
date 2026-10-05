#!/usr/bin/env python3
"""Capture a truthful installation identity; unresolved fields stay explicit."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from runtime import SOURCE_SHA, atomic_json, digest, read_json

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--source-repo", required=True)
parser.add_argument("--resolution-only", action="store_true")
args = parser.parse_args()
root = Path(args.root)


def command(argv):
    result = subprocess.run(argv, text=True, capture_output=True)
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


lock = Path(os.environ["FUSION_INSTALL_LOCK"])
if args.resolution_only:
    atomic_json(
        root / "receipts" / "resolution.json",
        {
            "source_sha": SOURCE_SHA,
            "source_manifest_sha256": digest(
                root / "source" / SOURCE_SHA / ".fusion-source.json"
            ),
            "created_unix": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "dependency_status": "resolved-hashed-not-installed",
            "requirements_path": str(lock),
            "requirements_sha256": digest(lock),
            "application_packages_installed_by_this_step": False,
            "native_functionality": "unvalidated",
        },
    )
    print("Resolution-only receipt written; no application package install claimed")
    sys.exit(0)
base_reference = os.environ.get("FUSION_BASE_DIGEST", "UNVERIFIED")
data = {
    "source_sha": SOURCE_SHA,
    "source_repository": args.source_repo,
    "source_patch_sha256": read_json(
        root / "source" / SOURCE_SHA / ".fusion-source.json"
    ).get("source_patch_sha256"),
    "source_manifest_sha256": digest(
        root / "source" / SOURCE_SHA / ".fusion-source.json"
    ),
    "created_unix": time.time(),
    "python": sys.version,
    "platform": platform.platform(),
    "dependency_status": os.environ["FUSION_DEPENDENCY_STATUS"],
    "import_check_status": os.environ.get(
        "FUSION_IMPORT_CHECK_STATUS", "unknown-no-import-pass-claimed"
    ),
    "build_platform": os.environ.get("FUSION_BUILD_PLATFORM", "not-an-image-build"),
    "target_platform": os.environ.get("FUSION_TARGET_PLATFORM", "not-an-image-build"),
    "gpu_execution_verified": False,
    "requirements_path": str(lock),
    "requirements_sha256": digest(lock),
    "installed": sorted(
        {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        }.items()
    ),
    "os_packages": command(["dpkg-query", "-W", "-f=${Package}\t${Version}\n"]),
    "ffmpeg": command(["ffmpeg", "-version"]),
    "image_digest": os.environ.get("FUSION_IMAGE_DIGEST", "UNVERIFIED"),
    "base_reference": base_reference,
    "base_digest": base_reference.rsplit("@", 1)[-1]
    if "@sha256:" in base_reference
    else "UNVERIFIED",
    "model_catalog_sha256": digest(
        root / "source" / SOURCE_SHA / "app/processors/models_data.py"
    ),
    "native_functionality": "unvalidated",
    "preview_audio": "noVNC does not transport audio",
}
atomic_json(root / "receipts" / "setup.json", data)
print("Setup receipt written (no runtime credentials recorded)")
