#!/usr/bin/env python3
"""Use the pinned native catalog/downloader, with nondestructive existing assets."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

from runtime import (
    SOURCE_SHA,
    atomic_json,
    check_storage,
    digest,
    read_json,
    validate_owner,
)


def catalog(app):
    sys.path.insert(0, str(app))
    from app.processors.models_data import models_list

    return models_list


def verify_models(app, names=None):
    models = catalog(app)
    if names is not None:
        models = [m for m in models if m["model_name"] in names]
        missing_names = set(names) - {m["model_name"] for m in models}
        if missing_names:
            raise RuntimeError(
                f"Unknown pinned native model names: {sorted(missing_names)}"
            )
    result = []
    for model in models:
        path = Path(model["local_path"])
        actual = digest(path) if path.is_file() else None
        result.append(
            {
                "name": model["model_name"],
                "path": str(path),
                "expected_sha256": model["hash"],
                "actual_sha256": actual,
                "verified": actual == model["hash"],
                "url": model["url"],
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=os.environ.get("FUSION_ROOT", "/workspace/fusion")
    )
    parser.add_argument(
        "--mode", choices=("catalog", "verify", "download"), default="verify"
    )
    parser.add_argument(
        "--models",
        default="all",
        help="all or comma-separated exact native model names",
    )
    parser.add_argument(
        "--reuse-root",
        type=Path,
        help="Existing model folder; files are reused only after exact SHA256 verification",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    app = root / "source" / SOURCE_SHA
    names = None if args.models == "all" else args.models.split(",")
    entries = catalog(app)
    if names is not None:
        unknown = set(names) - {m["model_name"] for m in entries}
        if unknown:
            raise RuntimeError(f"Unknown native model names: {sorted(unknown)}")
        entries = [m for m in entries if m["model_name"] in names]
    if args.mode == "download":
        validate_owner(root)
        from app.helpers.downloader import download_file

        manifest = root / "model-reuse-manifest.json"
        reused = read_json(manifest) if manifest.exists() else []
        for model in entries:
            dest = Path(model["local_path"])
            if dest.exists():
                if digest(dest) != model["hash"]:
                    raise RuntimeError(
                        f"Existing model hash mismatch; preserved for investigation: {dest}"
                    )
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if args.reuse_root:
                relative = dest.relative_to(app / "model_assets")
                candidate = args.reuse_root / relative
                if candidate.is_file() and digest(candidate) == model["hash"]:
                    try:
                        os.link(candidate, dest)
                        method = "hardlink"
                    except OSError:
                        # Cross-volume reuse has explicit provenance and is rehashed by preflight.
                        dest.symlink_to(candidate.resolve())
                        method = "symlink"
                    reused.append(
                        {
                            "source": str(candidate.resolve()),
                            "destination": str(dest),
                            "sha256": model["hash"],
                            "method": method,
                        }
                    )
                    continue
            # Download to a temporary sibling: upstream may delete/retry ONLY this file.
            quota = None
            if os.environ.get("FUSION_VOLUME_QUOTA_FREE_GB") is not None:
                import requests

                reserve_gb = float(os.environ.get("FUSION_MIN_FREE_GB", "2"))
                storage = check_storage(root, reserve_gb)
                quota = storage["allocated_volume_quota"]
                with requests.get(model["url"], stream=True, timeout=20) as response:
                    response.raise_for_status()
                    length = int(response.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise RuntimeError(
                        f"Cannot reserve operator quota for unknown download size: {model['model_name']}"
                    )
                if (
                    length + reserve_gb * 1024**3
                    > storage["effective_available_bytes_upper_bound"]
                ):
                    raise RuntimeError(
                        f"Model download exceeds operator quota bound/reserve: {model['model_name']} ({length} bytes)"
                    )
            fd, temp_name = tempfile.mkstemp(
                prefix=".download-", suffix=dest.suffix, dir=dest.parent
            )
            os.close(fd)
            temp = Path(temp_name)
            temp.unlink()
            try:
                if not download_file(
                    model["model_name"], str(temp), model["hash"], model["url"]
                ):
                    raise RuntimeError(
                        f"Native model download failed: {model['model_name']}"
                    )
                if digest(temp) != model["hash"]:
                    raise RuntimeError("Downloaded model hash mismatch")
                # link publishes without replacing a file produced concurrently.
                os.link(temp, dest)
                if quota:
                    atomic_json(
                        root / "receipts" / "quota-consumption.json",
                        {
                            "declared_free_bytes": quota[
                                "declared_free_bytes_upper_bound"
                            ],
                            "operator_checked_at": quota["operator_checked_at"],
                            "download_bytes_since_observation": quota[
                                "download_bytes_since_observation"
                            ]
                            + dest.stat().st_size,
                            "note": "new model payload bytes only; static observation, not authoritative dynamic quota",
                        },
                    )
            finally:
                temp.unlink(missing_ok=True)
        atomic_json(root / "model-reuse-manifest.json", reused)
    results = verify_models(app, names)
    atomic_json(
        root / "model-manifest.json",
        {
            "source_sha": SOURCE_SHA,
            "catalog_sha256": digest(app / "app/processors/models_data.py"),
            "models": results,
        },
    )
    if args.mode == "catalog":
        for entry in results:
            print(f"{entry['name']}\t{entry['expected_sha256']}\t{entry['path']}")
    else:
        missing = [m["name"] for m in results if not m["verified"]]
        print(
            f"Native model hashes: {len(results) - len(missing)}/{len(results)} verified"
        )
        if missing:
            raise RuntimeError(
                f"Missing or invalid native models: {', '.join(missing)}"
            )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Model setup refused: {exc}", file=sys.stderr)
        sys.exit(1)
