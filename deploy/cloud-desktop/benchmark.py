#!/usr/bin/env python3
"""Repeat native Fusion GUI trials; no Qt, Torch or model imports at import time.

Run with the Fusion Python environment and a visible DISPLAY. See results-schema.toml
for the explicit TOML profile format and the boundaries of the measurements.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
BASELINE = "f2d6f5ebe190b631ee5c8972332083cf7f8a3e72"
WEIGHT_SUFFIXES = {".onnx", ".pth", ".ckpt", ".dfm", ".safetensors"}
CACHE_NAMES = {"torch_compile_cache", "tensorrt-engines", "__pycache__"}
STATE_FIELDS = (
    "target_faces_data",
    "input_faces_data",
    "embeddings_data",
    "markers",
    "job_marker_pairs",
    "issue_frames_by_face",
    "dropped_frames",
    "control",
    "swap_faces_enabled",
    "edit_faces_enabled",
    "selected_media_id",
    "selected_target_face_id",
    "current_widget_parameters",
    "target_medias_data",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def command(args: list[str], timeout: float = 60) -> str:
    completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(
            f"{args[0]} failed ({completed.returncode}): {completed.stderr[-4000:]}"
        )
    return completed.stdout


def read_profile(path: Path) -> dict:
    profile = tomllib.loads(path.read_text(encoding="utf-8"))
    settings = profile.get("profile", {})
    for key in ("name", "provider", "precision", "threads", "streams", "hdr"):
        if key not in settings:
            raise ValueError(f"Profile must explicitly specify profile.{key}")
    if settings["provider"] not in {"CUDA", "TensorRT", "TensorRT-Engine"}:
        raise ValueError("provider must be CUDA, TensorRT or TensorRT-Engine")
    if settings["precision"] != "native-model-policy":
        raise ValueError(
            "This native revision only supports precision='native-model-policy'; inspect effective per-model types/options"
        )
    if type(settings["hdr"]) is not bool:
        raise ValueError("hdr must be a TOML boolean")
    for key, limit in (("threads", 30), ("streams", 15)):
        if type(settings[key]) is not int or not 1 <= settings[key] <= limit:
            raise ValueError(f"{key} must be in 1..{limit}")
    for key in ("DetectorModelSelection", "RecognitionModelSelection"):
        if key not in profile.get("control", {}):
            raise ValueError(f"Explicit control.{key} is required")
    for key in (
        "SwapModelSelection",
        "FaceRestorerEnableToggle",
        "FaceRestorerEnable2Toggle",
    ):
        if key not in profile.get("parameters", {}):
            raise ValueError(f"Explicit parameters.{key} is required")
    parameters = profile["parameters"]
    for enabled, model in (
        ("FaceRestorerEnableToggle", "FaceRestorerTypeSelection"),
        ("FaceRestorerEnable2Toggle", "FaceRestorerType2Selection"),
    ):
        if parameters[enabled] and model not in parameters:
            raise ValueError(f"Enabled restorer requires parameters.{model}")
    native = profile.setdefault("native", {})
    if not native.get("seek_frames") or not native.get("analysis_frames"):
        raise ValueError(
            "native.seek_frames and native.analysis_frames must contain actual source frame numbers"
        )
    if float(native.get("play_seconds", 0)) <= 0:
        raise ValueError("native.play_seconds must be positive")
    native.setdefault("require_assignments", True)
    native.setdefault("require_merged_embedding", True)
    native.setdefault("require_reference_kv", False)
    return profile


def fixed_control(profile: dict) -> dict:
    settings = profile["profile"]
    return {
        **profile.get("control", {}),
        "ProvidersPrioritySelection": settings["provider"],
        "nThreadsSlider": settings["threads"],
        "nStreamSlider": settings["streams"],
        "HDREncodeToggle": settings["hdr"],
    }


def model_manifest(app_root: Path) -> dict[str, dict]:
    """Read the literal native hash manifest without importing its directory writers."""
    source = (app_root / "app/processors/models_data.py").read_text(encoding="utf-8")
    manifest = {}
    for statement in ast.parse(source).body:
        if not isinstance(statement, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "models_list"
            for target in statement.targets
        ):
            continue
        if not isinstance(statement.value, ast.List):
            raise ValueError("Unsupported native models_list manifest structure")
        for row in statement.value.elts:
            if not isinstance(row, ast.Dict):
                continue
            fields = {
                key.value: value
                for key, value in zip(row.keys, row.values)
                if isinstance(key, ast.Constant)
            }
            name, digest, location = (
                fields.get("model_name"),
                fields.get("hash"),
                fields.get("local_path"),
            )
            if (
                not isinstance(name, ast.Constant)
                or not isinstance(digest, ast.Constant)
                or not isinstance(location, ast.JoinedStr)
            ):
                continue
            interpolations = [
                part for part in location.values if isinstance(part, ast.FormattedValue)
            ]
            if (
                len(interpolations) != 1
                or not isinstance(interpolations[0].value, ast.Name)
                or interpolations[0].value.id != "models_dir"
            ):
                continue
            relative = "".join(
                part.value
                for part in location.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            ).lstrip("/")
            if re.fullmatch(r"[0-9a-f]{64}", str(digest.value)):
                manifest[relative] = {
                    "model": name.value,
                    "expected_sha256": digest.value,
                }
    return manifest


def ensure_space(destination: Path, required: int, reserve: int) -> None:
    available = shutil.disk_usage(destination).free
    if available < required + reserve:
        raise RuntimeError(
            f"Insufficient space for isolated state: free={available}, copies={required}, reserve={reserve}. Existing files were not removed."
        )


def prepare_runtime(
    app_root: Path,
    project: Path,
    run_dir: Path,
    profile: dict,
    reserve: int = 2 * 1024**3,
    cache_snapshot: Path | None = None,
    preserve_settings: bool = False,
) -> dict:
    """Copy code/state; link ONLY hash-verified native weights. Never link KV state.

    Weight source is an immutable asset store contract: keep that store read-only
    in the container. No hardlinks, model overwrites or cache deletion are used.
    """
    app_root, project, run_dir = (
        app_root.resolve(),
        project.resolve(),
        run_dir.resolve(),
    )
    if app_root.is_relative_to(run_dir) or run_dir.is_relative_to(app_root):
        raise ValueError("Run directory must be separate from the application tree")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Run directory must be new or empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(project.read_text(encoding="utf-8"))
    manifest = model_manifest(app_root)
    copies: list[tuple[Path, Path]] = []
    links: list[tuple[Path, Path]] = []
    identities: list[dict] = []
    runtime = run_dir / "runtime"
    for folder in ("app", "model_assets"):
        source_folder = app_root / folder
        if not source_folder.is_dir():
            raise ValueError(f"Missing native directory: {source_folder}")
        for source in source_folder.rglob("*"):
            relative = source.relative_to(app_root)
            if any(part in CACHE_NAMES for part in relative.parts) or source.is_dir():
                continue
            if source.suffix in {".engine", ".trt", ".timing", ".profile", ".pyc"}:
                continue
            target = runtime / relative
            if source.suffix in WEIGHT_SUFFIXES:
                entry = manifest.get(
                    source.relative_to(app_root / "model_assets").as_posix()
                )
                if not entry:
                    raise ValueError(
                        f"Cannot safely reuse weight outside native hash manifest: {source}. Supply a reviewed manifest before benchmarking this custom model."
                    )
                digest = sha256(source)
                if digest != entry["expected_sha256"]:
                    raise ValueError(f"Model hash mismatch: {source}")
                identities.append(
                    {
                        **entry,
                        "path": str(source),
                        "sha256": digest,
                        "bytes": source.stat().st_size,
                        "storage": "verified-weight-symlink",
                    }
                )
                links.append((source, target))
            else:
                copies.append((source, target))
    for name in ("main.py", "version.json"):
        copies.append((app_root / name, runtime / name))
    inputs: dict[Path, Path] = {}

    def clone_paths(value, key=""):
        if isinstance(value, dict):
            return {k: clone_paths(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [clone_paths(v, key) for v in value]
        if (
            isinstance(value, str)
            and key
            in {"media_path", "kv_map", "assigned_kv_map", "loaded_embedding_filename"}
            and value
        ):
            source = Path(value).expanduser()
            if not source.is_absolute():
                source = app_root / source
            source = source.resolve()
            if not source.is_file():
                raise ValueError(f"Workspace references missing file: {source}")
            if source not in inputs:
                name = (
                    hashlib.sha256(str(source).encode()).hexdigest()[:12]
                    + "-"
                    + source.name
                )
                inputs[source] = run_dir / "inputs" / name
                copies.append((source, inputs[source]))
            return str(inputs[source])
        return value

    data = clone_paths(data)
    if cache_snapshot:
        cache_snapshot = cache_snapshot.resolve()
        for source in cache_snapshot.rglob("*"):
            if source.is_file():
                copies.append(
                    (
                        source,
                        runtime
                        / "tensorrt-engines"
                        / source.relative_to(cache_snapshot),
                    )
                )
    required = sum(source.stat().st_size for source, _ in copies)
    ensure_space(run_dir, required, reserve)
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)  # Never falls through to a full weight copy.
    for source, target in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    for source, target in inputs.items():
        identities.append(
            {
                "kind": "workspace-input",
                "path": str(source),
                "clone": str(target),
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
            }
        )
    controls = fixed_control(profile)
    controls["OutputMediaFolder"] = str(run_dir / "outputs")
    data.setdefault("control", {}).update(
        {"OutputMediaFolder": controls["OutputMediaFolder"]}
        if preserve_settings
        else controls
    )
    for face in data.get("target_faces_data", {}).values():
        if not preserve_settings:
            face.setdefault("parameters", {}).update(profile["parameters"])
    if not preserve_settings:
        data.setdefault("current_widget_parameters", {}).update(profile["parameters"])
    for marker in data.get("markers", {}).values():
        marker.setdefault("control", {}).update(
            {"OutputMediaFolder": controls["OutputMediaFolder"]}
            if preserve_settings
            else controls
        )
        if not preserve_settings:
            for parameters in marker.get("parameters", {}).values():
                parameters.update(profile["parameters"])
    data["last_target_media_folder_path"] = str(run_dir / "inputs")
    data["last_input_media_folder_path"] = str(run_dir / "inputs")
    workspace = run_dir / "workspace.json"
    write_json(workspace, data)
    # A real minimal workspace bypasses the native first-run provider dialog.
    # The explicit project is loaded via the native load action after MainWindow shows.
    write_json(
        runtime / "last_workspace.json",
        {
            "control": {**controls, "AutoLoadWorkspaceToggle": True},
            "target_medias_data": [],
            "input_faces_data": {},
            "target_faces_data": {},
            "embeddings_data": {},
            "markers": {},
            "job_marker_pairs": [],
        },
    )
    (run_dir / "outputs").mkdir()
    return {
        "runtime": str(runtime),
        "workspace": str(workspace),
        "original_project": str(project),
        "original_project_sha256": sha256(project),
        "original_app_root": str(app_root),
        "models_manifest_sha256": sha256(app_root / "app/processors/models_data.py"),
        "files": identities,
        "copy_bytes": required,
        "reserve_bytes": reserve,
        "cache_state": "explicit-snapshot" if cache_snapshot else "fresh-empty-cache",
        "weight_policy": "hash-verified symlinks; mount source assets read-only",
    }


def ffprobe(path: Path, count_frames: bool = True) -> dict:
    args = ["ffprobe", "-v", "error"]
    if count_frames:
        args += ["-count_frames"]
    args += ["-show_streams", "-show_format", "-of", "json", str(path)]
    return json.loads(command(args, timeout=600))


def video_metadata(probe: dict) -> dict:
    videos = [
        stream
        for stream in probe.get("streams", [])
        if stream.get("codec_type") == "video"
    ]
    audio = [
        stream
        for stream in probe.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]
    if not videos:
        raise ValueError("No video stream")
    video = videos[0]
    ratio = video.get("avg_frame_rate", "0/1").split("/")
    fps = (
        float(ratio[0]) / float(ratio[1]) if len(ratio) == 2 and float(ratio[1]) else 0
    )
    frames = video.get("nb_read_frames", video.get("nb_frames"))
    return {
        "width": video.get("width"),
        "height": video.get("height"),
        "frames": int(frames) if frames not in (None, "N/A") else None,
        "duration_seconds": float(
            video.get("duration") or probe.get("format", {}).get("duration") or 0
        ),
        "fps": fps,
        "video_codec": video.get("codec_name"),
        "pixel_format": video.get("pix_fmt"),
        "color_space": video.get("color_space"),
        "color_primaries": video.get("color_primaries"),
        "color_transfer": video.get("color_transfer"),
        "audio": [
            {
                "codec": item.get("codec_name"),
                "channels": item.get("channels"),
                "sample_rate": item.get("sample_rate"),
                "duration": item.get("duration"),
            }
            for item in audio
        ],
    }


def validate_output(
    path: Path,
    source: dict,
    expected_frames: int,
    expected_duration: float,
    hdr: bool,
    expected_size: tuple[int, int] | None = None,
) -> dict:
    metadata = video_metadata(ffprobe(path))
    errors = []
    tolerance = max(0.15, 2 / max(metadata["fps"], 1))
    if "_incomplete" in path.name.lower():
        errors.append("Native export marked incomplete")
    if metadata["frames"] is None or abs(metadata["frames"] - expected_frames) > 1:
        errors.append(
            f"Frame count {metadata['frames']} differs from expected {expected_frames}"
        )
    if abs(metadata["duration_seconds"] - expected_duration) > tolerance:
        errors.append(
            f"Duration {metadata['duration_seconds']} differs from expected {expected_duration}"
        )
    if metadata["video_codec"] != "hevc" or "10" not in (
        metadata["pixel_format"] or ""
    ):
        errors.append("Expected native HEVC 10-bit output")
    if hdr and (
        metadata["color_primaries"] != "bt2020"
        or metadata["color_transfer"] != "smpte2084"
    ):
        errors.append("HDR requested but output lacks native BT.2020/PQ metadata")
    if source.get("audio") and not metadata["audio"]:
        errors.append("Source audio missing from native output")
    for audio in metadata["audio"]:
        if audio.get("duration") and abs(
            float(audio["duration"]) - metadata["duration_seconds"]
        ) > max(0.3, tolerance):
            errors.append("Audio/video duration mismatch; inspect synchronization")
    if expected_size and (metadata["width"], metadata["height"]) != expected_size:
        errors.append(f"Resolution differs from expected {expected_size}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "metadata": metadata,
        "expected_frames": expected_frames,
        "expected_duration_seconds": expected_duration,
        "frame_delta": metadata["frames"] - expected_frames
        if metadata["frames"] is not None
        else None,
        "exact_frame_count": metadata["frames"] == expected_frames,
        "frame_tolerance": 1,
        "duration_tolerance_seconds": tolerance,
        "hdr_requested": hdr,
        "valid": not errors,
        "errors": errors,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    return ordered[lower] + (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    ) * (index - lower)


def cpu_limits() -> dict:
    limits: dict = {
        "logical_cpu_count": os.cpu_count(),
        "affinity_cpu_ids": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }
    for filename in (
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
        "/sys/fs/cgroup/cpuset/cpuset.effective_cpus",
    ):
        path = Path(filename)
        if path.is_file():
            limits[filename] = path.read_text().strip()
    return limits


def source_identity(app_root: Path, patch_receipt: Path | None = None) -> dict:
    """Verify an installed archive receipt, or record actual Git provenance."""
    app_root = app_root.resolve()
    observed = {
        str(path.relative_to(app_root)): sha256(path)
        for path in sorted((app_root / "app").rglob("*.py"))
    }
    observed["main.py"] = sha256(app_root / "main.py")
    identity: dict = {
        "expected_baseline": BASELINE,
        "observed_native_code": observed,
        "observed_native_code_manifest_sha256": hashlib.sha256(
            json.dumps(observed, sort_keys=True).encode()
        ).hexdigest(),
    }
    if (app_root / ".git").exists():
        identity.update(
            {
                "mode": "git",
                "head": command(
                    ["git", "-C", str(app_root), "rev-parse", "HEAD"]
                ).strip(),
                "diff_sha256": hashlib.sha256(
                    command(["git", "-C", str(app_root), "diff", "HEAD"]).encode()
                ).hexdigest(),
            }
        )
        return identity
    manifest_path = app_root / ".fusion-source.json"
    if not manifest_path.is_file():
        raise ValueError(
            "Archive source requires its .fusion-source.json provenance manifest"
        )
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files")
    if (
        not isinstance(files, dict)
        or not files
        or manifest.get("source_sha") != BASELINE
    ):
        raise ValueError(
            "Invalid archive source manifest or unexpected pinned baseline"
        )
    for relative, expected in files.items():
        path = app_root / relative
        if not path.resolve().is_relative_to(app_root):
            raise ValueError(f"Source manifest escapes application root: {relative}")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(
                f"Installed archive source differs from receipt: {relative}"
            )
    unmanifested = sorted(set(observed) - set(files))
    if unmanifested:
        raise ValueError(
            f"Archive contains native code absent from reviewed receipt: {unmanifested}"
        )
    identity.update(
        {
            "mode": "verified-archive",
            "upstream_source_sha": manifest["source_sha"],
            "source_manifest_path": str(manifest_path),
            "source_manifest_sha256": sha256(manifest_path),
            "source_patch_sha256": manifest.get("source_patch_sha256"),
            "source_manifest_files_verified": len(files),
            "unmanifested_native_code": unmanifested,
        }
    )
    patch_digest = manifest.get("source_patch_sha256")
    if patch_digest:
        receipt = (
            patch_receipt
            or app_root.parent.parent
            / "receipts"
            / f"source-patch-{patch_digest}.patch"
        )
        if not receipt.is_file() or sha256(receipt) != patch_digest:
            raise ValueError(f"Source patch receipt missing or changed: {receipt}")
        identity["source_patch_receipt"] = str(receipt)
    return identity


class ResourceSampler:
    """External total-device memory sampler, including ORT/TRT/encode allocations."""

    def __init__(
        self,
        gpu_id: int,
        interval: float = 0.2,
        max_native_gpu_percent: float | None = None,
        stop_request: Path | None = None,
    ):
        self.gpu_id, self.interval = gpu_id, interval
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.pid: int | None = None
        self.loaded_gpu_libraries: set[str] = set()
        self.max_native_gpu_percent = max_native_gpu_percent
        self.stop_request = stop_request
        self.stop_evidence: dict | None = None

    def _sample(self):
        while not self.stopping.is_set():
            started = time.monotonic()
            try:
                output = command(
                    [
                        "nvidia-smi",
                        "-i",
                        str(self.gpu_id),
                        "--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=5,
                )
                row = next(csv.reader(output.splitlines()))
                sample: dict = {
                    "monotonic_seconds": started,
                    "gpu_uuid": row[0].strip(),
                    "gpu_name": row[1].strip(),
                    "driver": row[2].strip(),
                    "total_mib": float(row[3]),
                    "used_mib": float(row[4]),
                    "gpu_utilization_percent": float(row[5]),
                }
                if self.pid:
                    processes = command(
                        [
                            "nvidia-smi",
                            "-i",
                            str(self.gpu_id),
                            "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        timeout=5,
                    )
                    for process_row in csv.reader(processes.splitlines()):
                        if process_row and process_row[0].strip() == str(self.pid):
                            sample["native_process_gpu_used_mib"] = float(
                                process_row[1]
                            )
                    native_used = sample.get("native_process_gpu_used_mib")
                    if (
                        self.max_native_gpu_percent
                        and native_used is not None
                        and native_used
                        > sample["total_mib"] * self.max_native_gpu_percent / 100
                        and self.stop_evidence is None
                    ):
                        self.stop_evidence = {
                            "reason": "Native GPU memory exceeded the configured stability safety boundary",
                            "max_native_gpu_percent": self.max_native_gpu_percent,
                            "sample": sample.copy(),
                        }
                        if self.stop_request:
                            write_json(self.stop_request, self.stop_evidence)
                if self.pid and Path(f"/proc/{self.pid}/status").is_file():
                    status = Path(f"/proc/{self.pid}/status").read_text()
                    match = re.search(r"^VmRSS:\s+(\d+)", status, re.MULTILINE)
                    if match:
                        sample["process_rss_mib"] = int(match[1]) / 1024
                    # Lifetime process CPU seconds, not a synthetic instantaneous utilization.
                    stat = (
                        Path(f"/proc/{self.pid}/stat")
                        .read_text()
                        .rsplit(")", 1)[1]
                        .split()
                    )
                    sample["process_cpu_seconds"] = (
                        int(stat[11]) + int(stat[12])
                    ) / os.sysconf("SC_CLK_TCK")
                    maps = Path(f"/proc/{self.pid}/maps")
                    if maps.is_file():
                        for line in maps.read_text().splitlines():
                            library = line.rsplit(maxsplit=1)[-1]
                            if ".so" in library and any(
                                name in library.lower()
                                for name in (
                                    "cuda",
                                    "cudnn",
                                    "cublas",
                                    "cufft",
                                    "onnxruntime",
                                    "nvinfer",
                                    "nvrtc",
                                    "nccl",
                                )
                            ):
                                self.loaded_gpu_libraries.add(library)
                self.samples.append(sample)
            except Exception as error:
                message = str(error)
                if message not in self.errors:
                    self.errors.append(message)
            self.stopping.wait(max(0.01, self.interval - (time.monotonic() - started)))

    def start(self):
        self.thread.start()

    def stop(self) -> dict:
        self.stopping.set()
        self.thread.join(timeout=6)
        return {
            "sample_interval_seconds": self.interval,
            "samples": self.samples,
            "peak_total_gpu_used_mib": max(
                (item["used_mib"] for item in self.samples), default=None
            ),
            "peak_process_rss_mib": max(
                (item.get("process_rss_mib", 0) for item in self.samples), default=None
            ),
            "errors": self.errors,
            "stability_safety_stop": self.stop_evidence,
            "peak_is_sampled_lower_bound": True,
            "loaded_gpu_libraries": sorted(self.loaded_gpu_libraries),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--app-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Warm analysis/preview/record repetitions after cold warmup",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=7200,
        help="Hard process deadline, including blocked native dialogs",
    )
    parser.add_argument("--stage-timeout", type=float, default=900)
    parser.add_argument("--reserve-gib", type=float, default=2)
    parser.add_argument(
        "--cache-snapshot",
        type=Path,
        help="Explicit compatible TensorRT cache snapshot; never moves/removes source",
    )
    parser.add_argument(
        "--source-patch-receipt",
        type=Path,
        help="Optional explicit patch receipt for an installed source archive",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Diagnostic only: cannot pass visible desktop/client gates",
    )
    parser.add_argument(
        "--out-of-order-probe",
        action="store_true",
        help="Separate untimed diagnostic: delay the penultimate native worker until the highest frame completes; never report throughput",
    )
    parser.add_argument(
        "--fault-probe",
        choices=("early-eof", "missing-frame", "known-skips"),
        help="Separate untimed native fault diagnostic; never reports throughput",
    )
    parser.add_argument(
        "--preserve-output",
        type=Path,
        help="Existing completed video copied into this diagnostic's isolated output directory and checked unchanged",
    )
    parser.add_argument(
        "--restore-only",
        action="store_true",
        help="Load an existing native snapshot in a fresh process and compare saved state/assets; no inference or Record trial",
    )
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Separate untimed backend trace pass; skips Play/Record trials and cannot establish export performance",
    )
    parser.add_argument(
        "--session-minutes",
        type=float,
        default=0,
        help="Bounded native seek/Play loop after normal validation; no extra Record trials or cache eviction",
    )
    parser.add_argument(
        "--hourly-rate", type=float, help="Actual rental rate supplied by operator"
    )
    parser.add_argument(
        "--max-native-gpu-percent",
        type=float,
        help="Fail and stop this native process if its sampled GPU memory crosses this percent of capacity",
    )
    args = parser.parse_args(argv)
    if (
        args.repeat < 1
        or args.timeout <= 0
        or args.stage_timeout <= 0
        or args.reserve_gib < 0
        or not 0 <= args.session_minutes <= 60
        or (
            args.max_native_gpu_percent is not None
            and not 0 < args.max_native_gpu_percent <= 100
        )
    ):
        parser.error(
            "repeat and timeouts must be positive; reserve cannot be negative; session must be 0–60 minutes"
        )
    if args.out_of_order_probe and (
        args.repeat != 1
        or args.restore_only
        or args.backend_only
        or args.session_minutes
    ):
        parser.error(
            "out-of-order-probe requires repeat1, normal visible runtime and no timed session"
        )
    if args.fault_probe and (
        args.repeat != 1
        or args.out_of_order_probe
        or args.restore_only
        or args.backend_only
        or args.session_minutes
        or (args.fault_probe == "missing-frame" and not args.preserve_output)
    ):
        parser.error(
            "fault-probe requires repeat1 and no other trial mode; missing-frame requires preserve-output"
        )
    if args.preserve_output and not args.fault_probe:
        parser.error("preserve-output requires an untimed fault-probe")
    if args.restore_only and (args.backend_only or args.session_minutes):
        parser.error(
            "restore-only cannot be combined with backend-only or a timed session"
        )
    if args.session_minutes and (
        args.backend_only or args.timeout <= args.session_minutes * 60
    ):
        parser.error(
            "A session requires normal validation and a hard timeout longer than its requested duration"
        )
    report: dict = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "unmeasured": [
            "Mac/Windows browser delivered cadence and stalls",
            "browser input-to-preview latency",
            "live preview audio delivery",
            "perceived visual quality and audio synchronization",
            "30 minute interactive stability",
            "replacement-pod recovery",
            "minimum GPU capacity",
            "processing models/features outside the selected profile",
        ],
        "host": {"platform": platform.platform(), "python": sys.version},
        "rental_hourly_rate": args.hourly_rate,
        "visible_display_requested": not args.offscreen,
        "trial_kind": "untimed-native-fault-probe:" + args.fault_probe
        if args.fault_probe
        else "untimed-native-out-of-order-probe"
        if args.out_of_order_probe
        else "fresh-process-native-restore"
        if args.restore_only
        else "backend-diagnostic"
        if args.backend_only
        else "native-functional-and-interactive-session"
        if args.session_minutes
        else "native-functional-and-performance",
        "session_minutes": args.session_minutes,
    }
    if args.backend_only:
        report["unmeasured"].extend(
            [
                "record throughput/frame/audio output verification (backend-only pass)",
                "local sustained playback (backend-only pass)",
            ]
        )
    report["host"]["cpu_limits"] = cpu_limits()
    report["encoder_environment"] = {
        name: os.environ.get(name)
        for name in ("VISOMASTER_SDR_ENCODER", "VISOMASTER_SDR_X265_PRESET")
    }
    report["runtime_environment"] = {
        name: os.environ.get(name)
        for name in (
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_MODULE_LOADING",
            "PYTORCH_CUDA_ALLOC_CONF",
            "VISOMASTER_CUDA_UNIFIED_STREAM",
            "VISOMASTER_CUDA_ARENA_SHRINK",
            "DISPLAY",
            "QT_QPA_PLATFORM",
        )
    }
    # Refuse reuse before a failure report could overwrite someone else's evidence.
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("run-dir must be new or empty")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sampler = None
    try:
        profile = read_profile(args.profile)
        if not args.offscreen and (
            not os.environ.get("DISPLAY")
            or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        ):
            raise RuntimeError(
                "A visible DISPLAY is required. --offscreen is explicitly diagnostic only."
            )
        report["profile"] = profile
        report["profile_sha256"] = sha256(args.profile)
        report["harness_sha256"] = {
            name: sha256(Path(__file__).with_name(name))
            for name in ("benchmark.py", "native_smoke.py", "results-schema.toml")
        }
        report["installed_dependency_versions"] = {
            item.metadata["Name"]: item.version
            for item in importlib.metadata.distributions()
            if item.metadata["Name"]
        }
        report["source"] = source_identity(args.app_root, args.source_patch_receipt)
        preparation = prepare_runtime(
            args.app_root,
            args.project,
            args.run_dir,
            profile,
            int(args.reserve_gib * 1024**3),
            args.cache_snapshot,
            preserve_settings=args.restore_only,
        )
        report["preparation"] = preparation
        preserved = None
        if args.preserve_output:
            source = args.preserve_output.resolve(strict=True)
            ensure_space(
                args.run_dir, source.stat().st_size, int(args.reserve_gib * 1024**3)
            )
            target = args.run_dir / "outputs" / ("preserved-" + source.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            preserved = {
                "original": str(source),
                "clone": str(target),
                "sha256": sha256(source),
            }
            report["preserved_output"] = preserved
        report["preparation_seconds"] = time.monotonic() - started
        write_json(
            args.run_dir / "request.json",
            {
                "profile": profile,
                "preparation": preparation,
                "repeat": args.repeat,
                "gpu_id": args.gpu_id,
                "stage_timeout": args.stage_timeout,
                "offscreen": args.offscreen,
                "backend_only": args.backend_only,
                "restore_only": args.restore_only,
                "out_of_order_probe": args.out_of_order_probe,
                "fault_probe": args.fault_probe,
                "preserved_output": preserved,
                "session_minutes": args.session_minutes,
            },
        )
        environment = os.environ.copy()
        if args.offscreen:
            environment["QT_QPA_PLATFORM"] = "offscreen"
        environment["PYTORCH_CUDA_ALLOC_CONF"] = environment.get(
            "PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8"
        )
        sampler = ResourceSampler(
            args.gpu_id,
            max_native_gpu_percent=args.max_native_gpu_percent,
            stop_request=args.run_dir / "stability-stop-request.json",
        )
        sampler.start()
        process_start = time.monotonic()
        with (args.run_dir / "native.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("native_smoke.py")),
                    "--request",
                    str((args.run_dir / "request.json").resolve()),
                ],
                cwd=preparation["runtime"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            sampler.pid = process.pid
            try:
                safety_deadline = None
                while True:
                    remaining = args.timeout - (time.monotonic() - process_start)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(process.args, args.timeout)
                    try:
                        exit_code = process.wait(timeout=min(0.5, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        if sampler.stop_evidence:
                            if safety_deadline is None:
                                safety_deadline = time.monotonic() + 5
                            elif time.monotonic() >= safety_deadline:
                                process.terminate()
                                try:
                                    exit_code = process.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    exit_code = process.wait(timeout=10)
                                break
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise RuntimeError(
                    "Native process exceeded hard timeout (possible model build/dialog/finalize stall); outputs are unverified"
                )
        report["native_process_seconds"] = time.monotonic() - process_start
        native_path = args.run_dir / "native-results.json"
        if native_path.is_file():
            report["native"] = json.loads(native_path.read_text())
        if exit_code or report.get("native", {}).get("status") != "passed":
            raise RuntimeError(
                f"Native validation failed; process exit={exit_code}. See native-results.json and native.log"
            )
        if sha256(args.project) != preparation["original_project_sha256"]:
            raise RuntimeError(
                "Original project changed during trial; preservation check failed"
            )
        report["original_project_unchanged"] = True
        report["status"] = (
            "passed_untimed_fault_diagnostic"
            if args.fault_probe
            else "passed_untimed_ordering_diagnostic"
            if args.out_of_order_probe
            else "passed_native_restore"
            if args.restore_only
            else "passed_backend_diagnostic"
            if args.backend_only
            else "passed_native_automation"
            if not args.offscreen
            else "passed_offscreen_diagnostic"
        )
    except Exception as error:
        report["error"] = str(error)
    finally:
        if sampler:
            report["resources"] = sampler.stop()
            if not report["resources"]["samples"] and report["status"].startswith(
                "passed"
            ):
                report["status"] = "failed"
                report["error"] = (
                    "No nvidia-smi samples: total GPU memory evidence is missing"
                )
            for stage in report.get("native", {}).get("stages", []):
                samples = [
                    item
                    for item in report["resources"]["samples"]
                    if stage["started_monotonic_seconds"]
                    <= item["monotonic_seconds"]
                    <= stage["ended_monotonic_seconds"]
                ]
                stage["peak_sampled_total_gpu_used_mib"] = max(
                    (item["used_mib"] for item in samples), default=None
                )
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.run_dir / "results.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "results": str((args.run_dir / "results.json").resolve()),
                "error": report.get("error"),
            },
            indent=2,
        )
    )
    return 0 if report["status"].startswith("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
