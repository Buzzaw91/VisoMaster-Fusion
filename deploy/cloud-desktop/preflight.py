#!/usr/bin/env python3
"""Fail-visible Linux/GPU/Qt/native-encoder smoke checks, with evidence receipts.

Synthetic GPU work proves loaded runtimes execute on CUDA. It does NOT validate
the native face pipeline, native Record, browser cadence, or preview audio.
"""

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from runtime import SOURCE_SHA, atomic_json, check_storage, digest, verify_source


def command(argv, **kwargs):
    result = subprocess.run(argv, capture_output=True, timeout=120, **kwargs)
    return {
        "argv": argv,
        "exit_code": result.returncode,
        "stdout": result.stdout.decode(errors="replace")
        if isinstance(result.stdout, bytes)
        else result.stdout,
        "stderr": result.stderr.decode(errors="replace")
        if isinstance(result.stderr, bytes)
        else result.stderr,
    }


def require_command(argv, **kwargs):
    result = command(argv, **kwargs)
    if result["exit_code"]:
        raise RuntimeError(json.dumps(result))
    return result


def import_probe():
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "onnxruntime",
        "onnx",
        "tensorrt",
        "cv2",
        "numpy",
        "PySide6",
        "qdarktheme",
        "pyqttoast",
        "kornia",
    ):
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "imported")
    # Native Fusion imports TensorFlow lazily for the optional mouth detector.
    # Co-loading it here changes the baseline's framework initialization order
    # and can crash subsequent ORT CUDA session creation. Check its import in a
    # child; this deliberately does not certify mixed-framework mouth analysis.
    tensorflow = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,tensorflow as tf; print(json.dumps({'version':tf.__version__}))",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    versions["tensorflow"] = json.loads(tensorflow.stdout.splitlines()[-1])["version"]
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput  # noqa: F401
    from PySide6.QtMultimediaWidgets import QVideoWidget  # noqa: F401

    return versions


def torch_probe(gpu_id):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is unavailable")
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    start = time.monotonic()
    x = torch.arange(4096, device=device, dtype=torch.float32).reshape(64, 64)
    result = x @ x.T
    torch.cuda.synchronize(gpu_id)
    expected = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    torch.testing.assert_close(result.cpu(), expected @ expected.T)
    properties = torch.cuda.get_device_properties(gpu_id)
    return {
        "device": device,
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(gpu_id)),
        "total_memory_bytes": properties.total_memory,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "elapsed_ms": (time.monotonic() - start) * 1000,
        "allocated_bytes": torch.cuda.memory_allocated(gpu_id),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(gpu_id),
        "gpu_execution_verified": True,
    }


def ort_probe(folder, gpu_id):
    import numpy as np
    from onnx import TensorProto, helper
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [64, 64])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [64, 64])
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "x"], ["y"], name="cuda_square")],
        "CUDA-smoke",
        [x],
        [y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(folder / "ort-smoke")
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"device_id": gpu_id})],
    )
    values = np.arange(4096, dtype=np.float32).reshape(64, 64)
    result = session.run(None, {"x": values})[0]
    np.testing.assert_allclose(result, values * values)
    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text())
    providers = sorted(
        {
            e.get("args", {}).get("provider")
            for e in events
            if e.get("args", {}).get("provider")
        }
    )
    if "CUDAExecutionProvider" not in providers or any(
        p != "CUDAExecutionProvider" for p in providers
    ):
        raise RuntimeError(
            f"ORT graph did not execute exclusively on CUDA: {providers}"
        )
    return {
        "session_providers": session.get_providers(),
        "executed_node_providers": providers,
        "profile_sha256": digest(profile),
        "gpu_execution_verified": True,
        "kind": "synthetic-runtime-smoke; native inference remains required",
    }


def encode_probe(folder, encoder="hevc_nvenc"):
    # Native SDR branch: raw bgr24 -> scale bt709 -> HEVC NVENC main10.
    import numpy as np

    if encoder not in ("hevc_nvenc", "libx265"):
        raise RuntimeError("VISOMASTER_SDR_ENCODER must be hevc_nvenc or libx265")
    output = folder / f"{encoder}-main10-smoke.mp4"
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[:, :, 1] = np.arange(128, dtype=np.uint8)[:, None]
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "128x128",
        "-r",
        "30",
        "-i",
        "pipe:0",
        "-c:v",
        encoder,
    ]
    if encoder == "hevc_nvenc":
        args += [
            "-preset",
            "p4",
            "-profile:v",
            "main10",
            "-cq",
            "20",
            "-pix_fmt",
            "yuv420p10le",
            "-colorspace",
            "rgb",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-spatial-aq",
            "0",
            "-temporal-aq",
            "0",
            "-tier",
            "high",
        ]
    else:
        args += [
            "-preset",
            os.environ.get("VISOMASTER_SDR_X265_PRESET", "veryfast"),
            "-profile:v",
            "main10",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p10le",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ]
    args += [
        "-tag:v",
        "hvc1",
        "-vf",
        "scale=in_range=pc:out_range=tv:out_color_matrix=bt709",
        str(output),
    ]
    encoded = require_command(args, input=frame.tobytes() * 30)
    probe = require_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(output),
        ]
    )
    metadata = json.loads(probe["stdout"])
    video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    if (
        video["codec_name"] != "hevc"
        or video["profile"] != "Main 10"
        or video["pix_fmt"] != "yuv420p10le"
        or int(video["nb_read_frames"]) != 30
    ):
        raise RuntimeError(f"Native encoder output metadata failed: {video}")
    return {
        "encoder": encoder,
        "command": encoded,
        "metadata": metadata,
        "sha256": digest(output),
        "real_encode_verified": True,
        "native_record_validated": False,
        "audio_validated": False,
    }


def qt_probe(app):
    if not os.environ.get("DISPLAY") or os.environ.get("QT_QPA_PLATFORM") in (
        "offscreen",
        "minimal",
    ):
        raise RuntimeError("A visible DISPLAY with the xcb Qt platform is required")
    # Run Qt in a child so plugin aborts are captured as a failed probe.
    code = "from PySide6.QtWidgets import QApplication,QLabel; from PySide6.QtMultimedia import QMediaPlayer,QAudioOutput; from app.ui.core import media_rc,main_window; a=QApplication([]); w=QLabel('Fusion visible Qt preflight'); w.show(); a.processEvents(); print(a.platformName()); assert a.platformName() == 'xcb'; w.close()"
    return require_command([sys.executable, "-c", code], cwd=app)


def native_encoder_policy(app):
    """An explicit compatibility encode is meaningful only if native Record uses it."""
    selected = os.environ.get("VISOMASTER_SDR_ENCODER", "hevc_nvenc")
    if selected == "hevc_nvenc":
        return {"selected": selected, "policy": "stock native SDR NVENC branch"}
    helper = app / "app/processors/video_utils/encoder_policy.py"
    wiring = app / "app/processors/video_utils/video_encoding.py"
    if not helper.is_file() or "sdr_encoder_arguments" not in wiring.read_text():
        raise RuntimeError(
            "Explicit SDR compatibility mode requires a reviewed native encoder-policy source overlay"
        )
    spec = importlib.util.spec_from_file_location(
        "fusion_deployment_encoder_policy", helper
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder, preset = module.sdr_encoder_settings()
    return {
        "selected": encoder,
        "preset": preset,
        "policy_sha256": digest(helper),
        "wiring_sha256": digest(wiring),
    }


def loaded_libraries():
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    return sorted(
        {
            line.split()[-1]
            for line in maps.read_text().splitlines()
            if ".so" in line
            and any(
                name in line.lower()
                for name in ("cuda", "cudnn", "nvinfer", "cublas", "onnx")
            )
        }
    )


def compatibility_key(report, root):
    manifest = root / "model-manifest.json"
    identity = {
        "source": SOURCE_SHA,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "source_manifest_sha256": digest(
            root / "source" / SOURCE_SHA / ".fusion-source.json"
        ),
        "source_patch_sha256": report.get("source_patch_sha256"),
        "versions": report.get("imports"),
        "gpu": report.get("nvidia_smi"),
        "torch": report.get("torch"),
        "model_manifest_sha256": digest(manifest) if manifest.exists() else "missing",
    }
    # Remove volatile memory/timing data from the cache identity.
    identity["torch"] = {
        k: v
        for k, v in (identity["torch"] or {}).items()
        if k in ("name", "capability", "torch_cuda", "cudnn", "total_memory_bytes")
    }
    raw = json.dumps(identity, sort_keys=True).encode()
    import hashlib

    return "linux-" + hashlib.sha256(raw).hexdigest()[:32]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=os.environ.get("FUSION_ROOT", "/workspace/fusion")
    )
    parser.add_argument("--mode", choices=("imports", "gpu", "full"), default="full")
    parser.add_argument(
        "--gpu-id", type=int, default=int(os.environ.get("FUSION_GPU_ID", "0"))
    )
    parser.add_argument(
        "--required-models",
        default=os.environ.get(
            "FUSION_REQUIRED_MODELS",
            "RetinaFace,Inswapper128ArcFace,Inswapper128,FaceLandmark5",
        ),
    )
    parser.add_argument(
        "--minimum-gb",
        type=float,
        default=float(os.environ.get("FUSION_MIN_FREE_GB", "2")),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = args.output or root / "logs" / "preflight.json"
    app = root / "source" / SOURCE_SHA
    report = {
        "source_sha": SOURCE_SHA,
        "time_unix": time.time(),
        "mode": args.mode,
        "platform": platform.platform(),
        "python": sys.version,
        "failures": [],
        "native_inference_validated": False,
        "preview_audio": "noVNC does not supply preview audio",
        "driver_capabilities": os.environ.get(
            "NVIDIA_DRIVER_CAPABILITIES", "host-specific/unspecified"
        ),
        "selected_sdr_encoder": os.environ.get("VISOMASTER_SDR_ENCODER", "hevc_nvenc"),
        "import_check_scope": {
            "tensorflow": "isolated child import only; optional mouth detector co-execution is not validated",
        },
        "selected_sdr_x265_preset": os.environ.get(
            "VISOMASTER_SDR_X265_PRESET", "veryfast"
        ),
        "source_patch_sha256": None,
    }

    def check(name, callback):
        try:
            report[name] = callback()
        except Exception as exc:
            report["failures"].append({"check": name, "error": str(exc)})

    check("storage", lambda: check_storage(root, args.minimum_gb))
    check("source", lambda: verify_source(app) or {"verified": True})
    if (app / ".fusion-source.json").is_file():
        report["source_patch_sha256"] = json.loads(
            (app / ".fusion-source.json").read_text()
        ).get("source_patch_sha256")
    check("imports", import_probe)
    check("ffmpeg", lambda: require_command(["ffmpeg", "-version"]))
    if args.mode in ("gpu", "full"):
        check("native_encoder_policy", lambda: native_encoder_policy(app))
        check(
            "nvidia_smi",
            lambda: require_command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,pci.bus_id,memory.total,compute_cap",
                    "--format=csv,noheader,nounits",
                ]
            ),
        )
        check(
            "nvidia_memory_before",
            lambda: require_command(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ]
            ),
        )
        with tempfile.TemporaryDirectory(
            prefix="preflight-", dir=root / "logs"
        ) as temporary:
            folder = Path(temporary)
            check("torch", lambda: torch_probe(args.gpu_id))
            check("ort", lambda: ort_probe(folder, args.gpu_id))
            selected_encoder = report["selected_sdr_encoder"]
            try:
                report["stock_nvenc"] = {
                    "passed": True,
                    "evidence": encode_probe(folder),
                }
            except Exception as exc:
                report["stock_nvenc"] = {"passed": False, "error": str(exc)}
            if selected_encoder == "hevc_nvenc":
                if report["stock_nvenc"]["passed"]:
                    report["encoder"] = report["stock_nvenc"]["evidence"]
                else:
                    report["failures"].append(
                        {"check": "encoder", "error": report["stock_nvenc"]["error"]}
                    )
            else:
                check("encoder", lambda: encode_probe(folder, selected_encoder))
        from models import verify_models

        def models_check():
            names = (
                None
                if args.required_models == "all"
                else args.required_models.split(",")
            )
            entries = verify_models(app, names)
            if any(not entry["verified"] for entry in entries):
                raise RuntimeError(
                    "Required native models missing or invalid: "
                    + ", ".join(e["name"] for e in entries if not e["verified"])
                )
            return entries

        check("models", models_check)
        check(
            "nvidia_memory_after",
            lambda: require_command(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ]
            ),
        )
    if args.mode == "full":
        check("visible_qt", lambda: qt_probe(app))
    report["loaded_gpu_libraries"] = loaded_libraries()
    report["passed"] = not report["failures"]
    if report["passed"] and args.mode != "imports":
        report["cache_key"] = compatibility_key(report, root)
    atomic_json(output, report)
    print(
        f"Preflight {args.mode}: {'PASS' if report['passed'] else 'FAIL'}; evidence: {output}"
    )
    for failure in report["failures"]:
        print(f"  {failure['check']}: {failure['error']}", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
