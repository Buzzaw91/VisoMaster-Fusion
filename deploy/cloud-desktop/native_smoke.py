#!/usr/bin/env python3
"""GUI-driven native validation worker for benchmark.py.

All native imports are delayed until main(). The observer watches native Qt signals
and viewport paint events. Explicit untimed fault probes inject delays/read failures
only in a disposable process; ordinary trials never patch the pipeline. Execute
through benchmark.py so state is isolated first.
"""

from __future__ import annotations

import argparse
import collections
import faulthandler
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from benchmark import (
    STATE_FIELDS,
    ffprobe,
    percentile,
    sha256,
    validate_output,
    video_metadata,
    write_json,
)


def state_signature(data: dict) -> dict:
    """Compare persisted semantic state, including external KV file identities."""

    def normalize(value, key=""):
        if isinstance(value, dict):
            return {str(k): normalize(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(item, key) for item in value]
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, str) and key in {"kv_map", "assigned_kv_map"} and value:
            path = Path(value)
            return (
                {"file_sha256": sha256(path)}
                if path.is_file()
                else {"missing_file": value}
            )
        return value

    return {key: normalize(data.get(key)) for key in STATE_FIELDS}


def expected_recording(window, source: dict) -> tuple[int, float]:
    """Native segment endpoints are inclusive; deliberate drop frames are removed."""
    processor = window.video_processor
    fps = float(processor.fps)
    if fps <= 0:
        raise ValueError("Native source FPS is invalid")
    if window.control.get("OutputFpsCapEnableToggle", False):
        raise ValueError(
            "FPS-cap validation is not implemented; use an explicit profile without FPS cap"
        )
    ranges = window.job_marker_pairs or [
        (window.videoSeekSlider.value(), processor.max_frame_number)
    ]
    frames = 0
    for start, end in ranges:
        if end is None or not 0 <= start < end <= processor.max_frame_number:
            raise ValueError(f"Invalid native record range {(start, end)}")
        frames += (
            end
            - start
            + 1
            - sum(start <= frame <= end for frame in window.dropped_frames)
        )
    return frames, frames / fps


def session_steps(frames: list[int], iteration: int, seek, play, check) -> list:
    """One bounded cycle, with frame/repetition values bound before Qt queues it."""
    sequence: list = [
        (
            f"session_seek_{iteration}_{frame}",
            lambda f=frame, r=iteration: seek(f, "session", r),
        )
        for frame in frames
    ]
    sequence += [
        (f"session_play_{iteration}", lambda r=iteration: play(r, "session")),
        (f"session_check_{iteration}", check),
    ]
    return sequence


def matching_processed_paint(
    paints: list[dict], frame_token, pixmap_key, frame
) -> dict | None:
    """Match native paints even when Qt delivers the completion observer later."""
    return next(
        (
            paint
            for paint in reversed(paints)
            if paint["native_frame_token"] == frame_token
            and paint["pixmap_key"] == pixmap_key
            and paint["slider_value"] == frame
        ),
        None,
    )


def out_of_order_errors(probe: dict, proofs: list[dict]) -> list[str]:
    """Reject output reordering even when decoded frame count remains correct."""
    highest = probe["highest_frame"]
    emitted = [item["frame"] for item in probe["signals_emitted"]]
    delivered = [item["frame"] for item in probe["signals_delivered"]]
    expected = list(range(highest + 1))
    errors = []
    if emitted != [highest, highest - 1] or delivered != [
        highest,
        highest - 1,
    ]:
        errors.append(
            "Injected highest-before-penultimate native signal order was not established"
        )
    if probe["encoder_frame_indices"] != expected:
        errors.append("Native encoder frame order differed from source order")
    if not any(
        item["pending_tasks"] == 0 and item["is_drain_complete"]
        for item in probe["natural_eof_observations"]
    ):
        errors.append("Natural EOF zero unfinished-task gate was not directly observed")
    if not all(item["exact_frame_count"] for item in proofs):
        errors.append("Untimed diagnostic did not preserve exact decoded frame count")
    return errors


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    preparation = request["preparation"]
    runtime = Path(preparation["runtime"]).resolve()
    run_dir = args.request.resolve().parent
    if runtime.parent != run_dir or not (runtime / "last_workspace.json").is_file():
        raise ValueError("Use benchmark.py to create an isolated runtime")
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    started = time.monotonic()
    result = {
        "status": "failed",
        "stages": [],
        "analysis": [],
        "preview": [],
        "playback": [],
        "records": [],
        "dialogs": [],
        "backend": {},
        "visible": not request["offscreen"],
        "measurement_scope": "native application compute and local viewport paint; excludes browser transport",
    }
    try:
        import torch

        torch.set_grad_enabled(False)
        from PySide6 import QtCore, QtWidgets
        import qdarktheme
        from app.ui.core.proxy_style import ProxyStyle
        from app.ui.main_ui import MainWindow
        from app.ui.widgets.actions import save_load_actions, video_control_actions
        from app.ui.widgets.widget_components import CreateEmbeddingDialog
        from app.processors.utils import platform_support

        profile = request["profile"]
        settings, native = profile["profile"], profile["native"]
        requested_provider = settings["provider"]
        if requested_provider not in platform_support.available_execution_providers():
            raise RuntimeError(
                f"Requested native provider {requested_provider} is unavailable; fallback is forbidden for this trial"
            )
        app = QtWidgets.QApplication([])
        app.setStyle(ProxyStyle())
        style = (runtime / "app/ui/styles/true_dark_styles.qss").read_text()
        app.setStyleSheet(
            qdarktheme.load_stylesheet(
                theme="dark", custom_colors={"primary": "#4090a3"}
            )
            + "\n"
            + style
        )
        window = MainWindow(gpu_id=request["gpu_id"])
        window.show()
        result["cold_startup_seconds"] = time.monotonic() - started
        result["qt_platform"] = app.platformName()

        class Driver(QtCore.QObject):
            def __init__(self):
                super().__init__()
                self.steps = collections.deque()
                self.pending = None
                self.pending_frame = None
                self.processed_frame = False
                self.preview_generation = None
                self.preview_signals = []
                self.preview_paints = []
                self.processed_token = None
                self.processed_pixmap_key = None
                self.play_paints = []
                self.last_paint_frame = None
                self.record_before = set()
                self.record_started = False
                self.encoder_processes = {}
                self.stage_started = 0.0
                self.current_label = "initialization"
                self.failure = None
                self.finished = False
                self.profiler_sessions = {}
                self.seen_worker_native_ids: set[int] = set()
                self.model_build_stack = []
                self.order_probe = None
                self.order_previous_trace = None
                self.order_original_process_frame = None
                self.order_highest_emitted = threading.Event()
                self.fault_probe = None
                self.fault_originals = []
                self.timer = QtCore.QTimer(self)
                self.timer.setInterval(100)
                self.timer.timeout.connect(self.tick)
                self.timer.start()
                window.graphicsViewFrame.viewport().installEventFilter(self)
                window.video_processor.single_frame_processed_signal.connect(
                    self.single_processed, QtCore.Qt.ConnectionType.QueuedConnection
                )
                if request.get("out_of_order_probe"):
                    window.video_processor.frame_processed_signal.connect(
                        self.order_signal_emitted,
                        QtCore.Qt.ConnectionType.DirectConnection,
                    )
                    window.video_processor.frame_processed_signal.connect(
                        self.order_signal_delivered,
                        QtCore.Qt.ConnectionType.QueuedConnection,
                    )
                window.video_processor.processing_started_signal.connect(
                    self.processing_started, QtCore.Qt.ConnectionType.QueuedConnection
                )
                window.video_processor.processing_stopped_signal.connect(
                    self.processing_stopped, QtCore.Qt.ConnectionType.QueuedConnection
                )
                window.video_processor.fatal_processing_error_signal.connect(
                    self.fail, QtCore.Qt.ConnectionType.QueuedConnection
                )
                window.models_processor.show_build_dialog.connect(
                    self.model_build_started
                )
                window.models_processor.hide_build_dialog.connect(
                    self.model_build_stopped
                )
                self.steps.append(("workspace_load", self.load))
                self.steps.append(("native_controls", self.controls))
                for index, frame in enumerate(native["analysis_frames"]):
                    self.steps.append(
                        (
                            f"cold_analysis_seek_{index}",
                            lambda f=frame: self.seek(f, "warmup", -1),
                        )
                    )
                    self.steps.append(
                        (
                            f"first_find_faces_{index}",
                            lambda f=frame: self.find_faces(
                                f, "first-after-project-load"
                            ),
                        )
                    )
                self.steps.append(("assignments_and_embedding", self.assignments))
                self.steps.append(
                    (
                        "warmup_processed_preview",
                        lambda: self.seek(native["seek_frames"][0], "warmup", -1),
                    )
                )
                self.steps.append(("native_state_roundtrip", self.roundtrip))
                for repeat in range(
                    0 if request.get("backend_only", False) else request["repeat"]
                ):
                    for index, frame in enumerate(native["analysis_frames"]):
                        self.steps.append(
                            (
                                f"warm_analysis_seek_{repeat}_{index}",
                                lambda f=frame, r=repeat: self.seek(
                                    f, "analysis-navigation", r
                                ),
                            )
                        )
                        self.steps.append(
                            (
                                f"warm_find_faces_{repeat}_{index}",
                                lambda f=frame: self.find_faces(f, "warm"),
                            )
                        )
                    for index, frame in enumerate(native["seek_frames"]):
                        self.steps.append(
                            (
                                f"warm_preview_{repeat}_{index}",
                                lambda f=frame, r=repeat: self.seek(f, "warm", r),
                            )
                        )
                    self.steps.append(
                        (
                            f"play_seek_{repeat}",
                            lambda r=repeat: self.seek(
                                native["seek_frames"][0], "play-navigation", r
                            ),
                        )
                    )
                    self.steps.append((f"play_{repeat}", lambda r=repeat: self.play(r)))
                    self.steps.append(
                        (
                            f"record_seek_{repeat}",
                            lambda r=repeat: self.seek(0, "record-navigation", r),
                        )
                    )
                    self.steps.append(
                        (f"record_{repeat}", lambda r=repeat: self.record(r))
                    )
                if not request.get("fault_probe"):
                    self.steps.append(("effective_backend_proof", self.backend_proof))
                if request.get("session_minutes", 0):
                    self.steps.insert(
                        len(self.steps) - 1,
                        ("interactive_session_begin", self.begin_session),
                    )
                self.steps.append(("final_state_roundtrip", self.roundtrip))
                self.steps.append(("complete", self.complete))
                if request.get("restore_only"):
                    self.steps.clear()
                    self.steps.extend(
                        [
                            ("workspace_load", self.load),
                            ("fresh_process_restore", self.restore_check),
                            ("complete", self.complete),
                        ]
                    )
                QtCore.QTimer.singleShot(0, self.advance)

            def record_stage(self):
                now = time.monotonic()
                result["stages"].append(
                    {
                        "name": self.current_label,
                        "elapsed_seconds": now - self.stage_started,
                        "started_monotonic_seconds": self.stage_started,
                        "ended_monotonic_seconds": now,
                        "torch_memory_after_stage": self.torch_memory(),
                        "native_resources_after_stage": self.native_resources(),
                    }
                )

            def torch_memory(self):
                # Observation only: never reset peaks, synchronize or evict the allocator.
                return {
                    "allocated_bytes": torch.cuda.memory_allocated(request["gpu_id"]),
                    "reserved_bytes": torch.cuda.memory_reserved(request["gpu_id"]),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                        request["gpu_id"]
                    ),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                        request["gpu_id"]
                    ),
                    "scope": "Torch allocator only; ORT arenas and other GPU processes are excluded",
                }

            def native_resources(self):
                pool = window.video_processor.worker_pool_manager

                def worker_info(worker):
                    if worker is None:
                        return None
                    native_id = getattr(worker, "native_id", None)
                    if native_id is not None:
                        self.seen_worker_native_ids.add(native_id)
                    return {
                        "name": worker.name,
                        "ident": worker.ident,
                        "native_id": native_id,
                        "alive": worker.is_alive(),
                    }

                sessions = {
                    name: {
                        "object_id": id(session),
                        "provider_options": session.get_provider_options()
                        if hasattr(session, "get_provider_options")
                        else None,
                        "providers": session.get_providers()
                        if hasattr(session, "get_providers")
                        else None,
                    }
                    for name, session in window.models_processor.models.items()
                    if session is not None
                }
                workers = [worker_info(worker) for worker in pool.worker_threads]
                single = worker_info(pool.current_single_frame_worker)
                status = Path("/proc/self/status")
                os_threads = (
                    next(
                        (
                            line.split()[1]
                            for line in status.read_text().splitlines()
                            if line.startswith("Threads:")
                        ),
                        None,
                    )
                    if status.is_file()
                    else None
                )
                return {
                    "tail_force_finalize_due_to_stall": window.video_processor.tail_force_finalize_due_to_stall,
                    "cuda_arena_shrink_configured": getattr(
                        window.function_worker, "_cuda_arena_shrink", None
                    ),
                    "loaded_model_count": len(sessions),
                    "loaded_model_sessions": sessions,
                    "pool_workers": workers,
                    "single_frame_worker": single,
                    "shared_cuda_stream_count": len(pool.worker_streams),
                    "seen_worker_native_id_count": len(self.seen_worker_native_ids),
                    "python_thread_count": threading.active_count(),
                    "os_thread_count": int(os_threads) if os_threads else None,
                }

            def begin_session(self):
                self.session_started = time.monotonic()
                self.session_deadline = (
                    self.session_started + request["session_minutes"] * 60
                )
                self.session_iteration = 0
                result["interactive_session"] = {
                    "status": "running",
                    "requested_minutes": request["session_minutes"],
                    "started_monotonic_seconds": self.session_started,
                    "initial_torch_memory": self.torch_memory(),
                    "scope": "Visible native seek/Play loop; client transport and live audio remain unmeasured",
                }
                self.session_continue()

            def session_continue(self):
                now = time.monotonic()
                session = result["interactive_session"]
                extension = run_dir / "session-extension.json"
                if extension.is_file() and not session.get("extension"):
                    minutes = float(
                        json.loads(extension.read_text())["session_minutes"]
                    )
                    if not request["session_minutes"] <= minutes <= 60:
                        raise ValueError(
                            "Session extension must be between the initial duration and 60 minutes"
                        )
                    session["extension"] = {
                        "effective_minutes": minutes,
                        "accepted_elapsed_seconds": now - self.session_started,
                        "request_file_sha256": sha256(extension),
                    }
                    self.session_deadline = self.session_started + minutes * 60
                session["completed_cycles"] = self.session_iteration
                session["elapsed_seconds"] = now - self.session_started
                session["latest_torch_memory"] = self.torch_memory()
                session["native_resources"] = self.native_resources()
                if now >= self.session_deadline:
                    session["status"] = "completed_native_loop"
                    session["ended_monotonic_seconds"] = now
                else:
                    index = self.session_iteration
                    sequence = session_steps(
                        native["seek_frames"],
                        index,
                        self.seek,
                        self.play,
                        self.session_continue,
                    )
                    self.steps.extendleft(reversed(sequence))
                    self.session_iteration += 1
                # Persist incremental evidence in case a later native OOM/stall ends the loop.
                write_json(run_dir / "native-session-progress.json", session)
                self.done()

            def advance(self):
                if self.failure or self.finished:
                    return
                self.pending = None
                if not self.steps:
                    return
                self.current_label, action = self.steps.popleft()
                self.stage_started = time.monotonic()
                print(f"[NATIVE VALIDATION] {self.current_label}", flush=True)
                try:
                    action()
                except Exception as error:
                    self.fail(
                        f"{self.current_label}: {error}\n{traceback.format_exc()}"
                    )

            def done(self):
                self.record_stage()
                self.pending = None
                QtCore.QTimer.singleShot(0, self.advance)

            def tick(self):
                if self.finished:
                    return
                stop_path = run_dir / "stability-stop-request.json"
                if stop_path.is_file():
                    stop = json.loads(stop_path.read_text())
                    self.fail(stop["reason"])
                    return
                if time.monotonic() - self.stage_started > request["stage_timeout"]:
                    self.fail(f"Timeout during {self.current_label}")
                if self.pending == "record":
                    self.observe_encoder()
                # Observe unexpected dialogs. Do not accept destructive confirmations.
                for widget in app.topLevelWidgets():
                    if isinstance(widget, QtWidgets.QMessageBox) and widget.isVisible():
                        message = {
                            "stage": self.current_label,
                            "title": widget.windowTitle(),
                            "text": widget.text(),
                        }
                        if message not in result["dialogs"]:
                            result["dialogs"].append(message)
                        if (
                            request.get("fault_probe") == "missing-frame"
                            and message["title"] == "Recording Error"
                            and f"required frame {self.fault_probe['missing_frame']}"
                            in message["text"]
                        ):
                            widget.accept()
                            continue
                        self.fail(
                            f"Native dialog requires inspection: {message['title']}: {message['text']}"
                        )

            def fail(self, message):
                if self.failure or self.finished:
                    return
                self.finish_order_probe()
                self.finish_fault_probe()
                self.failure = str(message)
                result["error"] = self.failure
                result["failed_stage"] = self.current_label
                result["failed_observer"] = {
                    "pending": self.pending,
                    "pending_frame": self.pending_frame,
                    "processed_frame_received": self.processed_frame,
                    "preview_requested_generation": self.preview_generation,
                    "preview_signals": self.preview_signals,
                    "preview_paints": self.preview_paints,
                    "native_active_generation": window.video_processor.worker_pool_manager.active_single_frame_request_generation,
                    "native_next_frame_to_display": window.video_processor.next_frame_to_display,
                    "slider_value": window.videoSeekSlider.value(),
                    "window_visible": window.isVisible(),
                    "viewport_visible": window.graphicsViewFrame.viewport().isVisible(),
                }
                session = result.get("interactive_session")
                if session and session["status"] == "running":
                    session["status"] = "failed"
                    session["error"] = self.failure
                    session["elapsed_seconds"] = time.monotonic() - self.session_started
                    write_json(run_dir / "native-session-progress.json", session)
                write_json(run_dir / "native-results.json", result)
                self.finished = True
                self.timer.stop()

                # Native teardown may block on a broken encoder/worker; parent hard deadline remains authoritative.
                def stop_failed_native():
                    if (run_dir / "stability-stop-request.json").is_file():
                        # Stop only this failed native trial using its ordinary worker cleanup.
                        window.video_processor.stop_processing()
                    app.exit(1)

                QtCore.QTimer.singleShot(0, stop_failed_native)

            def model_build_started(self, title, text):
                self.model_build_stack.append(
                    (time.monotonic(), self.current_label, title, text)
                )

            def model_build_stopped(self):
                if self.model_build_stack:
                    begin, stage, title, text = self.model_build_stack.pop()
                    result.setdefault("native_model_build_dialogs", []).append(
                        {
                            "stage": stage,
                            "title": title,
                            "text": text,
                            "elapsed_seconds": time.monotonic() - begin,
                            "scope": "native build dialog lifetime including probe/shape preparation",
                        }
                    )

            def load(self):
                self.pending = "load"
                save_load_actions.load_saved_workspace(window, preparation["workspace"])
                processor = window.video_processor
                if (
                    not window.target_videos
                    or not processor.media_path
                    or processor.file_type != "video"
                ):
                    raise RuntimeError(
                        "Project must select an existing video, with real source identities"
                    )
                self.source = video_metadata(ffprobe(Path(processor.media_path)))
                result["input_video"] = {
                    "path": processor.media_path,
                    "sha256": sha256(Path(processor.media_path)),
                    "metadata": self.source,
                }
                if (
                    native.get("require_source_audio", True)
                    and not self.source["audio"]
                ):
                    raise RuntimeError(
                        "Audio validation requires a representative source video with audio"
                    )
                if (
                    window.control.get("ProvidersPrioritySelection")
                    != requested_provider
                    or window.models_processor.provider_name != requested_provider
                ):
                    raise RuntimeError(
                        "Native provider fell back during workspace load"
                    )
                self.done()

            def set_widget(self, key, value):
                widget = window.parameter_widgets.get(key)
                if widget is None:
                    raise ValueError(f"Profile names unknown native widget {key}")
                if (
                    isinstance(widget, QtWidgets.QComboBox)
                    and widget.findText(str(value)) < 0
                    and widget.findData(value) < 0
                ):
                    raise ValueError(f"Native widget {key} has no selection {value!r}")
                widget.set_value(value)

            def controls(self):
                for key, value in profile.get("control", {}).items():
                    self.set_widget(key, value)
                for key, value in (
                    ("ProvidersPrioritySelection", requested_provider),
                    ("nThreadsSlider", settings["threads"]),
                    ("nStreamSlider", settings["streams"]),
                    ("HDREncodeToggle", settings["hdr"]),
                ):
                    self.set_widget(key, value)
                if window.target_faces:
                    next(iter(window.target_faces.values())).click()
                for key, value in profile["parameters"].items():
                    self.set_widget(key, value)
                if window.models_processor.provider_name != requested_provider:
                    raise RuntimeError("Provider does not match explicit profile")
                self.done()

            def seek(self, frame, phase, repeat):
                if not 0 <= int(frame) <= window.video_processor.max_frame_number:
                    raise ValueError(f"Seek outside selected source: {frame}")
                self.pending = "navigation"
                self.pending_frame = int(frame)
                self.processed_frame = False
                self.preview_generation = None
                self.preview_signals = []
                self.preview_paints = []
                self.processed_token = None
                self.processed_pixmap_key = None
                self.preview_info = {
                    "frame": int(frame),
                    "phase": phase,
                    "repeat": repeat,
                }
                # Native slider signals include marker application and the native worker request.
                window.videoSeekSlider.sliderPressed.emit()
                window.videoSeekSlider.setValue(int(frame))
                self.preview_trigger = time.monotonic()
                self.pending = "preview"
                window.videoSeekSlider.sliderReleased.emit()
                self.preview_generation = window.video_processor.worker_pool_manager.active_single_frame_request_generation

            @QtCore.Slot(int, int, object, object)
            def single_processed(self, generation, frame, pixels, cache):
                if self.pending == "preview" and len(self.preview_signals) < 20:
                    self.preview_signals.append(
                        {
                            "generation": generation,
                            "frame": frame,
                            "native_active_generation": window.video_processor.worker_pool_manager.active_single_frame_request_generation,
                            "native_next_frame_to_display": window.video_processor.next_frame_to_display,
                            "elapsed_seconds": time.monotonic() - self.preview_trigger,
                        }
                    )
                if self.pending != "preview" or frame != self.pending_frame:
                    return
                active = window.video_processor.worker_pool_manager.active_single_frame_request_generation
                if generation != 0 and generation != active:
                    return
                # Native display_current_frame is connected first and has applied this frame.
                self.processed_frame = True
                self.processed_token = id(pixels)
                self.processed_pixmap_key = self.current_pixmap_key()
                self.preview_info["processed_signal_seconds"] = (
                    time.monotonic() - self.preview_trigger
                )
                # A native display slot can paint synchronously before our queued observer.
                if window.video_processor.current_frame is pixels:
                    paint = matching_processed_paint(
                        self.preview_paints,
                        self.processed_token,
                        self.processed_pixmap_key,
                        self.pending_frame,
                    )
                    if paint:
                        self.finish_preview(paint)

            def current_pixmap_key(self):
                for item in window.graphicsViewFrame.scene().items():
                    if isinstance(item, QtWidgets.QGraphicsPixmapItem):
                        return int(item.pixmap().cacheKey())
                return None

            def finish_preview(self, paint):
                self.preview_info["native_viewport_paint_seconds"] = paint[
                    "elapsed_seconds"
                ]
                self.preview_info["completion_observer_after_paint"] = (
                    self.preview_info["processed_signal_seconds"]
                    > paint["elapsed_seconds"]
                )
                result["preview"].append(self.preview_info)
                self.pending = "paint-completed"
                QtCore.QTimer.singleShot(0, self.done)

            def eventFilter(self, obj, event):
                if event.type() == QtCore.QEvent.Paint:
                    now = time.monotonic()
                    if self.pending == "preview" and len(self.preview_paints) < 20:
                        self.preview_paints.append(
                            {
                                "elapsed_seconds": now - self.preview_trigger,
                                "processed_frame_received": self.processed_frame,
                                "slider_value": window.videoSeekSlider.value(),
                                "native_frame_token": id(
                                    window.video_processor.current_frame
                                ),
                                "pixmap_key": self.current_pixmap_key(),
                            }
                        )
                    if self.pending == "preview" and self.processed_frame:
                        paint = matching_processed_paint(
                            self.preview_paints,
                            self.processed_token,
                            self.processed_pixmap_key,
                            self.pending_frame,
                        )
                        if paint:
                            self.finish_preview(paint)
                    elif self.pending == "play":
                        frame = window.videoSeekSlider.value()
                        if frame is not None and frame != self.last_paint_frame:
                            self.play_paints.append(
                                {"frame": frame, "monotonic_seconds": now}
                            )
                            self.last_paint_frame = frame
                return False

            def find_faces(self, frame, phase):
                begin = time.monotonic()
                before = len(window.target_faces)
                window.findTargetFacesButton.click()  # Native synchronous detection/recognition action.
                result["analysis"].append(
                    {
                        "frame": frame,
                        "phase": phase,
                        "elapsed_seconds": time.monotonic() - begin,
                        "faces_before": before,
                        "faces_after": len(window.target_faces),
                    }
                )
                self.done()

            def assignments(self):
                if not window.target_faces:
                    raise RuntimeError(
                        "Native Find Faces produced no target faces across the specified analysis frames"
                    )
                for assignment in profile.get("assignments", []):
                    if "target_face_id" in assignment:
                        target = window.target_faces[assignment["target_face_id"]]
                    else:
                        target = list(window.target_faces.values())[
                            assignment.get("target_index", 0)
                        ]
                    target.click()
                    for key, collection in (
                        ("source_face_ids", window.input_faces),
                        ("merged_embedding_ids", window.merged_embeddings),
                    ):
                        for identifier in assignment.get(key, []):
                            button = collection[identifier]
                            if not button.isChecked():
                                button.click()
                if native.get("create_merged_embedding", False):
                    selected = [
                        window.input_faces[key]
                        for key in native.get("merge_source_ids", [])
                    ]
                    if not selected:
                        selected = list(window.input_faces.values())
                    if not selected:
                        raise RuntimeError(
                            "Creating a native merged embedding requires source faces"
                        )
                    dialog = CreateEmbeddingDialog(window, selected)
                    dialog.embed_name_edit.setText("Cloud validation disposable merge")
                    dialog.merge_type_selection.setCurrentText(
                        native.get("merge_type", "Mean")
                    )
                    dialog.include_kv_checkbox.setChecked(
                        native.get("generate_reference_kv", False)
                    )
                    before = set(window.merged_embeddings)
                    dialog.show()
                    dialog.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).click()
                    new = set(window.merged_embeddings) - before
                    if not new:
                        raise RuntimeError(
                            "Native merged embedding dialog did not create an embedding"
                        )
                    next(iter(window.target_faces.values())).click()
                    window.merged_embeddings[next(iter(new))].click()
                if native["require_assignments"] and not any(
                    face.assigned_input_faces or face.assigned_merged_embeddings
                    for face in window.target_faces.values()
                ):
                    raise RuntimeError(
                        "No native assignments. Supply a prepared project or explicit [[assignments]]"
                    )
                if native["require_merged_embedding"] and not window.merged_embeddings:
                    raise RuntimeError(
                        "Project does not exercise a merged embedding; supply one or create_merged_embedding=true"
                    )
                if native["require_reference_kv"] and not any(
                    getattr(face, "assigned_kv_map", None) is not None
                    for face in window.target_faces.values()
                ):
                    raise RuntimeError(
                        "Requested external reference KV validation has no assigned KV data"
                    )
                for face in window.target_faces.values():
                    face.click()
                    for key, value in profile["parameters"].items():
                        self.set_widget(key, value)
                assigned = [
                    str(identifier)
                    for identifier, face in window.target_faces.items()
                    if face.assigned_input_faces or face.assigned_merged_embeddings
                ]
                result["assignment_scope"] = {
                    "detected_target_faces": len(window.target_faces),
                    "assigned_target_face_ids": assigned,
                    "assigned_target_face_count": len(assigned),
                    "source_image_count": len(window.input_faces),
                    "merged_embedding_count": len(window.merged_embeddings),
                }
                # A prepared workspace may already exercise markers/ranges. Optional explicit actions add them to the clone.
                for frame in native.get("add_parameter_markers", []):
                    window.videoSeekSlider.setValue(int(frame))
                    video_control_actions.add_video_slider_marker(window)
                if native.get("segments"):
                    if window.job_marker_pairs:
                        raise ValueError(
                            "Use workspace segment pairs OR native.segments; existing ranges are not silently discarded"
                        )
                    for start, end in native["segments"]:
                        window.videoSeekSlider.setValue(start)
                        video_control_actions.set_job_start_frame(window)
                        window.videoSeekSlider.setValue(end)
                        video_control_actions.set_job_end_frame(window)
                if native.get("require_markers", True) and not window.markers:
                    raise RuntimeError(
                        "Marker validation requires native parameter markers in the project/profile"
                    )
                if native.get("require_segments", True) and not window.job_marker_pairs:
                    raise RuntimeError(
                        "Segment validation requires native record marker pairs in the project/profile"
                    )
                if not window.swapfacesButton.isChecked():
                    window.swapfacesButton.click()
                self.done()

            def restore_check(self):
                expected = json.loads(
                    Path(request["preparation"]["workspace"]).read_text()
                )
                saved = run_dir / "fresh-process-restored.json"
                save_load_actions.save_current_workspace(window, str(saved))
                actual = json.loads(saved.read_text())
                before, after = state_signature(expected), state_signature(actual)
                changed = [key for key in STATE_FIELDS if before[key] != after[key]]
                assets = []
                for item in request["preparation"]["files"]:
                    if item.get("kind") != "workspace-input":
                        continue
                    original_digest = sha256(Path(item["path"]))
                    clone_digest = sha256(Path(item["clone"]))
                    assets.append(
                        {
                            **item,
                            "original_sha256_after": original_digest,
                            "clone_sha256_after": clone_digest,
                        }
                    )
                    if (
                        original_digest != item["sha256"]
                        or clone_digest != item["sha256"]
                    ):
                        changed.append("external_asset_hash:" + item["path"])
                result["fresh_process_restore"] = {
                    "fields_compared": list(STATE_FIELDS),
                    "changed_fields": changed,
                    "expected_signature": before,
                    "restored_signature": after,
                    "external_assets": assets,
                    "saved_workspace": str(saved),
                    "runtime_identity": self.runtime_signature(),
                    "path_rebasing": "Only isolated input/KV paths, output folder and remembered media directories change during preparation; profile settings are preserved",
                }
                write_json(
                    run_dir / "fresh-process-restore-proof.json",
                    result["fresh_process_restore"],
                )
                if changed:
                    raise RuntimeError(
                        f"Fresh-process native restore changed state/assets: {changed}"
                    )
                self.done()

            def roundtrip(self):
                before_path = run_dir / f"{self.current_label}-before.json"
                after_path = run_dir / f"{self.current_label}-after.json"
                runtime_before = self.runtime_signature()
                save_load_actions.save_current_workspace(window, str(before_path))
                before = json.loads(before_path.read_text())
                save_load_actions.load_saved_workspace(window, str(before_path))
                save_load_actions.save_current_workspace(window, str(after_path))
                after = json.loads(after_path.read_text())
                old, new = state_signature(before), state_signature(after)
                changed = [key for key in STATE_FIELDS if old[key] != new[key]]
                runtime_after = self.runtime_signature()
                if runtime_before != runtime_after:
                    changed.append("runtime_assignments_embeddings_or_kv")
                result.setdefault("roundtrips", []).append(
                    {
                        "before": str(before_path),
                        "after": str(after_path),
                        "changed_fields": changed,
                        "runtime_before": runtime_before,
                        "runtime_after": runtime_after,
                    }
                )
                if changed:
                    raise RuntimeError(
                        f"Native save/load changed semantic state: {changed}"
                    )
                result["effective_workspace_sha256"] = sha256(after_path)
                result["effective_state"] = new
                self.done()

            def runtime_signature(self):
                def tensor_identity(value):
                    if isinstance(value, dict):
                        return {
                            str(key): tensor_identity(item)
                            for key, item in value.items()
                        }
                    if isinstance(value, (list, tuple)):
                        return [tensor_identity(item) for item in value]
                    if isinstance(value, torch.Tensor):
                        buffer = (
                            value.detach()
                            .cpu()
                            .contiguous()
                            .view(torch.uint8)
                            .numpy()
                            .tobytes()
                        )
                        return {
                            "sha256": hashlib.sha256(buffer).hexdigest(),
                            "dtype": str(value.dtype),
                            "shape": list(value.shape),
                        }
                    if hasattr(value, "tobytes") and hasattr(value, "shape"):
                        # Native JSON restoration can promote FP32 embedding arrays to FP64.
                        # Compare numerical identity at the same 1e-6 tolerance as persisted JSON.
                        import numpy

                        values = numpy.round(value.astype("float64"), 6).tolist()
                        return {
                            "sha256": hashlib.sha256(
                                json.dumps(values, sort_keys=True).encode()
                            ).hexdigest(),
                            "shape": list(value.shape),
                        }
                    return value

                return {
                    str(identifier): {
                        "input_faces": sorted(face.assigned_input_faces),
                        "merged_embeddings": sorted(face.assigned_merged_embeddings),
                        "assigned_embedding": tensor_identity(
                            face.assigned_input_embedding
                        ),
                        "assigned_kv": tensor_identity(
                            getattr(face, "assigned_kv_map", None)
                        ),
                    }
                    for identifier, face in window.target_faces.items()
                }

            def play(self, repeat, phase="warm"):
                self.pending = "play"
                self.play_paints = []
                self.last_paint_frame = None
                self.play_repeat = repeat
                self.play_phase = phase
                self.play_start = time.monotonic()
                self.play_started_signal = False
                window.buttonMediaPlay.click()
                QtCore.QTimer.singleShot(
                    int(native["play_seconds"] * 1000),
                    lambda r=repeat: self.stop_play(r),
                )

            @QtCore.Slot()
            def processing_started(self):
                if self.pending == "record":
                    self.record_started = True
                    self.record_start_signal = time.monotonic()
                    self.observe_encoder()
                elif self.pending == "play":
                    self.play_started_signal = True

            def stop_play(self, repeat):
                if self.pending != "play" or self.play_repeat != repeat:
                    return
                if window.buttonMediaPlay.isChecked():
                    window.buttonMediaPlay.click()

            @QtCore.Slot()
            def processing_stopped(self):
                if self.pending == "play":
                    elapsed = time.monotonic() - self.play_start
                    intervals = [
                        b["monotonic_seconds"] - a["monotonic_seconds"]
                        for a, b in zip(self.play_paints, self.play_paints[1:])
                    ]
                    result["playback"].append(
                        {
                            "repeat": self.play_repeat,
                            "phase": self.play_phase,
                            "elapsed_seconds": elapsed,
                            "native_viewport_distinct_frames": len(self.play_paints),
                            "paint_events": self.play_paints,
                            "native_paint_interval_median_seconds": percentile(
                                intervals, 0.5
                            ),
                            "native_paint_interval_p95_seconds": percentile(
                                intervals, 0.95
                            ),
                            "native_local_paint_fps": (len(self.play_paints) - 1)
                            / sum(intervals)
                            if intervals and sum(intervals)
                            else None,
                        }
                    )
                    if not self.play_started_signal or len(self.play_paints) < 2:
                        self.fail(
                            "Native processed playback did not start and present multiple frames"
                        )
                        return
                    self.done()
                elif self.pending == "record":
                    # Native default and segment finalize both emit this signal. Independently prove success.
                    try:
                        if not self.record_started:
                            raise RuntimeError(
                                "Record completion occurred without native processing_started signal"
                            )
                        self.observe_encoder()
                        elapsed = time.monotonic() - self.record_trigger
                        torch_memory_after_record = self.torch_memory()
                        self.finish_order_probe()
                        if request.get("fault_probe") == "missing-frame":
                            self.validate_missing_abort()
                            self.done()
                            return
                        self.finish_fault_probe()
                        files = {
                            path.resolve()
                            for path in (run_dir / "outputs").rglob("*")
                            if path.is_file()
                        }
                        candidates = files - self.record_before
                        videos = [
                            path
                            for path in candidates
                            if path.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi"}
                        ]
                        if not videos:
                            raise RuntimeError(
                                "Native Record stopped without a completed output (possible unsupported NVENC/encode failure)"
                            )
                        proofs = [
                            validate_output(
                                path,
                                self.source,
                                self.expected_frames,
                                self.expected_duration,
                                settings["hdr"],
                                self.expected_size,
                            )
                            for path in videos
                        ]
                        result["records"].append(
                            {
                                "repeat": self.record_repeat,
                                "elapsed_seconds": elapsed,
                                "torch_memory_after_record": torch_memory_after_record,
                                "native_start_signal_delay_seconds": self.record_start_signal
                                - self.record_trigger,
                                "verified_output_fps": None
                                if request.get("out_of_order_probe")
                                or request.get("fault_probe")
                                else sum(
                                    item["metadata"]["frames"] or 0 for item in proofs
                                )
                                / elapsed,
                                "outputs": proofs,
                                "native_skipped_frames": window.video_processor.total_skipped_frames,
                                "observed_ffmpeg_encoders": self.encoder_evidence(),
                                "final_mux_exit": "Native finalize does not expose its subprocess.run return code; completed output is independently verified",
                            }
                        )
                        write_json(
                            run_dir / "native-record-progress.json", result["records"]
                        )
                        failed_encoders = [
                            item
                            for item in self.encoder_evidence()
                            if item["exit_code"] not in (0,)
                        ]
                        if failed_encoders:
                            raise RuntimeError(
                                f"Observed native FFmpeg encoder did not exit cleanly: {failed_encoders}"
                            )
                        self.validate_fault_output(proofs)
                        if not all(item["valid"] for item in proofs):
                            raise RuntimeError(
                                f"Native output verification failed: {[item['errors'] for item in proofs]}"
                            )
                        self.validate_order_probe(proofs)
                        self.done()
                    except Exception as error:
                        self.fail(str(error))

            @QtCore.Slot(int, object)
            def order_signal_emitted(self, frame_number, pixels):
                if self.order_probe and frame_number in self.order_probe["frames"]:
                    self.order_probe["signals_emitted"].append(
                        {"frame": frame_number, "monotonic_seconds": time.monotonic()}
                    )
                    if frame_number == self.order_probe["highest_frame"]:
                        self.order_highest_emitted.set()

            @QtCore.Slot(int, object)
            def order_signal_delivered(self, frame_number, pixels):
                if self.order_probe and frame_number in self.order_probe["frames"]:
                    self.order_probe["signals_delivered"].append(
                        {"frame": frame_number, "monotonic_seconds": time.monotonic()}
                    )

            def begin_order_probe(self):
                self.order_highest_emitted.clear()
                if window.job_marker_pairs or window.dropped_frames:
                    raise ValueError(
                        "Ordering diagnostic requires a full clip without segments/dropped frames"
                    )
                from app.processors.workers.frame_worker import FrameWorker

                highest = window.video_processor.max_frame_number
                if highest < 2:
                    raise ValueError(
                        "Ordering diagnostic needs at least three source frames"
                    )
                self.order_probe = {
                    "scope": "Untimed diagnostic with injected worker delay; no throughput claim",
                    "frames": [highest - 1, highest],
                    "highest_frame": highest,
                    "signals_emitted": [],
                    "signals_delivered": [],
                    "encoder_frame_indices": [],
                    "natural_eof_observations": [],
                    "delay_events": [],
                }
                self.order_original_process_frame = FrameWorker.process_frame
                original = self.order_original_process_frame

                def delayed_process(worker, control, stop_event):
                    pixels = original(worker, control, stop_event)
                    if (
                        worker.is_pool_worker
                        and worker.frame_number == highest - 1
                        and window.video_processor.recording
                    ):
                        self.order_probe["delay_events"].append(
                            {
                                "event": "penultimate_processed_delay_begin",
                                "monotonic_seconds": time.monotonic(),
                            }
                        )
                        observed = self.order_highest_emitted.wait(timeout=5)
                        time.sleep(0.25)
                        self.order_probe["delay_events"].append(
                            {
                                "event": "penultimate_delay_end",
                                "highest_native_signal_observed": observed,
                                "monotonic_seconds": time.monotonic(),
                            }
                        )
                    return pixels

                FrameWorker.process_frame = delayed_process
                self.install_writer_trace(self.order_probe)

            def install_writer_trace(self, proof):
                self.order_previous_trace = sys.gettrace()
                last_written = 0
                last_eof = None

                def local_trace(frame, event, arg):
                    nonlocal last_written, last_eof
                    if event != "line":
                        return local_trace
                    values = frame.f_locals
                    if (
                        values.get("hit_eof")
                        and "pending_tasks" in values
                        and "is_drain_complete" in values
                    ):
                        state = (
                            values["pending_tasks"],
                            values["is_drain_complete"],
                            window.video_processor.next_frame_to_display,
                            window.video_processor.frames_written,
                        )
                        if state != last_eof:
                            proof["natural_eof_observations"].append(
                                {
                                    "pending_tasks": state[0],
                                    "is_drain_complete": state[1],
                                    "next_frame_to_display": state[2],
                                    "frames_written": state[3],
                                    "monotonic_seconds": time.monotonic(),
                                }
                            )
                            last_eof = state
                    written = window.video_processor.frames_written
                    if written < last_written:
                        # Native segment setup resets the counter between encoders.
                        last_written = written
                    if written > last_written and "frame_number_to_display" in values:
                        proof["encoder_frame_indices"].append(
                            values["frame_number_to_display"]
                        )
                        last_written = written
                    return local_trace

                def global_trace(frame, event, arg):
                    if (
                        event == "call"
                        and frame.f_code.co_name == "display_next_frame"
                        and frame.f_code.co_filename.endswith(
                            "video_utils/media_pipeline.py"
                        )
                    ):
                        return local_trace
                    return None

                sys.settrace(global_trace)

            def finish_order_probe(self):
                if self.order_original_process_frame is not None:
                    from app.processors.workers.frame_worker import FrameWorker

                    FrameWorker.process_frame = self.order_original_process_frame
                    self.order_original_process_frame = None
                    sys.settrace(self.order_previous_trace)
                    result["out_of_order_probe"] = self.order_probe
                    write_json(
                        run_dir / "untimed-ordering-proof.json", self.order_probe
                    )

            def validate_order_probe(self, proofs):
                if not self.order_probe:
                    return
                errors = out_of_order_errors(self.order_probe, proofs)
                self.order_probe["errors"] = errors
                self.order_probe["valid"] = not errors
                write_json(run_dir / "untimed-ordering-proof.json", self.order_probe)
                if errors:
                    raise RuntimeError("; ".join(errors))

            def begin_fault_probe(self):
                from app.processors.workers.frame_worker import FrameWorker
                from app.processors.video_utils import media_pipeline

                vp = window.video_processor
                mode = request["fault_probe"]
                highest = vp.max_frame_number
                ranges = window.job_marker_pairs or [(0, highest)]
                expected = [
                    frame
                    for start, end in ranges
                    for frame in range(start, end + 1)
                    if frame not in window.dropped_frames
                ]
                self.fault_probe = {
                    "kind": mode,
                    "scope": "Untimed disposable-process native fault injection; no throughput claim",
                    "expected_encoder_frame_indices": expected,
                    "encoder_frame_indices": [],
                    "natural_eof_observations": [],
                    "events": [],
                    "preserved_output": request.get("preserved_output"),
                }
                proof = self.fault_probe

                def replace(obj, name, method):
                    self.fault_originals.append((obj, name, getattr(obj, name)))
                    setattr(obj, name, method)

                if mode == "known-skips":
                    if not window.dropped_frames:
                        raise ValueError(
                            "known-skips diagnostic requires deliberate dropped_frames in the supplied project"
                        )
                    proof["configured_dropped_frames"] = sorted(window.dropped_frames)
                else:
                    if window.job_marker_pairs or window.dropped_frames:
                        raise ValueError(
                            "Injected EOF/missing-frame diagnostic requires an untrimmed full clip"
                        )
                    cutoff = highest - 1
                    if cutoff < 2:
                        raise ValueError(
                            "Fault diagnostic requires at least four frames"
                        )
                    if mode == "missing-frame":
                        proof["missing_frame"] = cutoff
                        original_emit = FrameWorker.process_and_emit_task

                        def suppress_one(worker):
                            if (
                                worker.is_pool_worker
                                and vp.recording
                                and worker.frame_number == cutoff
                            ):
                                proof["events"].append(
                                    {
                                        "event": "required_frame_emit_suppressed",
                                        "frame": cutoff,
                                        "monotonic_seconds": time.monotonic(),
                                    }
                                )
                                return None
                            return original_emit(worker)

                        replace(FrameWorker, "process_and_emit_task", suppress_one)
                    else:
                        held = threading.Event()
                        eof = threading.Event()
                        proof["unavailable_tail"] = list(range(cutoff, highest + 1))
                        proof["expected_encoder_frame_indices"] = list(range(cutoff))
                        self.expected_frames = cutoff
                        self.expected_duration = cutoff / float(vp.fps)
                        original_process = FrameWorker.process_frame

                        def hold_previous(worker, control, stop_event):
                            pixels = original_process(worker, control, stop_event)
                            if (
                                worker.is_pool_worker
                                and vp.recording
                                and worker.frame_number == cutoff - 1
                            ):
                                held.set()
                                observed = eof.wait(timeout=5)
                                proof["events"].append(
                                    {
                                        "event": "held_worker_released",
                                        "frame": worker.frame_number,
                                        "eof_observed": observed,
                                        "monotonic_seconds": time.monotonic(),
                                    }
                                )
                                time.sleep(0.25)
                            return pixels

                        replace(FrameWorker, "process_frame", hold_previous)

                        def read_wrapper(original):
                            def read(*arguments, **keywords):
                                if (
                                    threading.current_thread() is vp.feeder_thread
                                    and vp.recording
                                    and vp.current_frame_number == cutoff
                                ):
                                    observed = held.wait(timeout=5)
                                    proof["events"].append(
                                        {
                                            "event": "early_eof_injected",
                                            "frame": cutoff,
                                            "held_worker_observed": observed,
                                            "consumer_cursor": vp.next_frame_to_display,
                                            "buffered_frames": sorted(
                                                vp.frames_to_display
                                            ),
                                            "pending_tasks": vp.worker_pool_manager.frame_queue.unfinished_tasks,
                                            "monotonic_seconds": time.monotonic(),
                                        }
                                    )
                                    eof.set()
                                    return False, None
                                return original(*arguments, **keywords)

                            return read

                        replace(
                            media_pipeline.misc_helpers,
                            "read_frame",
                            read_wrapper(media_pipeline.misc_helpers.read_frame),
                        )
                        replace(
                            vp,
                            "_read_frame_from_ffmpeg_input_stream",
                            read_wrapper(vp._read_frame_from_ffmpeg_input_stream),
                        )
                        original_mark = vp.media_pipeline._mark_unavailable_tail

                        def mark_tail(start, end):
                            before = vp.next_frame_to_display
                            original_mark(start, end)
                            proof["events"].append(
                                {
                                    "event": "native_unavailable_tail_marked",
                                    "range": [start, end],
                                    "consumer_cursor_before": before,
                                    "consumer_cursor_after": vp.next_frame_to_display,
                                    "skipped_frames": sorted(
                                        vp.media_pipeline.skipped_frames
                                    ),
                                    "monotonic_seconds": time.monotonic(),
                                }
                            )

                        replace(vp.media_pipeline, "_mark_unavailable_tail", mark_tail)
                self.install_writer_trace(proof)

            def finish_fault_probe(self):
                if self.fault_originals or self.fault_probe:
                    for obj, name, original in reversed(self.fault_originals):
                        setattr(obj, name, original)
                    self.fault_originals.clear()
                    sys.settrace(self.order_previous_trace)
                    result["fault_probe"] = self.fault_probe
                    write_json(run_dir / "untimed-fault-proof.json", self.fault_probe)

            def preserved_output_errors(self):
                preserved = request.get("preserved_output")
                if not preserved:
                    return []
                return [
                    f"Preserved output changed: {key}"
                    for key in ("original", "clone")
                    if not Path(preserved[key]).is_file()
                    or sha256(Path(preserved[key])) != preserved["sha256"]
                ]

            def validate_fault_output(self, proofs):
                if not self.fault_probe:
                    return
                proof = self.fault_probe
                errors = self.preserved_output_errors()
                if (
                    proof["encoder_frame_indices"]
                    != proof["expected_encoder_frame_indices"]
                ):
                    errors.append(
                        "Native fault output indices differ from required source order"
                    )
                if not all(item["exact_frame_count"] for item in proofs):
                    errors.append(
                        "Native fault output did not preserve the exact expected decoded frame count"
                    )
                for output in proofs:
                    for audio in output["metadata"]["audio"]:
                        duration = audio.get("duration")
                        if (
                            duration is None
                            or abs(float(duration) - self.expected_duration) > 0.15
                        ):
                            errors.append(
                                "Native rebuilt audio duration differs from the shortened video duration"
                            )
                if proof["kind"] == "early-eof":
                    injections = [
                        item
                        for item in proof["events"]
                        if item["event"] == "early_eof_injected"
                    ]
                    marks = [
                        item
                        for item in proof["events"]
                        if item["event"] == "native_unavailable_tail_marked"
                    ]
                    if not injections or not all(
                        item["held_worker_observed"] for item in injections
                    ):
                        errors.append(
                            "EOF while earlier work was held was not established"
                        )
                    if not marks or any(
                        item["consumer_cursor_before"] != item["consumer_cursor_after"]
                        for item in marks
                    ):
                        errors.append(
                            "Native tail marking changed the consumer cursor or was not observed"
                        )
                proof.update({"errors": errors, "valid": not errors, "outputs": proofs})
                write_json(run_dir / "untimed-fault-proof.json", proof)
                if errors:
                    raise RuntimeError("; ".join(errors))

            def validate_missing_abort(self):
                vp = window.video_processor
                proof = self.fault_probe
                proof["native_processing_error"] = vp.last_processing_error
                proof["native_skipped_frames_at_abort"] = sorted(
                    vp.media_pipeline.skipped_frames
                )
                new = {
                    path.resolve()
                    for path in (run_dir / "outputs").rglob("*")
                    if path.is_file()
                } - self.record_before
                proof["new_files_after_abort"] = [str(path) for path in sorted(new)]
                proof["observed_ffmpeg_encoders"] = self.encoder_evidence()
                temporary = [
                    str(path)
                    for path in (runtime / "temp_files").rglob("*.mp4")
                    if path.is_file()
                ]
                proof["temporary_recordings_after_abort"] = temporary
                errors = self.preserved_output_errors()
                missing = proof["missing_frame"]
                if f"required frame {missing}" not in (vp.last_processing_error or ""):
                    errors.append("Native missing-frame timeout abort was not observed")
                if missing in vp.media_pipeline.skipped_frames:
                    errors.append(
                        "The required missing frame was silently converted into a known skip"
                    )
                if proof["encoder_frame_indices"] != list(range(missing)):
                    errors.append(
                        "Native encoder consumed frames after the required missing frame"
                    )
                if new:
                    errors.append(
                        "Native abort retained or published new recording files"
                    )
                if temporary:
                    errors.append("Native abort retained a temporary recording")
                proof.update({"valid": not errors, "errors": errors})
                self.finish_fault_probe()
                self.steps.clear()
                self.steps.append(("complete", self.complete))
                if errors:
                    raise RuntimeError("; ".join(errors))

            def record(self, repeat):
                self.pending = "record"
                self.record_repeat = repeat
                self.record_started = False
                self.encoder_processes = {}
                self.record_before = {
                    path.resolve()
                    for path in (run_dir / "outputs").rglob("*")
                    if path.is_file()
                }
                self.expected_frames, self.expected_duration = expected_recording(
                    window, self.source
                )
                pixels = window.video_processor.current_frame
                self.expected_size = (int(pixels.shape[1]), int(pixels.shape[0]))
                if native.get("expected_width") and native.get("expected_height"):
                    self.expected_size = (
                        native["expected_width"],
                        native["expected_height"],
                    )
                if request.get("out_of_order_probe"):
                    self.begin_order_probe()
                if request.get("fault_probe"):
                    self.begin_fault_probe()
                self.record_trigger = time.monotonic()
                window.buttonMediaRecord.click()  # Original native Record entry point.
                self.observe_encoder()

            def observe_encoder(self):
                process = window.video_processor.encoder.recording_sp
                if process is not None:
                    self.encoder_processes[process.pid] = process

            def encoder_evidence(self):
                evidence = []
                for process in self.encoder_processes.values():
                    arguments = (
                        list(process.args)
                        if isinstance(process.args, (list, tuple))
                        else [str(process.args)]
                    )
                    encoder = (
                        arguments[arguments.index("-c:v") + 1]
                        if "-c:v" in arguments
                        else None
                    )
                    evidence.append(
                        {
                            "pid": process.pid,
                            "arguments": arguments,
                            "selected_video_encoder": encoder,
                            "exit_code": process.poll(),
                            "stderr": "native.log (native encoder inherits stderr)",
                        }
                    )
                return evidence

            def backend_proof(self):
                import onnxruntime

                processor = window.models_processor
                loaded = [
                    name
                    for name, session in processor.models.items()
                    if session is not None
                ]
                result["backend"] = {
                    "requested_provider": requested_provider,
                    "effective_provider": processor.provider_name,
                    "torch_device": str(processor.device),
                    "ort_version": onnxruntime.__version__,
                    "torch_version": torch.__version__,
                    "loaded_sessions": {},
                    "proof_is_separate_from_timed_trials": True,
                    "torch_cuda_runtime": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(),
                    "provider_route": "Native baseline routes both TensorRT labels through ORT TensorrtExecutionProvider",
                }
                # Native public model unload/load methods permit session options. This separate
                # diagnostic pass adds ORT profiling; it does not wrap inference or alter graph settings.
                previous_keep = window.control.get("KeepModelsAliveToggle", False)
                if previous_keep:
                    self.set_widget("KeepModelsAliveToggle", False)
                for name in loaded:
                    session = processor.models[name]
                    entry = {
                        "type": type(session).__name__,
                        "model_path": processor.models_path.get(name),
                    }
                    if entry["model_path"] and Path(entry["model_path"]).is_file():
                        entry["model_sha256"] = sha256(Path(entry["model_path"]))
                    if hasattr(session, "get_providers"):
                        entry["session_providers"] = session.get_providers()
                        entry["provider_options"] = session.get_provider_options()
                        entry["input_types"] = {
                            item.name: item.type for item in session.get_inputs()
                        }
                        processor.unload_model(name, force_immediate=True)
                        options = onnxruntime.SessionOptions()
                        options.enable_profiling = True
                        options.profile_file_prefix = str(run_dir / ("ort-" + name))
                        new_session = processor.load_model(
                            name, session_options=options
                        )
                        if new_session is None:
                            raise RuntimeError(
                                f"Native diagnostic session load failed for {name}"
                            )
                        self.profiler_sessions[name] = new_session
                    result["backend"]["loaded_sessions"][name] = entry
                if previous_keep:
                    self.set_widget("KeepModelsAliveToggle", True)
                activities = [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
                run_options_values = set()
                run_options_calls = 0
                no_shrink_calls = 0

                def observe_run_options(frame, event, arg):
                    nonlocal run_options_calls, no_shrink_calls
                    if event != "call" or frame.f_code.co_name != "run_with_iobinding":
                        return
                    if "onnxruntime" not in frame.f_code.co_filename:
                        return
                    run_options_calls += 1
                    options = frame.f_locals.get("run_options")
                    try:
                        value = (
                            options.get_run_config_entry(
                                "memory.enable_memory_arena_shrinkage"
                            )
                            if options is not None
                            else None
                        )
                    except RuntimeError:
                        value = None
                    if value is None:
                        no_shrink_calls += 1
                    else:
                        run_options_values.add(value)

                previous_profile = sys.getprofile()
                try:
                    # Untimed diagnostic only: observe actual Python method arguments;
                    # never replace ORT methods or add per-frame timing instrumentation.
                    sys.setprofile(observe_run_options)
                    with torch.profiler.profile(activities=activities) as trace:
                        window.findTargetFacesButton.click()
                        window.video_processor.process_current_frame(synchronous=True)
                        torch.cuda.synchronize(request["gpu_id"])
                finally:
                    sys.setprofile(previous_profile)
                result["backend"]["observed_ort_run_options"] = {
                    "scope": "Untimed native diagnostic, actual run_with_iobinding arguments",
                    "calls_observed": run_options_calls,
                    "calls_without_shrink_entry": no_shrink_calls,
                    "arena_shrink_entries": sorted(run_options_values),
                }
                if (
                    getattr(window.function_worker, "_cuda_arena_shrink", False)
                    and f"gpu:{request['gpu_id']}" not in run_options_values
                ):
                    raise RuntimeError(
                        "Configured CUDA arena shrink lacks actual native RunOptions argument evidence"
                    )
                trace.export_chrome_trace(
                    str(run_dir / "torch-native-backend-trace.json")
                )
                cuda_events = [
                    event
                    for event in trace.events()
                    if "cuda" in str(event.device_type).lower()
                ]
                result["backend"]["torch_cuda_event_count"] = len(cuda_events)
                actual_gpu_nodes = 0
                for name, session in self.profiler_sessions.items():
                    trace_path = session.end_profiling()
                    events = json.loads(Path(trace_path).read_text())
                    counts = collections.Counter(
                        event.get("args", {}).get("provider")
                        for event in events
                        if event.get("cat") == "Node"
                        and event.get("args", {}).get("provider")
                    )
                    result["backend"]["loaded_sessions"][name][
                        "executed_node_providers"
                    ] = dict(counts)
                    result["backend"]["loaded_sessions"][name]["trace_path"] = (
                        trace_path
                    )
                    actual_gpu_nodes += (
                        counts["CUDAExecutionProvider"]
                        + counts["TensorrtExecutionProvider"]
                    )
                result["backend"]["executed_gpu_ort_nodes"] = actual_gpu_nodes
                if not cuda_events:
                    raise RuntimeError(
                        "Native Torch trace did not establish CUDA execution"
                    )
                if self.profiler_sessions and not actual_gpu_nodes:
                    raise RuntimeError(
                        "ORT profile did not establish CUDA/TensorRT node execution"
                    )
                from app.processors.models_data import detection_model_mapping

                aliases = {
                    "GhostFace-v1": "GhostFacev1",
                    "GhostFace-v2": "GhostFacev2",
                    "GhostFace-v3": "GhostFacev3",
                    "Yolov11 VR180": "YoloFace11nVR180",
                    "Yolov12 VR180": "YoloFace12nVR180",
                }
                detector = profile["control"]["DetectorModelSelection"]
                swapper = profile["parameters"]["SwapModelSelection"]
                required = {
                    detection_model_mapping.get(
                        detector, aliases.get(detector, detector)
                    ),
                    profile["control"]["RecognitionModelSelection"],
                    aliases.get(swapper, swapper),
                }
                for enabled, model in (
                    ("FaceRestorerEnableToggle", "FaceRestorerTypeSelection"),
                    ("FaceRestorerEnable2Toggle", "FaceRestorerType2Selection"),
                ):
                    if profile["parameters"][enabled]:
                        required.add(
                            window.function_worker.face_restorers.model_map[
                                profile["parameters"][model]
                            ]
                        )
                result["backend"]["required_representative_models"] = sorted(required)
                for name in required:
                    counts = (
                        result["backend"]["loaded_sessions"]
                        .get(name, {})
                        .get("executed_node_providers", {})
                    )
                    if not (
                        counts.get("CUDAExecutionProvider")
                        or counts.get("TensorrtExecutionProvider")
                    ):
                        raise RuntimeError(
                            f"Selected representative model {name} lacks actual GPU node execution evidence"
                        )
                if requested_provider.startswith("TensorRT") and not any(
                    entry.get("executed_node_providers", {}).get(
                        "TensorrtExecutionProvider"
                    )
                    for entry in result["backend"]["loaded_sessions"].values()
                ):
                    raise RuntimeError(
                        "TensorRT requested but only CUDA/CPU node execution was observed"
                    )
                self.done()

            def complete(self):
                warm = [
                    item["native_viewport_paint_seconds"]
                    for item in result["preview"]
                    if item["phase"] == "warm"
                ]
                result["warm_preview_median_seconds"] = percentile(warm, 0.5)
                result["warm_preview_p95_seconds"] = percentile(warm, 0.95)
                result["status"] = "passed"
                result["elapsed_seconds"] = time.monotonic() - started
                write_json(run_dir / "native-results.json", result)
                self.finished = True
                self.timer.stop()
                window.quit_without_saving = True
                window.close()
                app.exit(0)

        driver = Driver()
        exit_code = app.exec()
        if driver.failure:
            return 1
        return exit_code
    except Exception as error:
        result["error"] = f"{error}\n{traceback.format_exc()}"
        result["failed_stage"] = result.get("failed_stage", "native_boot")
        write_json(run_dir / "native-results.json", result)
        print(result["error"], flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
