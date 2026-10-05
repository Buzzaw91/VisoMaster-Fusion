"""Native finalization fault tests using real source and fake FFmpeg processes.

Run with ``python -m unittest discover -s tests/cloud_desktop -p test_encoder_failure.py``.
No Qt, GPU, Torch, or FFmpeg installation is needed.
"""

from __future__ import annotations

import ast
import gc
import io
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock
import uuid


ROOT = Path(__file__).resolve().parents[2]


def _compile_source(path, namespace, names, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(
            node.body for node in nodes if getattr(node, "name", None) == class_name
        )
    selected = [node for node in nodes if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == set(names)
    for node in selected:
        node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + selected,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class _FFmpegProcess:
    """A process already exited, or one whose first wait times out."""

    def __init__(self, exit_code, *, timeout=False):
        self.returncode = None if timeout else exit_code
        self.exit_code = exit_code
        self.timeout = timeout
        self.stdin = io.BytesIO()
        self.terminate = Mock()
        self.kill = Mock()
        self.poll = Mock(return_value=self.returncode)
        self.wait = Mock(side_effect=self._wait)

    def _wait(self, timeout=None):
        if self.timeout:
            self.timeout = False
            raise subprocess.TimeoutExpired("fake ffmpeg", timeout)
        self.returncode = self.exit_code
        return self.exit_code


def _encoder(process=None):
    namespace = {
        "os": os,
        "math": math,
        "json": json,
        "subprocess": subprocess,
        "print": Mock(),
    }
    _compile_source(
        ROOT / "app/processors/video_utils/video_encoding.py",
        namespace,
        {"FFmpegEncoder"},
    )
    encoder = namespace["FFmpegEncoder"]()
    encoder.recording_sp = process
    encoder.is_running = Mock(
        side_effect=AssertionError("finalization must not skip exited encoders")
    )
    return encoder


def _native_functions(output):
    namespace = {
        "os": os,
        "Path": Path,
        "queue": queue,
        "time": time,
        "shutil": shutil,
        "gc": gc,
        "uuid": uuid,
        "print": Mock(),
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        "layout_actions": SimpleNamespace(
            enable_all_parameters_and_control_widget=Mock()
        ),
        "video_control_actions": SimpleNamespace(reset_media_buttons=Mock()),
        "list_view_actions": SimpleNamespace(open_output_media_folder=Mock()),
        "common_widget_actions": SimpleNamespace(create_and_show_toast_message=Mock()),
        "misc_helpers": SimpleNamespace(
            release_capture=Mock(), get_output_file_path=Mock(return_value=str(output))
        ),
        "subprocess": SimpleNamespace(
            run=Mock(), CalledProcessError=subprocess.CalledProcessError
        ),
        "FFmpegPostProcessor": SimpleNamespace(write_video_only_output=Mock()),
    }
    source = ROOT / "app/processors/video_processor.py"
    _compile_source(source, namespace, {"_close_recording_encoder"})
    _compile_source(
        source,
        namespace,
        {
            "_finalize_default_style_recording",
            "stop_current_segment",
            "finalize_segment_concatenation",
            "stop_processing",
            "_cleanup_temp_dir",
            "_handle_fatal_processing_error",
        },
        class_name="VideoProcessor",
    )
    return namespace


def _processor(root, namespace, encoder, *, segments=False):
    segment_dir = root / "temporary-segments"
    segment_dir.mkdir()
    temporary_video = (
        segment_dir / "segment.mp4" if segments else root / "temporary-video.mp4"
    )
    temporary_video.write_bytes(b"some encoded bytes do not prove success")
    slider = SimpleNamespace(
        value=Mock(return_value=12),
        maximum=lambda: 100,
        blockSignals=Mock(),
        setValue=Mock(),
    )
    processor = SimpleNamespace(
        encoder=encoder,
        recording=not segments,
        processing=True,
        is_processing_segments=segments,
        triggered_by_job_manager=False,
        last_processing_error=None,
        _fatal_processing_error_latched=False,
        gpu_memory_update_timer=SimpleNamespace(stop=Mock()),
        preroll_timer=SimpleNamespace(stop=Mock()),
        stop_live_sound=Mock(),
        _stop_recording_ffmpeg_input_stream=Mock(),
        media_capture=None,
        feeder_thread=None,
        detector_thread=None,
        join_and_clear_threads=Mock(),
        _purge_queues_and_buffers=Mock(),
        frames_to_display={},
        frame_queue=queue.Queue(),
        webcam_frames_to_display=queue.Queue(),
        media_pipeline=SimpleNamespace(
            absolute_frames_processed=30, raw_frame_queue=queue.Queue()
        ),
        sequential_detector=SimpleNamespace(reset_state=Mock()),
        _cancel_single_frame_preview_state=Mock(),
        _clear_single_frame_preview_caches=Mock(),
        temp_file=str(temporary_video) if not segments else "",
        temp_segment_files=[str(temporary_video)] if segments else [],
        segment_temp_dir=str(segment_dir),
        segments_to_process=[(0, 29)],
        current_segment_index=0,
        current_segment_end_frame=29,
        stopped_by_error_limit=False,
        consecutive_read_errors=0,
        total_skipped_frames=0,
        read_error_skip_count=0,
        manual_dropped_skip_count=0,
        tail_pending_stall_start_sec=0,
        tail_force_finalize_due_to_stall=False,
        _used_ffmpeg_cap=False,
        active_output_folder=str(root),
        file_type="image",
        media_path=str(root / "source.mp4"),
        fps=30.0,
        recording_source_fps=30.0,
        frames_written=30,
        play_start_time=0.0,
        play_end_time=0.0,
        start_time=0.0,
        processing_start_frame=0,
        next_frame_to_display=31,
        last_displayed_frame=29,
        _compute_play_end=Mock(return_value=(1.0, 30, 30, None)),
        _apply_job_timestamp_to_output_name=lambda *args: (None, None),
        _probe_video_duration=Mock(return_value=1.0),
        _log_hevc_thumbnail_hint_once=Mock(),
        _log_processing_summary=Mock(),
        _auto_save_workspace_for_output=Mock(),
        disable_virtualcam=Mock(),
        _reopen_video_capture=Mock(return_value=True),
        _restore_source_frame_state_after_capture_reopen=Mock(),
        _rebuild_segment_audio_if_needed=Mock(),
        process_next_segment=Mock(),
        _attempt_segment_video_only_fallback=Mock(return_value=False),
        _format_duration=lambda seconds: f"{seconds:.2f}s",
        main_window=SimpleNamespace(
            control={"OpenOutputToggle": True, "OutputMediaFolder": str(root)},
            display_messagebox_signal=SimpleNamespace(emit=Mock()),
            videoSeekSlider=slider,
            models_processor=SimpleNamespace(execute_all_deferred_unloads=Mock()),
        ),
        fatal_processing_error_signal=SimpleNamespace(emit=Mock()),
        processing_stopped_signal=SimpleNamespace(emit=Mock()),
    )
    # Exercise real native shutdown instead of modeling its state changes.
    processor.stop_processing = Mock(
        wraps=MethodType(namespace["stop_processing"], processor)
    )
    processor._cleanup_temp_dir = MethodType(namespace["_cleanup_temp_dir"], processor)
    # Native Qt connects this signal directly; the latch must prevent reentry.
    processor.fatal_processing_error_signal.emit.side_effect = lambda reason: namespace[
        "_handle_fatal_processing_error"
    ](processor, reason)
    return processor


class EncoderFailureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "previously-completed.mp4"
        self.output.write_bytes(b"previous completed output")
        self.native = _native_functions(self.output)

    def assert_failed(self, processor):
        self.assertTrue(processor.last_processing_error)
        self.assertTrue(processor._fatal_processing_error_latched)
        processor.fatal_processing_error_signal.emit.assert_called_once()
        processor.main_window.display_messagebox_signal.emit.assert_called_once()
        processor.processing_stopped_signal.emit.assert_called_once()
        self.assertFalse(processor.processing)
        self.assertFalse(processor.recording)
        self.assertFalse(processor.is_processing_segments)
        self.assertEqual(self.output.read_bytes(), b"previous completed output")
        processor._auto_save_workspace_for_output.assert_not_called()
        self.native["subprocess"].run.assert_not_called()
        self.native[
            "common_widget_actions"
        ].create_and_show_toast_message.assert_not_called()
        self.native["list_view_actions"].open_output_media_folder.assert_not_called()

    def test_default_already_exited_nonzero_is_reaped_and_never_muxed(self):
        process = _FFmpegProcess(1)
        processor = _processor(self.root, self.native, _encoder(process))
        self.native["_finalize_default_style_recording"](processor)
        process.wait.assert_called_once()
        self.assertTrue(process.stdin.closed)
        self.assert_failed(processor)
        processor._compute_play_end.assert_not_called()
        self.native[
            "layout_actions"
        ].enable_all_parameters_and_control_widget.assert_called_once()
        self.native["video_control_actions"].reset_media_buttons.assert_called_once()

    def test_default_timeout_then_zero_exit_is_still_failure(self):
        process = _FFmpegProcess(0, timeout=True)
        encoder = _encoder(process)
        processor = _processor(self.root, self.native, encoder)
        self.native["_finalize_default_style_recording"](processor)
        self.assertEqual(encoder.last_exit_code, 0)
        self.assertFalse(encoder.last_close_succeeded)
        process.terminate.assert_called_once()
        self.assert_failed(processor)

    def test_default_failed_encoder_reopens_native_video_preview(self):
        processor = _processor(self.root, self.native, _encoder(_FFmpegProcess(1)))
        processor.file_type = "video"
        processor._used_ffmpeg_cap = True
        self.native["_finalize_default_style_recording"](processor)
        self.assert_failed(processor)
        processor._reopen_video_capture.assert_called_once_with(12)
        processor._restore_source_frame_state_after_capture_reopen.assert_called_once()
        self.assertFalse(processor._used_ffmpeg_cap)

    def test_default_unexpected_close_error_reports_failure_and_resets(self):
        encoder = _encoder()
        encoder.close_process = Mock(side_effect=OSError("injected encoder error"))
        processor = _processor(self.root, self.native, encoder)
        self.native["_finalize_default_style_recording"](processor)
        self.assert_failed(processor)
        self.assertIn("injected encoder error", processor.last_processing_error)

    def test_default_zero_exit_keeps_native_mux_and_completion_workflow(self):
        process = _FFmpegProcess(0)
        processor = _processor(self.root, self.native, _encoder(process))

        def successful_mux(args, check):
            Path(args[-1]).write_bytes(b"new completed output")

        self.native["subprocess"].run.side_effect = successful_mux
        self.native["_finalize_default_style_recording"](processor)
        self.assertEqual(self.output.read_bytes(), b"new completed output")
        processor._auto_save_workspace_for_output.assert_called_once_with(
            str(self.output)
        )
        processor.fatal_processing_error_signal.emit.assert_not_called()
        self.native[
            "common_widget_actions"
        ].create_and_show_toast_message.assert_called_once()
        processor.processing_stopped_signal.emit.assert_called_once()
        self.assertIsNone(processor.last_processing_error)

    def test_failed_segment_aborts_before_audio_rebuild_or_next_segment(self):
        processor = _processor(
            self.root, self.native, _encoder(_FFmpegProcess(1)), segments=True
        )
        self.native["stop_current_segment"](processor)
        self.assert_failed(processor)
        processor.stop_processing.assert_called_once()
        processor._rebuild_segment_audio_if_needed.assert_not_called()
        processor.process_next_segment.assert_not_called()
        self.assertEqual(processor.current_segment_index, -1)
        self.assertEqual(processor.temp_segment_files, [])
        self.assertEqual(processor.active_output_folder, "")

    def test_successful_segment_keeps_audio_rebuild_and_next_segment(self):
        processor = _processor(
            self.root, self.native, _encoder(_FFmpegProcess(0)), segments=True
        )
        self.native["stop_current_segment"](processor)
        processor.stop_processing.assert_not_called()
        processor._rebuild_segment_audio_if_needed.assert_called_once_with(1)
        processor.process_next_segment.assert_called_once()
        processor.fatal_processing_error_signal.emit.assert_not_called()

    def test_failed_segment_failsafe_does_not_delete_existing_completed_output(self):
        processor = _processor(
            self.root, self.native, _encoder(_FFmpegProcess(1)), segments=True
        )
        self.native["finalize_segment_concatenation"](processor)
        self.assert_failed(processor)
        self.native["misc_helpers"].get_output_file_path.assert_not_called()
        processor.stop_processing.assert_called_once()

    def test_concat_rechecks_failed_timeout_even_after_process_is_cleared(self):
        encoder = _encoder(_FFmpegProcess(0, timeout=True))
        self.assertIs(encoder.close_process(), False)
        processor = _processor(self.root, self.native, encoder, segments=True)
        self.native["finalize_segment_concatenation"](processor)
        self.assert_failed(processor)
        self.assertEqual(encoder.last_exit_code, 0)

    def test_concat_accepts_idempotent_clean_exit_and_preserves_success_workflow(self):
        encoder = _encoder(_FFmpegProcess(0))
        self.assertIs(encoder.close_process(), True)
        processor = _processor(self.root, self.native, encoder, segments=True)

        def successful_concat(args, check):
            Path(args[-1]).write_bytes(b"new concatenated output")

        self.native["subprocess"].run.side_effect = successful_concat
        self.native["finalize_segment_concatenation"](processor)
        self.assertEqual(self.output.read_bytes(), b"new concatenated output")
        processor._auto_save_workspace_for_output.assert_called_once_with(
            str(self.output)
        )
        processor.fatal_processing_error_signal.emit.assert_not_called()
        processor.processing_stopped_signal.emit.assert_called_once()
        self.assertFalse(processor.processing)
        self.assertFalse(processor.is_processing_segments)
        self.assertEqual(processor.current_segment_index, -1)

    def test_explicit_abort_reaps_exited_encoder_without_publishing(self):
        process = _FFmpegProcess(1)
        processor = _processor(self.root, self.native, _encoder(process), segments=True)
        self.assertIs(processor.stop_processing(), True)
        process.wait.assert_called_once()
        self.assertEqual(self.output.read_bytes(), b"previous completed output")
        processor.fatal_processing_error_signal.emit.assert_not_called()
        processor._auto_save_workspace_for_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
