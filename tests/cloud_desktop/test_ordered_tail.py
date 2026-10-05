"""Exercise the native consumer and EOF helpers without optional GUI packages."""

import ast
from pathlib import Path
import queue
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import Mock, patch


def native_methods():
    source = (
        Path(__file__).resolve().parents[2]
        / "app/processors/video_utils/media_pipeline.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    names = {
        "display_next_frame",
        "_handle_tail_drain_wait",
        "_mark_skipped_frame",
        "_mark_unavailable_tail",
        "is_draining_tail",
    }
    methods = [n for n in cls.body if getattr(n, "name", "") in names]
    scope = {
        "time": time,
        "TAIL_PENDING_STALL_TIMEOUT_SEC": 8.0,
        "common_widget_actions": SimpleNamespace(
            get_pixmap_from_frame=lambda _, frame: frame
        ),
        "graphics_view_actions": Mock(),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), scope)
    return type("NativeTail", (), {name: scope[name] for name in names})


class OrderedTailTests(unittest.TestCase):
    def pipeline(self):
        pipe = native_methods()()
        tasks = queue.Queue()
        pipe.frames_to_display = {}
        pipe.skipped_frames = set()
        pipe.state_lock = threading.Lock()
        pipe.total_skipped_frames = pipe.manual_dropped_skip_count = (
            pipe.read_error_skip_count
        ) = 0
        pipe.segment_jumps = {}
        pipe._wrap_frame_target = -1
        pipe.last_display_schedule_time_sec = time.perf_counter()
        pipe.target_delay_sec = 0.005
        pipe.heartbeat_frame_counter = 0
        pipe.precise_metronome = Mock()
        pipe._update_playback_fps_display = Mock()
        pipe.feeder_thread = pipe.detector_thread = SimpleNamespace(
            is_alive=lambda: False
        )
        pipe.main_window = SimpleNamespace(
            control={},
            markers={},
            models_processor=Mock(),
            display_messagebox_signal=Mock(),
        )
        encoded = []
        encoder = SimpleNamespace(
            is_running=lambda: True,
            write_frame=lambda frame: encoded.append(frame) is None,
        )
        pipe.vp = SimpleNamespace(
            file_type="video",
            is_processing_segments=False,
            recording=True,
            processing=True,
            max_frame_number=2,
            next_frame_to_display=1,
            current_segment_end_frame=None,
            frames_written=0,
            tail_pending_stall_start_sec=0.0,
            tail_force_finalize_due_to_stall=False,
            fps=30,
            recording_source_fps=30,
            _used_ffmpeg_cap=False,
            encoder=encoder,
            send_frame_to_virtualcam=Mock(),
            _safe_unfinished_tasks=lambda: tasks.unfinished_tasks,
            _finalize_default_style_recording=Mock(),
            _handle_fatal_processing_error=Mock(),
        )
        return pipe, tasks, encoded

    def test_later_tail_frame_waits_for_earlier_worker_and_queued_delivery(self):
        pipe, tasks, encoded = self.pipeline()
        tasks.put("earlier in flight")
        task = tasks.get()
        pipe.frames_to_display[2] = 2
        pipe.display_next_frame()
        self.assertEqual(encoded, [])
        self.assertEqual(pipe.vp.next_frame_to_display, 1)
        # Worker finished, but its Qt signal is still waiting for GUI delivery.
        self.assertEqual(task, "earlier in flight")
        tasks.task_done()
        pipe.display_next_frame()
        self.assertEqual(encoded, [])
        pipe.frames_to_display[1] = 1
        pipe.display_next_frame()
        pipe.display_next_frame()
        self.assertEqual(encoded, [1, 2])
        pipe.display_next_frame()
        pipe.vp._finalize_default_style_recording.assert_called_once_with()

    def test_marked_interior_and_final_skips_preserve_order(self):
        pipe, _, encoded = self.pipeline()
        pipe.vp.max_frame_number = 3
        pipe.vp.next_frame_to_display = 0
        pipe.frames_to_display = {2: 2, 0: 0}
        pipe.skipped_frames = {1, 3}
        for _ in range(4):
            pipe.display_next_frame()
        self.assertEqual(encoded, [0, 2])
        pipe.vp._finalize_default_style_recording.assert_called_once_with()

    def test_decoder_unavailable_tail_does_not_overwrite_buffered_cursor(self):
        pipe, _, encoded = self.pipeline()
        pipe.vp.max_frame_number = 4
        pipe.frames_to_display = {2: 2, 1: 1}
        pipe._mark_unavailable_tail(3, 4)
        pipe._mark_unavailable_tail(3, 4)
        self.assertEqual(pipe.vp.next_frame_to_display, 1)
        self.assertEqual(pipe.total_skipped_frames, 2)
        for _ in range(4):
            pipe.display_next_frame()
        self.assertEqual(encoded, [1, 2])
        pipe.vp._finalize_default_style_recording.assert_called_once_with()

    def test_permanently_missing_frame_aborts_at_unchanged_timeout(self):
        pipe, _, encoded = self.pipeline()
        pipe.frames_to_display[2] = 2
        pipe.vp.tail_pending_stall_start_sec = 1.0
        with patch.object(time, "perf_counter", return_value=9.1):
            pipe.display_next_frame()
        self.assertEqual(encoded, [])
        pipe.vp._handle_fatal_processing_error.assert_called_once()
        pipe.main_window.display_messagebox_signal.emit.assert_called_once()
        pipe.vp._finalize_default_style_recording.assert_not_called()
        self.assertFalse(pipe.vp.tail_force_finalize_due_to_stall)


if __name__ == "__main__":
    unittest.main()
