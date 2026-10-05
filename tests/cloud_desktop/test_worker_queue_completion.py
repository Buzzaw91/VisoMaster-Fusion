"""Run the actual native pool loop against Python's real task queue."""

import ast
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import queue
from types import SimpleNamespace
import threading
import traceback
import unittest
from unittest.mock import Mock


def native_pool_run():
    source = (
        Path(__file__).resolve().parents[2] / "app/processors/workers/frame_worker.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if getattr(n, "name", "") == "run")
    scope = {
        "queue": queue,
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        "traceback": traceback,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope
    )
    return scope["run"]


class WorkerQueueCompletionTests(unittest.TestCase):
    def worker(self, tasks=None, process=None):
        return SimpleNamespace(
            is_pool_worker=True,
            frame_queue=queue.Queue() if tasks is None else tasks,
            stop_event=threading.Event(),
            name="completion-test-worker",
            process_and_emit_task=Mock() if process is None else process,
        )

    def run_worker(self, worker):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            native_pool_run()(worker)

    def test_consumed_eof_sentinel_completes_its_queue_task(self):
        worker = self.worker()
        worker.frame_queue.put(None)
        self.run_worker(worker)
        self.assertEqual(worker.frame_queue.unfinished_tasks, 0)
        worker.process_and_emit_task.assert_not_called()

    def test_processing_exception_still_acknowledges_frame_and_sentinel(self):
        worker = self.worker(process=Mock(side_effect=RuntimeError("test failure")))
        worker.frame_queue.put((0, object(), {}, {}, None, None, None, None))
        worker.frame_queue.put(None)
        self.run_worker(worker)
        self.assertEqual(worker.frame_queue.unfinished_tasks, 0)
        worker.process_and_emit_task.assert_called_once_with()

    def test_stop_immediately_after_acquisition_acknowledges_discarded_frame(self):
        worker = self.worker()
        worker.frame_queue.put((0, object(), {}, {}, None, None, None, None))
        original_get = worker.frame_queue.get

        def stop_after_get(**kwargs):
            task = original_get(**kwargs)
            worker.stop_event.set()
            return task

        worker.frame_queue.get = stop_after_get
        self.run_worker(worker)
        self.assertEqual(worker.frame_queue.unfinished_tasks, 0)
        worker.process_and_emit_task.assert_not_called()

    def test_empty_queue_timeout_does_not_acknowledge_an_unacquired_task(self):
        worker = self.worker()

        def timeout(**kwargs):
            worker.stop_event.set()
            raise queue.Empty

        worker.frame_queue.get = timeout
        worker.frame_queue.task_done = Mock(wraps=worker.frame_queue.task_done)
        self.run_worker(worker)
        worker.frame_queue.task_done.assert_not_called()

    def test_in_flight_frame_remains_pending_until_processing_returns(self):
        entered, release = threading.Event(), threading.Event()

        def process():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test release timed out")

        worker = self.worker(process=process)
        worker.frame_queue.put((0, object(), {}, {}, None, None, None, None))
        worker.frame_queue.put(None)
        thread = threading.Thread(target=self.run_worker, args=(worker,))
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertEqual(worker.frame_queue.unfinished_tasks, 2)
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(worker.frame_queue.unfinished_tasks, 0)


if __name__ == "__main__":
    unittest.main()
