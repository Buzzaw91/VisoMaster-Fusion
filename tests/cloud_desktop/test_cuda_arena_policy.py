"""Exercise the native dispatcher without loading optional GPU dependencies."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class ArenaPolicyTests(unittest.TestCase):
    def dispatch(self, *, enabled, device="cuda", providers=None):
        source = Path(__file__).resolve().parents[2] / (
            "app/processors/workers/function_worker.py"
        )
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(
            n for n in cls.body if getattr(n, "name", "") == "run_ort_with_iobinding"
        )
        scope = {"Any": object, "platform_support": Mock()}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
            scope,
        )
        held = []

        @contextmanager
        def lock(name):
            held.append(name)
            try:
                yield
            finally:
                held.remove(name)

        session, binding = Mock(), Mock()
        session.get_providers.return_value = providers or ["CUDAExecutionProvider"]
        session.run_with_iobinding.side_effect = lambda *args: self.assertEqual(
            held, ["gate", "session"]
        )
        binding.synchronize_outputs.side_effect = lambda: self.assertEqual(
            held, ["gate", "session"]
        )
        worker = SimpleNamespace(
            mp=SimpleNamespace(device_type=device, gpu_id=2, models={}),
            _cuda_arena_shrink=enabled,
            _get_session_lock=lambda _: lock("session"),
            _ort_inference_gate=lambda _: lock("gate"),
        )
        ort = Mock()
        with patch.dict("sys.modules", {"onnxruntime": ort}):
            scope["run_ort_with_iobinding"](worker, session, binding)
        self.assertEqual(held, [])
        binding.synchronize_outputs.assert_called_once_with()
        return session, binding, ort

    def test_default_keeps_original_run_signature(self):
        session, binding, ort = self.dispatch(enabled=False)
        session.run_with_iobinding.assert_called_once_with(binding)
        ort.RunOptions.assert_not_called()

    def test_opt_in_targets_assigned_cuda_device_and_keeps_locks(self):
        session, binding, ort = self.dispatch(enabled=True)
        ort.RunOptions.return_value.add_run_config_entry.assert_called_once_with(
            "memory.enable_memory_arena_shrinkage", "gpu:2"
        )
        session.run_with_iobinding.assert_called_once_with(
            binding, ort.RunOptions.return_value
        )

    def test_cpu_and_non_cuda_sessions_keep_native_path(self):
        for device, providers in (
            ("cpu", ["CPUExecutionProvider"]),
            ("cuda", ["CPUExecutionProvider"]),
        ):
            with self.subTest(device=device):
                session, binding, ort = self.dispatch(
                    enabled=True, device=device, providers=providers
                )
                session.run_with_iobinding.assert_called_once_with(binding)
                ort.RunOptions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
