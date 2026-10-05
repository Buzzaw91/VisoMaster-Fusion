"""Managed native startup and real process-held file exclusion."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.helpers.cloud_instance import acquire_cloud_instance


class CloudInstanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.app = self.root / "source" / "baseline"
        self.app.mkdir(parents=True)

    def authorize(self, **overrides):
        (self.app / ".fusion-source.json").write_text("{}")
        marker = {
            "active": True,
            "pid": os.getpid(),
            "role": "fusion",
            "token": "test-token",
        }
        marker.update(overrides)
        (self.root / ".writer-state.json").write_text(json.dumps(marker))
        return patch.dict(
            os.environ,
            {
                "FUSION_LOCK_OWNER_PID": str(os.getpid()),
                "FUSION_NATIVE_GUARD_PID": str(os.getpid()),
                "FUSION_LOCK_TOKEN": "test-token",
            },
        )

    def test_ordinary_install_is_unaffected(self):
        self.assertIsNone(acquire_cloud_instance(self.app))
        self.assertFalse((self.root / ".native-instance.lock").exists())

    def test_direct_managed_launch_fails_before_creating_native_lock(self):
        (self.app / ".fusion-source.json").write_text("{}")
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            acquire_cloud_instance(self.app)
        self.assertFalse((self.root / ".native-instance.lock").exists())

    def test_real_lock_remains_held_until_native_handle_closes(self):
        with self.authorize():
            first = acquire_cloud_instance(self.app)
            self.assertIsNotNone(first)
            with self.assertRaises(RuntimeError):
                acquire_cloud_instance(self.app)
            assert first is not None
            first.close()
            second = acquire_cloud_instance(self.app)
            assert second is not None
            second.close()

    def test_unclean_desktop_cannot_authorize_native_restart(self):
        with (
            self.authorize(role="desktop", unclean_fusion=True),
            self.assertRaises(RuntimeError),
        ):
            acquire_cloud_instance(self.app)

    def test_desktop_without_fusion_wrapper_handshake_is_refused(self):
        with self.authorize(role="desktop"):
            with (
                patch.dict(os.environ, {"FUSION_NATIVE_GUARD_PID": "-1"}),
                self.assertRaises(RuntimeError),
            ):
                acquire_cloud_instance(self.app)

    def test_container_pid_one_guard_keeps_native_instance_exclusion(self):
        with (
            self.authorize(pid=1),
            patch.dict(
                os.environ,
                {"FUSION_LOCK_OWNER_PID": "1", "FUSION_NATIVE_GUARD_PID": "1"},
            ),
            patch("app.helpers.cloud_instance._ancestor_pids", return_value={1}),
        ):
            handle = acquire_cloud_instance(self.app)
            assert handle is not None
            try:
                with self.assertRaises(RuntimeError):
                    acquire_cloud_instance(self.app)
            finally:
                handle.close()


if __name__ == "__main__":
    unittest.main()
