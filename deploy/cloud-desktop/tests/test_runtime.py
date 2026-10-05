"""CPU-only protection tests; no model packages, desktop, cloud or installs."""

import difflib
import importlib.util
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))
import runtime  # noqa: E402 — load the deployment helper after adding its path.

SSH_SPEC = importlib.util.spec_from_file_location(
    "fusion_test_ssh", DEPLOY / "prepare-ssh.py"
)
assert SSH_SPEC is not None and SSH_SPEC.loader is not None
SSH = importlib.util.module_from_spec(SSH_SPEC)
SSH_SPEC.loader.exec_module(SSH)


class ProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "persistent"
        self.root.mkdir()
        self.root = self.root.resolve()
        self.env = {
            k: v for k, v in os.environ.items() if not k.startswith("FUSION_LOCK_")
        }

    def tearDown(self):
        self.temporary.cleanup()

    def command(self, *args):
        return [
            sys.executable,
            str(DEPLOY / "runtime.py"),
            "--root",
            str(self.root),
            *args,
        ]

    def invoke(self, *args):
        return subprocess.run(
            self.command(*args), env=self.env, capture_output=True, text=True
        )

    def guarded(self, code):
        return self.invoke("guard", "--role", "setup", "--", sys.executable, "-c", code)

    def fake_source(self):
        app = self.root / "source" / runtime.SOURCE_SHA
        (app / "model_assets").mkdir(parents=True)
        (app / "main.py").write_text("print('native')\n")
        (app / "model_assets" / "seed.bin").write_bytes(b"pinned seed")
        runtime.atomic_json(
            app / ".fusion-source.json",
            {
                "source_sha": runtime.SOURCE_SHA,
                "files": {"main.py": runtime.digest(app / "main.py")},
                "seed_files": {
                    "model_assets/seed.bin": runtime.digest(
                        app / "model_assets/seed.bin"
                    )
                },
            },
        )
        return app

    def test_second_root_writer_is_refused_without_altering_state(self):
        code = "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(2)"
        ready = self.root / "ready"
        first = subprocess.Popen(
            self.command(
                "guard",
                "--role",
                "fusion",
                "--",
                sys.executable,
                "-c",
                code,
                str(ready),
            ),
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            for _ in range(100):
                if ready.exists():
                    break
                time.sleep(0.02)
            self.assertTrue(ready.exists())
            marker = (self.root / ".writer-state.json").read_bytes()
            second = self.guarded("raise SystemExit(0)")
            self.assertEqual(second.returncode, 73, second.stderr)
            self.assertEqual(marker, (self.root / ".writer-state.json").read_bytes())
            self.assertEqual(first.wait(timeout=5), 0)
            self.assertFalse(
                runtime.read_json(self.root / ".writer-state.json")["active"]
            )
        finally:
            if first.poll() is None:
                first.kill()
                first.wait()
            first.stdout.close()
            first.stderr.close()

    def test_abnormal_exit_needs_explicit_exact_token_recovery(self):
        self.assertEqual(self.guarded("raise SystemExit(19)").returncode, 19)
        marker = runtime.read_json(self.root / ".writer-state.json")
        self.assertTrue(marker["active"])
        self.assertEqual(self.guarded("raise SystemExit(0)").returncode, 73)
        self.assertEqual(
            self.invoke("recover-lock", "--expect-token", "wrong").returncode, 73
        )
        self.assertTrue(runtime.read_json(self.root / ".writer-state.json")["active"])
        self.assertEqual(
            self.invoke("recover-lock", "--expect-token", marker["token"]).returncode, 0
        )
        self.assertTrue(list(self.root.glob("writer-recovery-*.json")))
        self.assertEqual(self.guarded("raise SystemExit(0)").returncode, 0)

    def test_same_live_process_identity_refuses_recovery_without_changing_marker(self):
        identity = {
            "pid_namespace_inode": 101,
            "start_ticks": 202,
            "boot_id": "00112233-4455-6677-8899-aabbccddeeff",
        }
        marker = {
            "active": True,
            "pid": os.getpid(),
            "role": "desktop",
            "token": "a" * 32,
            "source_sha": runtime.SOURCE_SHA,
            "process_identity": identity,
        }
        path = self.root / ".writer-state.json"
        runtime.atomic_json(path, marker)
        previous = path.read_bytes()
        with mock.patch.object(runtime, "process_identity", return_value=identity):
            with self.assertRaisesRegex(RuntimeError, "PID is alive"):
                runtime.recover_lock(self.root, marker["token"])
        self.assertEqual(path.read_bytes(), previous)

    def test_reused_pid_in_new_namespace_still_requires_explicit_token_recovery(self):
        old = {
            "pid_namespace_inode": 101,
            "start_ticks": 202,
            "boot_id": "00112233-4455-6677-8899-aabbccddeeff",
        }
        new = old | {"pid_namespace_inode": 303, "start_ticks": 404}
        marker = {
            "active": True,
            "pid": os.getpid(),
            "role": "desktop",
            "token": "b" * 32,
            "source_sha": runtime.SOURCE_SHA,
            "process_identity": old,
        }
        path = self.root / ".writer-state.json"
        runtime.atomic_json(path, marker)
        previous = path.read_bytes()
        # Even a provably different owner identity never authorizes automatic start.
        self.assertEqual(self.guarded("raise SystemExit(0)").returncode, 73)
        self.assertEqual(path.read_bytes(), previous)
        with mock.patch.object(runtime, "process_identity", return_value=new):
            with self.assertRaisesRegex(RuntimeError, "token"):
                runtime.recover_lock(self.root, "wrong-token")
            self.assertEqual(path.read_bytes(), previous)
            runtime.recover_lock(self.root, marker["token"])
        recovered = runtime.read_json(path)
        self.assertFalse(recovered["active"])
        self.assertTrue(recovered["recovery_required_snapshot_review"])
        self.assertEqual(recovered["recovery_actor_identity"], new)
        self.assertEqual(
            runtime.read_json(next(self.root.glob("writer-recovery-*.json"))), marker
        )

    def test_unknown_marker_is_not_silently_treated_as_clean(self):
        runtime.atomic_json(self.root / ".writer-state.json", {"active": False})
        self.assertEqual(self.guarded("raise SystemExit(0)").returncode, 73)

    def test_native_mapping_preserves_workspace_and_reference_assets(self):
        app = self.fake_source()
        native = self.root / "native" / runtime.SOURCE_SHA
        native.mkdir(parents=True)
        workspace = native / "last_workspace.json"
        workspace.write_text('{"assignments": "keep"}')
        refs = (
            self.root
            / "assets"
            / runtime.SOURCE_SHA
            / "model_assets"
            / "reference_kv_data"
        )
        refs.mkdir(parents=True)
        reference = refs / "kv_keep.pt"
        reference.write_bytes(b"user reference tensors")

        def map_key(key):
            result = self.invoke(
                "guard",
                "--role",
                "setup",
                "--",
                *self.command("map", "--cache-key", key),
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        map_key("gpu-a")
        map_key("gpu-b")
        self.assertEqual(
            (app / "last_workspace.json").read_text(), workspace.read_text()
        )
        self.assertEqual(reference.read_bytes(), b"user reference tensors")
        self.assertEqual(
            (app / "model_assets" / "seed.bin").read_bytes(), b"pinned seed"
        )
        self.assertEqual(
            (app / "tensorrt-engines").resolve(),
            self.root / "cache/gpu-b/tensorrt-engines",
        )
        self.assertTrue((self.root / "cache/gpu-a/tensorrt-engines").is_dir())

    def test_conflicting_seed_is_preserved_and_setup_fails(self):
        self.fake_source()
        assets = self.root / "assets" / runtime.SOURCE_SHA / "model_assets"
        assets.mkdir(parents=True)
        (assets / "seed.bin").write_bytes(b"existing different user asset")
        result = self.invoke(
            "guard",
            "--role",
            "setup",
            "--",
            *self.command("map", "--cache-key", "trial"),
        )
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertEqual(
            (assets / "seed.bin").read_bytes(), b"existing different user asset"
        )

    def test_source_tampering_is_detected_before_mapping(self):
        app = self.fake_source()
        (app / "main.py").write_text("changed code\n")
        result = self.invoke(
            "guard",
            "--role",
            "setup",
            "--",
            *self.command("map", "--cache-key", "trial"),
        )
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertFalse((self.root / "native").exists())

    def test_mapping_without_a_live_guard_is_refused(self):
        self.fake_source()
        result = self.invoke("map", "--cache-key", "trial")
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertFalse((self.root / "native").exists())

    def test_full_storage_probe_fails_before_native_state_changes(self):
        result = self.invoke("storage", "--minimum-gb", "1000000000")
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertFalse((self.root / "native").exists())

    def test_operator_volume_quota_caps_backing_capacity_and_retains_consumption(self):
        env = {
            "FUSION_VOLUME_QUOTA_FREE_GB": "3",
            "FUSION_VOLUME_QUOTA_CHECKED_AT": "manual-test-observation",
        }
        with mock.patch.dict(os.environ, env):
            report = runtime.check_storage(self.root, 2)
            self.assertEqual(
                report["effective_available_bytes_upper_bound"], 3 * 1024**3
            )
            self.assertTrue(report["backing_free_is_not_allocated_volume_quota"])
            self.assertIn(
                "manual static observation",
                report["allocated_volume_quota"]["authority"],
            )
            runtime.atomic_json(
                self.root / "receipts/quota-consumption.json",
                {
                    "declared_free_bytes": 3 * 1024**3,
                    "operator_checked_at": "manual-test-observation",
                    "download_bytes_since_observation": 2 * 1024**3,
                },
            )
            with self.assertRaisesRegex(
                RuntimeError, "backing free is not volume quota"
            ):
                runtime.check_storage(self.root, 2)

    def test_supervisor_style_closed_fds_preserve_validated_owner_and_single_fusion(
        self,
    ):
        child = "import os,subprocess,sys; os.close(int(os.environ['FUSION_LOCK_FD'])); r=subprocess.run(sys.argv[1:]); raise SystemExit(r.returncode)"
        result = self.invoke(
            "guard",
            "--role",
            "desktop",
            "--",
            sys.executable,
            "-c",
            child,
            *self.command(
                "guard",
                "--role",
                "fusion",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(0)",
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(runtime.read_json(self.root / ".writer-state.json")["active"])

    def test_public_endpoint_missing_credentials_fails_before_service_start(self):
        env = self.env | {
            "FUSION_ROOT": str(self.root),
            "FUSION_ACCESS_MODE": "public",
            "FUSION_TRANSPORT": "novnc",
        }
        env.pop("FUSION_HTPASSWD_FILE", None)
        result = subprocess.run(
            [str(DEPLOY / "entrypoint.sh")], env=env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertIn("runtime FUSION_HTPASSWD_FILE required", result.stderr)
        self.assertFalse((self.root / "runtime-state.json").exists())

    def test_public_xpra_is_refused_even_with_credentials(self):
        password_file = self.root / "htpasswd"
        password_file.write_text("test:unused-test-hash\n")
        env = self.env | {
            "FUSION_ROOT": str(self.root),
            "FUSION_ACCESS_MODE": "public",
            "FUSION_TRANSPORT": "xpra",
            "FUSION_HTPASSWD_FILE": str(password_file),
        }
        result = subprocess.run(
            [str(DEPLOY / "entrypoint.sh")], env=env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertIn("Public Xpra refused", result.stderr)
        self.assertFalse((self.root / "receipts").exists())

    def test_explicit_import_deferral_survives_real_guard_and_records_unchecked_scope(
        self,
    ):
        # Exercise the actual re-exec/guard/stage/map/receipt path without installing
        # packages. Only platform/Python availability and heavy imports are stubs.
        app = self.fake_source()
        catalog = app / "app/processors/models_data.py"
        catalog.parent.mkdir(parents=True)
        catalog.write_text("models = {}\n")
        manifest = runtime.read_json(app / ".fusion-source.json")
        manifest["files"]["app/processors/models_data.py"] = runtime.digest(catalog)
        runtime.atomic_json(app / ".fusion-source.json", manifest)
        shim_dir = Path(self.temporary.name) / "bin"
        shim_dir.mkdir()
        uname = shim_dir / "uname"
        uname.write_text(
            '#!/bin/sh\ncase "$1" in -s) echo Linux;; -m) echo x86_64;; *) exit 1;; esac\n'
        )
        uname.chmod(0o755)
        for name in ("dpkg-query", "ffmpeg"):
            shim = shim_dir / name
            shim.write_text("#!/bin/sh\nprintf 'isolated CPU test system probe\\n'\n")
            shim.chmod(0o755)
        for name in ("python3", "python3.12"):
            shim = shim_dir / name
            shim.write_text(
                f"#!{sys.executable}\nimport os,sys\nos.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
            )
            shim.chmod(0o755)
        borrowed = shim_dir / "borrowed-python"
        borrowed.write_text(
            f"#!{sys.executable}\n"
            "import os,pathlib,sys\n"
            "if len(sys.argv)>2 and sys.argv[1]=='-c' and 'Borrowed runtime must use Python' in sys.argv[2]: sys.exit(0)\n"
            "if len(sys.argv)>1 and sys.argv[1].endswith('/preflight.py'):\n"
            " pathlib.Path(os.environ['FUSION_ROOT'],'heavy-import-attempted').write_text('required')\n"
            " sys.exit(86)\n"
            f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
        )
        borrowed.chmod(0o755)
        env = self.env | {
            "FUSION_ROOT": str(self.root),
            "FUSION_MIN_FREE_GB": "0",
            "PATH": str(shim_dir) + os.pathsep + os.environ["PATH"],
        }
        command = [
            str(DEPLOY / "install.sh"),
            "install",
            "--skip-apt",
            "--skip-models",
            "--source-repo",
            str(app),
            "--borrow-python",
            str(borrowed),
        ]
        result = subprocess.run(
            [*command, "--defer-import-check"], env=env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("explicitly deferred", result.stderr)
        self.assertFalse((self.root / "heavy-import-attempted").exists())
        receipt = runtime.read_json(self.root / "receipts/setup.json")
        self.assertEqual(receipt["import_check_status"], "deferred-no-import-check")
        self.assertEqual(receipt["native_functionality"], "unvalidated")
        self.assertFalse(receipt["gpu_execution_verified"])
        self.assertFalse(
            runtime.read_json(self.root / "logs/setup-imports.json")["imports_checked"]
        )
        self.assertFalse(runtime.read_json(self.root / ".writer-state.json")["active"])
        # The ordinary path must still call the required import probe and fail.
        ordinary = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(ordinary.returncode, 86, ordinary.stderr)
        self.assertTrue((self.root / "heavy-import-attempted").exists())

    def test_internal_locked_flag_cannot_bypass_the_live_root_guard(self):
        env = self.env | {"FUSION_ROOT": str(self.root)}
        result = subprocess.run(
            [str(DEPLOY / "launch-fusion.sh"), "--locked"],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertFalse((self.root / "logs").exists())

    def test_managed_ssh_accepts_real_public_keys_and_rejects_private_or_invalid_material(
        self,
    ):
        private = self.root / "test-key"
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(private)],
            check=True,
            capture_output=True,
        )
        public = private.with_suffix(".pub").read_text()
        validated = SSH.validated_keys(public + public, self.root)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0], " ".join(public.split()[:2]))
        for invalid in (
            private.read_text(),
            'command="whoami" ' + public,
            "ssh-ed25519 AAAAinvalid",
            "",
        ):
            with self.assertRaises(RuntimeError) as rejected:
                SSH.validated_keys(invalid, self.root)
            if invalid:
                self.assertNotIn(invalid, str(rejected.exception))

    def test_native_guard_pid_is_injected_only_for_fusion_children(self):
        code = "import os,sys; assert int(os.environ['FUSION_NATIVE_GUARD_PID']) == os.getppid()"
        self.assertEqual(
            self.invoke(
                "guard", "--role", "fusion", "--", sys.executable, "-c", code
            ).returncode,
            0,
        )
        absent = "import os; assert 'FUSION_NATIVE_GUARD_PID' not in os.environ"
        self.assertEqual(self.guarded(absent).returncode, 0)

    def test_crashed_fusion_child_cannot_relaunch_under_same_desktop_owner(self):
        attempt = self.root / "second-app-started"
        child = """import pathlib,subprocess,sys
marker = pathlib.Path(sys.argv[1]); attempt = pathlib.Path(sys.argv[2]); command = sys.argv[3:]
assert subprocess.run(command + ['-c', 'raise SystemExit(17)']).returncode == 17
before = marker.read_bytes()
second = subprocess.run(command + ['-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("started")', str(attempt)])
assert second.returncode == 73
assert marker.read_bytes() == before
assert not attempt.exists()
"""
        result = self.invoke(
            "guard",
            "--role",
            "desktop",
            "--",
            sys.executable,
            "-c",
            child,
            str(self.root / ".writer-state.json"),
            str(attempt),
            *self.command("guard", "--role", "fusion", "--", sys.executable),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        marker = runtime.read_json(self.root / ".writer-state.json")
        self.assertTrue(marker["active"])
        self.assertTrue(marker["unclean_fusion"])
        self.assertEqual(
            self.invoke("recover-lock", "--expect-token", marker["token"]).returncode, 0
        )

    def test_native_instance_survives_guard_kill_and_blocks_recovery(self):
        ready = self.root / "native-ready"
        native = "import fcntl,os,pathlib,sys,time; f=open(sys.argv[1],'a+'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); pathlib.Path(sys.argv[2]).write_text(str(os.getpid())); time.sleep(30)"
        guard = subprocess.Popen(
            self.command(
                "guard",
                "--role",
                "fusion",
                "--",
                sys.executable,
                "-c",
                native,
                str(self.root / ".native-instance.lock"),
                str(ready),
            ),
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        native_pid = None
        try:
            for _ in range(100):
                if ready.exists():
                    break
                time.sleep(0.02)
            self.assertTrue(ready.exists())
            native_pid = int(ready.read_text())
            marker = (self.root / ".writer-state.json").read_bytes()
            token = runtime.read_json(self.root / ".writer-state.json")["token"]
            guard.kill()
            guard.wait(timeout=5)
            self.assertEqual(
                self.invoke("recover-lock", "--expect-token", token).returncode, 73
            )
            self.assertEqual(self.guarded("raise SystemExit(0)").returncode, 73)
            self.assertEqual((self.root / ".writer-state.json").read_bytes(), marker)
            os.kill(native_pid, signal.SIGTERM)
            native_pid = None
            for _ in range(100):
                recovered = self.invoke("recover-lock", "--expect-token", token)
                if recovered.returncode == 0:
                    break
                time.sleep(0.02)
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
        finally:
            if native_pid is not None:
                os.kill(native_pid, signal.SIGTERM)
            if guard.poll() is None:
                guard.kill()
                guard.wait()
            guard.stdout.close()
            guard.stderr.close()

    def test_explicit_source_overlay_has_receipt_and_preserves_durable_state(self):
        repository = DEPLOY.parents[1]
        original = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{runtime.SOURCE_SHA}:main.py"],
            text=True,
        )
        patched = original + "\n# Explicit deployment overlay test\n"
        patch = self.root / "input.patch"
        patch.write_text(
            "".join(
                difflib.unified_diff(
                    original.splitlines(True),
                    patched.splitlines(True),
                    fromfile="a/main.py",
                    tofile="b/main.py",
                )
            )
        )

        def stage(*extra):
            return self.invoke(
                "guard",
                "--role",
                "setup",
                "--",
                *self.command("stage", "--repository", str(repository), *extra),
            )

        result = stage()
        self.assertEqual(result.returncode, 0, result.stderr)
        app = self.root / "source" / runtime.SOURCE_SHA
        native = self.root / "native" / runtime.SOURCE_SHA
        native.mkdir(parents=True)
        (native / "last_workspace.json").write_text('{"assignments":"preserve"}')
        # Switching overlays cannot silently replace an existing source tree.
        result = stage("--source-patch", str(patch))
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertEqual((app / "main.py").read_text(), original)
        token = runtime.read_json(self.root / ".writer-state.json")["token"]
        self.assertEqual(
            self.invoke("recover-lock", "--expect-token", token).returncode, 0
        )
        result = stage("--source-patch", str(patch), "--replace-source")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((app / "main.py").read_text(), patched)
        metadata = runtime.read_json(app / ".fusion-source.json")
        self.assertEqual(metadata["source_patch_sha256"], runtime.digest(patch))
        self.assertEqual(
            (native / "last_workspace.json").read_text(), '{"assignments":"preserve"}'
        )
        self.assertTrue(list((self.root / "source-archives").iterdir()))
        self.assertTrue(
            (
                self.root / "receipts" / f"source-patch-{runtime.digest(patch)}.patch"
            ).exists()
        )
        result = stage("--source-patch", str(patch))
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
