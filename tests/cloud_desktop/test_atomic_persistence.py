"""Persistence fault tests without Qt, CUDA, pytest, or installed model weights.

Run with ``python -m unittest discover -s tests/cloud_desktop -p test_atomic_persistence.py``.
The native functions are compiled from their actual source with small UI fakes;
this exercises their save/close control flow without importing the GPU application.
An additional real Torch round trip runs when Torch is already installed.
"""

from __future__ import annotations

import ast
import copy
import errno
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import Mock, patch

from app.helpers import atomic_io


ROOT = Path(__file__).resolve().parents[2]
SAVE_SOURCE = ROOT / "app/ui/widgets/actions/save_load_actions.py"


def _compile_functions(path, names, namespace, *, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(
            node.body for node in nodes if getattr(node, "name", None) == class_name
        )
    selected = [node for node in nodes if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == set(names)
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


class _ParametersDict:
    def __init__(self, data):
        self.data = data


class _Card:
    pass


def _native_functions():
    # A deterministic serializer stands in for Torch in fault/control-flow tests.
    # No untrusted pickle is loaded. The optional Torch test below covers real bytes.
    namespace = {
        "Path": Path,
        "copy": copy,
        "cast": cast,
        "MarkerTypes": dict,
        "json": json,
        "hashlib": hashlib,
        "io": io,
        "print": Mock(),
        "torch": SimpleNamespace(
            save=lambda payload, stream: stream.write(json.dumps(payload).encode())
        ),
        "atomic_write_bytes": atomic_io.atomic_write_bytes,
        "atomic_write_json": atomic_io.atomic_write_json,
        "_KV_REGISTRY_LOCK": threading.Lock(),
        "misc_helpers": SimpleNamespace(
            ParametersDict=_ParametersDict,
            is_file_exists=lambda path: Path(path).is_file(),
        ),
        "widget_components": SimpleNamespace(TargetMediaCardButton=_Card),
        "list_view_actions": SimpleNamespace(
            _FACE_BUTTON_SIZE=100, _SMALL_FACE_BUTTON_SIZE=50
        ),
        "common_widget_actions": SimpleNamespace(
            create_and_show_toast_message=Mock(), create_and_show_messagebox=Mock()
        ),
        "QtWidgets": SimpleNamespace(
            QFileDialog=SimpleNamespace(getSaveFileName=Mock(return_value=("", ""))),
            QDialog=SimpleNamespace(Accepted=1),
            QMessageBox=SimpleNamespace(Yes=1, No=0, question=Mock(return_value=1)),
        ),
    }
    _compile_functions(
        SAVE_SOURCE,
        {
            "_load_registry",
            "_update_registry",
            "_save_hashed_kv_payload",
            "sanitize_state_dictionary",
            "sanitize_markers_dictionary",
            "convert_markers_to_supported_type",
            "convert_parameters_to_supported_type",
            "save_current_workspace",
            "_save_current_workspace",
            "save_current_job",
            "save_embeddings_to_file",
        },
        namespace,
    )
    return namespace


def _window(root):
    checked = SimpleNamespace(isChecked=lambda: True)
    params = _ParametersDict({})
    geometry = SimpleNamespace(
        x=lambda: 1, y=lambda: 2, width=lambda: 800, height=lambda: 600
    )
    return SimpleNamespace(
        project_root_path=root,
        last_workspace_path=root / "last_workspace.json",
        control={},
        target_videos={},
        input_faces={},
        target_faces={},
        merged_embeddings={},
        parameters={},
        default_parameters=params,
        current_widget_parameters=params,
        markers={},
        issue_frames_by_face={},
        dropped_frames=set(),
        job_marker_pairs=[],
        last_target_media_folder_path="",
        last_input_media_folder_path="",
        loaded_embedding_filename="",
        selected_video_button=False,
        is_full_screen=False,
        isMaximized=lambda: False,
        geometry=lambda: geometry,
        saveState=lambda: SimpleNamespace(
            toBase64=lambda: SimpleNamespace(data=lambda: b"layout")
        ),
        panel_visibility_state={},
        targetVideosFilterImagesCheckBox=checked,
        targetVideosFilterVideosCheckBox=checked,
        targetVideosFilterWebcamsCheckBox=checked,
        targetVideosFilterMenuButton=checked,
        swapfacesButton=checked,
        editFacesButton=checked,
        tabWidget=SimpleNamespace(currentIndex=lambda: 0, count=lambda: 0),
        quit_without_saving=False,
        video_processor=SimpleNamespace(
            stop_processing=Mock(), join_and_clear_threads=Mock()
        ),
    )


class AtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.target = self.root / "workspace.json"
        self.target.write_bytes(b"previous workspace")

    def assert_preserved(self):
        self.assertEqual(self.target.read_bytes(), b"previous workspace")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_json_serialization_failure_preserves_previous_file(self):
        with self.assertRaises(TypeError):
            atomic_io.atomic_write_json(self.target, {"bad": object()})
        self.assert_preserved()

    def test_partial_write_enospc_preserves_previous_file_and_cleans_temp(self):
        original_fdopen = atomic_io.os.fdopen

        class FullDiskWriter:
            def __init__(self, fd, mode):
                self.stream = original_fdopen(fd, mode)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def write(self, content):
                self.stream.write(content[:3])
                raise OSError(errno.ENOSPC, "injected full storage")

        with patch.object(atomic_io.os, "fdopen", FullDiskWriter):
            with self.assertRaises(OSError) as error:
                atomic_io.atomic_write_bytes(self.target, b"new workspace")
        self.assertEqual(error.exception.errno, errno.ENOSPC)
        self.assert_preserved()

    def test_file_sync_failure_preserves_previous_file(self):
        with patch.object(
            atomic_io.os,
            "fsync",
            side_effect=OSError(errno.EIO, "injected I/O failure"),
        ):
            with self.assertRaises(OSError):
                atomic_io.atomic_write_bytes(self.target, b"new workspace")
        self.assert_preserved()

    def test_replace_failure_preserves_previous_file(self):
        with patch.object(
            atomic_io.os,
            "replace",
            side_effect=OSError(errno.EACCES, "injected replace failure"),
        ):
            with self.assertRaises(OSError):
                atomic_io.atomic_write_bytes(self.target, b"new workspace")
        self.assert_preserved()

    def test_symlink_and_dangling_symlink_preserve_mapping(self):
        mounted = self.root / "persistent-volume"
        mounted.mkdir()
        for already_exists in (False, True):
            with self.subTest(already_exists=already_exists):
                target = mounted / f"state-{already_exists}.json"
                if already_exists:
                    target.write_bytes(b"old state")
                link = self.root / f"native-{already_exists}.json"
                link.symlink_to(target)
                atomic_io.atomic_write_json(link, {"saved": True})
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.resolve(), target)
                self.assertEqual(json.loads(target.read_text()), {"saved": True})
                self.assertEqual(list(mounted.glob(".*.tmp")), [])

    def test_success_preserves_existing_permissions(self):
        self.target.chmod(0o640)
        atomic_io.atomic_write_bytes(self.target, b"new workspace")
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o640)
        self.assertEqual(self.target.read_bytes(), b"new workspace")

    def test_directory_sync_runs_after_replacement_and_reports_real_failure(self):
        def failed_sync(directory):
            self.assertEqual(directory, self.root)
            self.assertEqual(self.target.read_bytes(), b"new workspace")
            raise OSError(errno.EIO, "injected directory sync failure")

        with patch.object(atomic_io, "_sync_directory", side_effect=failed_sync):
            with self.assertRaises(OSError):
                atomic_io.atomic_write_bytes(self.target, b"new workspace")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_unsupported_directory_sync_is_tolerated(self):
        if atomic_io.os.name == "nt":
            self.skipTest("Windows does not attempt directory fsync")
        with patch.object(
            atomic_io.os, "fsync", side_effect=OSError(errno.EINVAL, "unsupported")
        ):
            atomic_io._sync_directory(self.root)


class NativeSaveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.native = _native_functions()
        self.window = _window(self.root)
        self.workspace = self.window.last_workspace_path
        self.workspace.write_bytes(b"previous workspace")

    def save(self):
        return self.native["save_current_workspace"](self.window, self.workspace)

    def assert_failed_save(self):
        self.assertEqual(self.workspace.read_bytes(), b"previous workspace")
        self.native[
            "common_widget_actions"
        ].create_and_show_messagebox.assert_called_once()
        self.native[
            "common_widget_actions"
        ].create_and_show_toast_message.assert_not_called()

    def add_reference(self, kind):
        if kind == "input":
            self.window.control["DenoiserUNetEnableBeforeRestorersToggle"] = True
            self.window.input_faces["input"] = SimpleNamespace(
                media_path="source.png", kv_map={"tensor": [1, 2]}
            )
        else:
            self.window.merged_embeddings["merged"] = SimpleNamespace(
                embedding_name="source",
                embedding_store={},
                kv_map={"tensor": [1, 2]} if kind == "map" else None,
                kv_map_list=[{"tensor": [1, 2]}] if kind == "list" else None,
            )

    def test_save_returns_explicit_success_and_keeps_native_schema(self):
        self.assertIs(self.save(), True)
        saved = json.loads(self.workspace.read_text())
        self.assertEqual(saved["window_state_data"]["dock_state"], "layout")
        self.assertIn("target_faces_data", saved)

    def test_cancelled_save_returns_false_without_creating_state(self):
        self.assertIs(self.native["save_current_workspace"](self.window), False)
        self.assertEqual(self.workspace.read_bytes(), b"previous workspace")

    def test_native_save_through_dangling_symlink_persists_on_volume(self):
        mounted = self.root / "persistent-volume"
        mounted.mkdir()
        target = mounted / "last_workspace.json"
        self.workspace.unlink()
        self.workspace.symlink_to(target)
        self.add_reference("list")
        self.assertIs(self.save(), True)
        self.assertTrue(self.workspace.is_symlink())
        self.assertTrue(target.is_file())
        registry = self.root / "model_assets/reference_kv_data/kv_registry.json"
        self.assertIn(
            str(target), next(iter(json.loads(registry.read_text()).values()))
        )

    def test_autosave_serialization_failure_is_reported_and_preserves_previous(self):
        self.window.control["unserializable"] = object()
        self.assertIs(self.save(), False)
        self.assert_failed_save()

    def test_workspace_replace_failure_is_reported_and_preserves_previous(self):
        with patch.object(
            atomic_io.os,
            "replace",
            side_effect=OSError(errno.ENOSPC, "injected full storage"),
        ):
            self.assertIs(self.save(), False)
        self.assert_failed_save()

    def test_required_input_and_merged_maps_must_succeed(self):
        for kind in ("input", "map", "list"):
            with self.subTest(kind=kind):
                self.window.input_faces.clear()
                self.window.merged_embeddings.clear()
                self.add_reference(kind)
                self.native[
                    "common_widget_actions"
                ].create_and_show_messagebox.reset_mock()
                with patch.dict(
                    self.native,
                    atomic_write_bytes=Mock(
                        side_effect=OSError(errno.ENOSPC, "injected full storage")
                    ),
                ):
                    self.assertIs(self.save(), False)
                self.assert_failed_save()

    def test_registry_write_failure_preserves_registry_and_workspace(self):
        self.add_reference("list")
        registry = self.root / "model_assets/reference_kv_data/kv_registry.json"
        registry.parent.mkdir(parents=True)
        registry.write_text('{"kv_prior.pt": ["prior.json"]}')
        prior_registry = registry.read_bytes()
        original_replace = atomic_io.os.replace

        def replace(source, destination):
            if Path(destination) == registry:
                raise OSError(errno.ENOSPC, "injected registry failure")
            original_replace(source, destination)

        with patch.object(atomic_io.os, "replace", side_effect=replace):
            self.assertIs(self.save(), False)
        self.assert_failed_save()
        self.assertEqual(registry.read_bytes(), prior_registry)
        self.assertEqual(list(registry.parent.glob(".*.tmp")), [])

    def test_corrupt_registry_is_not_reset_or_published(self):
        self.add_reference("map")
        registry = self.root / "model_assets/reference_kv_data/kv_registry.json"
        registry.parent.mkdir(parents=True)
        registry.write_bytes(b"interrupted registry")
        self.assertIs(self.save(), False)
        self.assert_failed_save()
        self.assertEqual(registry.read_bytes(), b"interrupted registry")

    def test_existing_corrupt_payload_is_repaired_before_workspace_publish(self):
        self.add_reference("list")
        self.assertIs(self.save(), True)
        kv_path = Path(
            json.loads(self.workspace.read_text())["embeddings_data"]["merged"][
                "kv_map"
            ]
        )
        expected = kv_path.read_bytes()
        kv_path.write_bytes(b"truncated")
        self.assertIs(self.save(), True)
        self.assertEqual(kv_path.read_bytes(), expected)
        self.assertEqual(kv_path.stem, "kv_" + hashlib.sha256(expected).hexdigest())

    def test_verified_existing_payload_deduplicates_without_loading_pickle(self):
        payload = {"kv_map": {"tensor": [1, 2]}}
        helper = self.native["_save_hashed_kv_payload"]
        saved = helper(self.window, payload, raise_on_error=True)
        with patch.dict(
            self.native,
            atomic_write_bytes=Mock(side_effect=AssertionError("unexpected rewrite")),
        ):
            self.assertEqual(helper(self.window, payload, raise_on_error=True), saved)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None, "Torch is not installed"
    )
    def test_real_torch_payload_uses_native_buffer_hash_and_safe_cpu_round_trip(self):
        import torch

        self.native["torch"] = torch
        payload = {"kv_map_list": [{"block": torch.arange(8).reshape(2, 4)}]}
        saved = Path(
            self.native["_save_hashed_kv_payload"](
                self.window, payload, raise_on_error=True
            )
        )
        self.assertEqual(
            saved.stem, "kv_" + hashlib.sha256(saved.read_bytes()).hexdigest()
        )
        restored = torch.load(saved, map_location="cpu", weights_only=True)
        self.assertTrue(
            torch.equal(
                restored["kv_map_list"][0]["block"], payload["kv_map_list"][0]["block"]
            )
        )

    def test_embedding_reference_failure_does_not_replace_previous_embedding_file(self):
        self.add_reference("list")
        saved = self.root / "embeddings.json"
        saved.write_bytes(b"previous embeddings")
        self.native["QtWidgets"].QFileDialog.getSaveFileName.return_value = (
            str(saved),
            "",
        )
        with patch.dict(
            self.native,
            atomic_write_bytes=Mock(
                side_effect=OSError(errno.ENOSPC, "injected full storage")
            ),
        ):
            self.native["save_embeddings_to_file"](self.window, save_as=True)
        self.assertEqual(saved.read_bytes(), b"previous embeddings")
        self.native[
            "common_widget_actions"
        ].create_and_show_toast_message.assert_not_called()

    def test_job_reference_failure_does_not_replace_previous_job_file(self):
        self.add_reference("input")
        self.window.selected_video_button = True
        self.window.target_faces["face"] = SimpleNamespace(
            assigned_input_faces={"input": True}
        )
        self.native["widget_components"].SaveJobDialog = lambda window: SimpleNamespace(
            exec=lambda: 1,
            job_name="saved-job",
            use_job_name_for_output=True,
            output_file_name="video.mp4",
        )
        saved = self.root / ".jobs/saved-job.json"
        saved.parent.mkdir()
        saved.write_bytes(b"previous job")
        with patch.dict(
            self.native,
            atomic_write_bytes=Mock(
                side_effect=OSError(errno.ENOSPC, "injected full storage")
            ),
        ):
            self.native["save_current_job"](self.window)
        self.assertEqual(saved.read_bytes(), b"previous job")
        self.native[
            "common_widget_actions"
        ].create_and_show_toast_message.assert_not_called()

    def close(self, saver=None):
        namespace = {
            "print": Mock(),
            "list_view_actions": SimpleNamespace(
                clear_stop_loading_input_media=Mock(),
                clear_stop_loading_target_media=Mock(),
            ),
            "save_load_actions": SimpleNamespace(
                save_current_workspace=saver or self.native["save_current_workspace"]
            ),
            "QtWidgets": SimpleNamespace(QMessageBox=SimpleNamespace(critical=Mock())),
        }
        _compile_functions(
            ROOT / "app/ui/main_ui.py",
            {"closeEvent"},
            namespace,
            class_name="MainWindow",
        )
        event = SimpleNamespace(ignore=Mock(), accept=Mock())
        namespace["closeEvent"](self.window, event)
        return namespace, event

    def test_failed_close_uses_real_save_result_and_keeps_window_state_available(self):
        self.window.control["unserializable"] = object()
        namespace, event = self.close()
        self.assert_failed_save()
        event.ignore.assert_called_once()
        event.accept.assert_not_called()
        self.window.video_processor.join_and_clear_threads.assert_not_called()
        namespace[
            "list_view_actions"
        ].clear_stop_loading_input_media.assert_called_once_with(
            self.window, clear_list=False
        )
        namespace[
            "list_view_actions"
        ].clear_stop_loading_target_media.assert_called_once_with(
            self.window, clear_list=False
        )

    def test_successful_close_finishes_thread_shutdown_and_accepts_event(self):
        _, event = self.close()
        event.accept.assert_called_once()
        event.ignore.assert_not_called()
        self.window.video_processor.join_and_clear_threads.assert_called_once()

    def test_explicit_quit_without_saving_does_not_call_saver(self):
        self.window.quit_without_saving = True
        saver = Mock(side_effect=AssertionError("should skip save"))
        _, event = self.close(saver)
        saver.assert_not_called()
        event.accept.assert_called_once()
        self.assertEqual(self.workspace.read_bytes(), b"previous workspace")

    def test_unexpected_save_exception_reports_failure_and_ignores_close(self):
        namespace, event = self.close(
            Mock(side_effect=RuntimeError("injected exception"))
        )
        namespace["QtWidgets"].QMessageBox.critical.assert_called_once()
        event.ignore.assert_called_once()
        event.accept.assert_not_called()


if __name__ == "__main__":
    unittest.main()
