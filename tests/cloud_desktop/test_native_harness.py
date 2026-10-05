"""CPU-only contract tests: run with unittest, without repository GPU conftest."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "deploy/cloud-desktop"
spec = importlib.util.spec_from_file_location("benchmark", HARNESS / "benchmark.py")
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
sys.modules["benchmark"] = benchmark
native_spec = importlib.util.spec_from_file_location(
    "cloud_native_smoke", HARNESS / "native_smoke.py"
)
assert native_spec is not None and native_spec.loader is not None
native = importlib.util.module_from_spec(native_spec)
native_spec.loader.exec_module(native)


def sample_probe(frames=90, duration=3, audio=True, codec="hevc", pixels="yuv420p10le"):
    streams = [
        {
            "codec_type": "video",
            "codec_name": codec,
            "width": 1280,
            "height": 720,
            "avg_frame_rate": "30/1",
            "nb_read_frames": str(frames),
            "duration": str(duration),
            "pix_fmt": pixels,
        }
    ]
    if audio:
        streams.append(
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "channels": 2,
                "sample_rate": "48000",
                "duration": str(duration),
            }
        )
    return {"streams": streams, "format": {"duration": str(duration)}}


class NativeHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profile_text = tomllib.loads(
            (HARNESS / "results-schema.toml").read_text()
        )["example"]["profile_toml"]
        self.profile_path = self.root / "profile.toml"
        self.profile_path.write_text(self.profile_text)
        self.profile = benchmark.read_profile(self.profile_path)

    def tearDown(self):
        self.temporary.cleanup()

    def source_tree(self):
        app_root = self.root / "source"
        (app_root / "app/processors").mkdir(parents=True)
        (app_root / "model_assets/reference_kv_data").mkdir(parents=True)
        weight = app_root / "model_assets/test.onnx"
        weight.write_bytes(b"immutable model")
        digest = benchmark.sha256(weight)
        (app_root / "app/processors/models_data.py").write_text(
            f'models_list = [{{"model_name": "Test", "local_path": f"{{models_dir}}/test.onnx", "hash": "{digest}"}}]'
        )
        (app_root / "main.py").write_text("# native entry point\n")
        (app_root / "version.json").write_text('{"version": "test"}')
        (app_root / "last_workspace.json").write_text('{"original": true}')
        kv = app_root / "model_assets/reference_kv_data/kv.pt"
        kv.write_bytes(b"reference state")
        media = self.root / "source-video.mp4"
        media.write_bytes(b"source media")
        project = self.root / "user-project.json"
        project.write_text(
            json.dumps(
                {
                    "control": {"ProvidersPrioritySelection": "TensorRT"},
                    "target_medias_data": [
                        {"media_id": "video", "media_path": str(media)}
                    ],
                    "embeddings_data": {
                        "merged": {"kv_map": str(kv), "embedding_store": {}}
                    },
                    "markers": {
                        "90": {
                            "control": {"ProvidersPrioritySelection": "TensorRT"},
                            "parameters": {"face": {}},
                        }
                    },
                    "target_faces_data": {"face": {"parameters": {}}},
                }
            )
        )
        return app_root, project, weight, kv, media

    def test_import_is_lightweight(self):
        script = (HARNESS / "native_smoke.py").read_text()
        before_main = script.split("def main(", 1)[0]
        self.assertNotIn("import torch", before_main)
        self.assertNotIn("from PySide6", before_main)
        self.assertNotIn("from app.", before_main)

    def test_fault_mode_refuses_a_timed_session(self):
        with self.assertRaises(SystemExit) as failure:
            benchmark.main(
                [
                    "--project",
                    str(self.root / "project.json"),
                    "--profile",
                    str(self.profile_path),
                    "--run-dir",
                    str(self.root / "run"),
                    "--repeat",
                    "1",
                    "--fault-probe",
                    "early-eof",
                    "--session-minutes",
                    "5",
                ]
            )
        self.assertEqual(failure.exception.code, 2)
        self.assertFalse((self.root / "run").exists())

    def test_missing_frame_probe_requires_preserved_output(self):
        with self.assertRaises(SystemExit) as failure:
            benchmark.main(
                [
                    "--project",
                    str(self.root / "project.json"),
                    "--profile",
                    str(self.profile_path),
                    "--run-dir",
                    str(self.root / "run"),
                    "--repeat",
                    "1",
                    "--fault-probe",
                    "missing-frame",
                ]
            )
        self.assertEqual(failure.exception.code, 2)
        self.assertFalse((self.root / "run").exists())

    def test_session_cycle_queues_distinct_frames_without_recording(self):
        calls = []
        steps = native.session_steps(
            [0, 30, 150],
            7,
            lambda *args: calls.append(("seek", *args)),
            lambda *args: calls.append(("play", *args)),
            lambda: calls.append(("check",)),
        )
        for _, action in steps:
            action()
        self.assertEqual(
            calls,
            [
                ("seek", 0, "session", 7),
                ("seek", 30, "session", 7),
                ("seek", 150, "session", 7),
                ("play", 7, "session"),
                ("check",),
            ],
        )

    def test_processed_paint_match_rejects_raw_and_wrong_frame_paints(self):
        paints = [
            {"native_frame_token": 1, "pixmap_key": 101, "slider_value": 150},
            {"native_frame_token": 2, "pixmap_key": 102, "slider_value": 30},
            {"native_frame_token": 2, "pixmap_key": 102, "slider_value": 150},
        ]
        self.assertIs(native.matching_processed_paint(paints, 2, 102, 150), paints[-1])
        self.assertIsNone(native.matching_processed_paint(paints[:2], 2, 102, 150))

    def test_example_profile_is_explicit_and_valid(self):
        self.assertEqual(self.profile["profile"]["provider"], "CUDA")
        self.assertEqual(benchmark.fixed_control(self.profile)["nThreadsSlider"], 2)

    def test_rejects_invented_global_precision(self):
        self.profile_path.write_text(
            self.profile_text.replace(
                'precision = "native-model-policy"', 'precision = "fp16"'
            )
        )
        with self.assertRaisesRegex(ValueError, "only supports"):
            benchmark.read_profile(self.profile_path)

    def test_requires_explicit_restorer_model(self):
        self.profile_path.write_text(
            self.profile_text.replace(
                "FaceRestorerEnableToggle = false", "FaceRestorerEnableToggle = true"
            )
        )
        with self.assertRaisesRegex(ValueError, "Enabled restorer"):
            benchmark.read_profile(self.profile_path)

    def test_native_manifest_is_read_without_model_imports(self):
        manifest = benchmark.model_manifest(ROOT)
        self.assertGreater(len(manifest), 60)
        self.assertEqual(manifest["inswapper_128.fp16.onnx"]["model"], "Inswapper128")
        self.assertEqual(
            manifest["InStyleSwapper256_Version_B.fp16.onnx"]["model"],
            "InStyleSwapper256 Version B",
        )

    def test_isolation_links_verified_weights_and_copies_mutable_references(self):
        source, project, weight, kv, media = self.source_tree()
        original = project.read_bytes()
        prepared = benchmark.prepare_runtime(
            source, project, self.root / "run", self.profile, reserve=0
        )
        runtime = Path(prepared["runtime"])
        self.assertTrue((runtime / "model_assets/test.onnx").is_symlink())
        cloned_kv = runtime / "model_assets/reference_kv_data/kv.pt"
        self.assertFalse(cloned_kv.is_symlink())
        cloned_kv.write_bytes(b"changed clone")
        self.assertEqual(kv.read_bytes(), b"reference state")
        cloned_project = json.loads(Path(prepared["workspace"]).read_text())
        self.assertNotEqual(
            cloned_project["target_medias_data"][0]["media_path"], str(media)
        )
        self.assertEqual(
            cloned_project["markers"]["90"]["control"]["ProvidersPrioritySelection"],
            "CUDA",
        )
        self.assertEqual(project.read_bytes(), original)
        self.assertEqual(
            (source / "last_workspace.json").read_text(), '{"original": true}'
        )

    def test_hash_mismatch_fails_without_linking_or_deleting(self):
        source, project, weight, _, _ = self.source_tree()
        weight.write_bytes(b"unexpected model")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            benchmark.prepare_runtime(
                source, project, self.root / "run", self.profile, reserve=0
            )
        self.assertEqual(weight.read_bytes(), b"unexpected model")

    def test_space_rejection_precedes_file_copies(self):
        source, project, weight, _, _ = self.source_tree()
        with mock.patch.object(
            benchmark.shutil, "disk_usage", return_value=SimpleNamespace(free=0)
        ):
            with self.assertRaisesRegex(RuntimeError, "Insufficient space"):
                benchmark.prepare_runtime(
                    source, project, self.root / "run", self.profile, reserve=0
                )
        self.assertFalse((self.root / "run/runtime").exists())
        self.assertTrue(weight.exists())

    def test_missing_workspace_reference_fails_clearly(self):
        source, project, _, _, media = self.source_tree()
        media.unlink()
        with self.assertRaisesRegex(ValueError, "missing file"):
            benchmark.prepare_runtime(
                source, project, self.root / "run", self.profile, reserve=0
            )

    def test_archive_provenance_verifies_real_code_and_patch_receipt(self):
        source, _, _, _, _ = self.source_tree()
        receipt = self.root / "patch.patch"
        receipt.write_text("real native patch\n")
        files = {
            "main.py": benchmark.sha256(source / "main.py"),
            "app/processors/models_data.py": benchmark.sha256(
                source / "app/processors/models_data.py"
            ),
        }
        (source / ".fusion-source.json").write_text(
            json.dumps(
                {
                    "source_sha": benchmark.BASELINE,
                    "source_patch_sha256": benchmark.sha256(receipt),
                    "files": files,
                }
            )
        )
        identity = benchmark.source_identity(source, receipt)
        self.assertEqual(identity["mode"], "verified-archive")
        self.assertNotIn("head", identity)
        self.assertEqual(identity["source_manifest_files_verified"], 2)
        extra = source / "app/unreviewed.py"
        extra.write_text("unreviewed native code\n")
        with self.assertRaisesRegex(ValueError, "absent from reviewed receipt"):
            benchmark.source_identity(source, receipt)
        extra.unlink()
        (source / "main.py").write_text("changed code\n")
        with self.assertRaisesRegex(ValueError, "differs from receipt"):
            benchmark.source_identity(source, receipt)

    def test_archive_without_receipt_cannot_invent_git_identity(self):
        source, _, _, _, _ = self.source_tree()
        with self.assertRaisesRegex(ValueError, "provenance manifest"):
            benchmark.source_identity(source)

    def test_output_requires_real_count_duration_audio_and_main10(self):
        video = self.root / "output.mp4"
        video.write_bytes(b"output")
        source = benchmark.video_metadata(sample_probe())
        with mock.patch.object(benchmark, "ffprobe", return_value=sample_probe()):
            self.assertTrue(
                benchmark.validate_output(video, source, 90, 3, False, (1280, 720))[
                    "valid"
                ]
            )
        trimmed = sample_probe(frames=89, duration=89 / 30)
        with mock.patch.object(benchmark, "ffprobe", return_value=trimmed):
            proof = benchmark.validate_output(video, source, 90, 3, False)
        self.assertTrue(proof["valid"])
        self.assertFalse(proof["exact_frame_count"])
        self.assertEqual(proof["frame_delta"], -1)
        bad = sample_probe(
            frames=4, duration=0.1, audio=False, codec="h264", pixels="yuv420p"
        )
        with mock.patch.object(benchmark, "ffprobe", return_value=bad):
            proof = benchmark.validate_output(video, source, 90, 3, False, (1920, 1080))
        self.assertFalse(proof["valid"])
        self.assertEqual(len(proof["errors"]), 5)

    def test_incomplete_filename_cannot_pass(self):
        video = self.root / "output_incomplete.mp4"
        video.write_bytes(b"output")
        source = benchmark.video_metadata(sample_probe())
        with mock.patch.object(benchmark, "ffprobe", return_value=sample_probe()):
            self.assertFalse(
                benchmark.validate_output(video, source, 90, 3, False)["valid"]
            )

    def test_hdr_cannot_pass_with_only_sdr_main10_metadata(self):
        video = self.root / "hdr.mp4"
        video.write_bytes(b"output")
        source = benchmark.video_metadata(sample_probe())
        probe = sample_probe()
        with mock.patch.object(benchmark, "ffprobe", return_value=probe):
            self.assertFalse(
                benchmark.validate_output(video, source, 90, 3, True)["valid"]
            )
        probe["streams"][0].update(color_primaries="bt2020", color_transfer="smpte2084")
        with mock.patch.object(benchmark, "ffprobe", return_value=probe):
            self.assertTrue(
                benchmark.validate_output(video, source, 90, 3, True)["valid"]
            )

    def test_state_comparison_tracks_assignment_marker_and_external_kv_changes(self):
        kv = self.root / "kv.pt"
        kv.write_bytes(b"a")
        state = {
            "embeddings_data": {"merge": {"kv_map": str(kv)}},
            "markers": {"9": {"parameters": {}}},
            "target_faces_data": {"face": {"assigned_input_faces": ["source"]}},
        }
        before = native.state_signature(state)
        kv.write_bytes(b"b")
        self.assertNotEqual(before, native.state_signature(state))
        state["target_faces_data"]["face"]["assigned_input_faces"] = []
        self.assertNotEqual(
            before["target_faces_data"],
            native.state_signature(state)["target_faces_data"],
        )

    def test_segment_expectations_are_inclusive_and_remove_deliberate_drops(self):
        window = SimpleNamespace(
            video_processor=SimpleNamespace(fps=30, max_frame_number=300),
            job_marker_pairs=[(0, 29), (60, 89)],
            dropped_frames={10, 70},
            control={},
        )
        frames, duration = native.expected_recording(window, {})
        self.assertEqual(frames, 58)
        self.assertAlmostEqual(duration, 58 / 30)

    def test_fps_cap_is_not_silently_measured_with_wrong_expected_frames(self):
        window = SimpleNamespace(
            video_processor=SimpleNamespace(fps=30),
            control={"OutputFpsCapEnableToggle": True},
        )
        with self.assertRaisesRegex(ValueError, "not implemented"):
            native.expected_recording(window, {})

    def test_ordering_diagnostic_rejects_reordering_with_correct_frame_count(self):
        probe = {
            "highest_frame": 2,
            "signals_emitted": [{"frame": 2}, {"frame": 1}],
            "signals_delivered": [{"frame": 2}, {"frame": 1}],
            "encoder_frame_indices": [0, 1, 2],
            "natural_eof_observations": [
                {"pending_tasks": 0, "is_drain_complete": True}
            ],
        }
        proof = [{"exact_frame_count": True}]
        self.assertEqual(native.out_of_order_errors(probe, proof), [])
        probe["encoder_frame_indices"] = [0, 2, 1]
        self.assertIn(
            "Native encoder frame order differed from source order",
            native.out_of_order_errors(probe, proof),
        )
        probe["encoder_frame_indices"] = [0, 1, 2]
        probe["natural_eof_observations"][0]["pending_tasks"] = 1
        self.assertIn(
            "Natural EOF zero unfinished-task gate was not directly observed",
            native.out_of_order_errors(probe, proof),
        )

    def test_restore_preparation_preserves_saved_parameters_and_markers(self):
        source, project, _, _, _ = self.source_tree()
        saved = json.loads(project.read_text())
        saved["target_faces_data"]["face"]["parameters"] = {
            "FaceRestorerEnableToggle": True,
            "custom": 37,
        }
        saved["current_widget_parameters"] = {"custom": 41}
        saved["markers"]["90"]["parameters"]["face"] = {"custom": 43}
        project.write_text(json.dumps(saved))
        prepared = benchmark.prepare_runtime(
            source,
            project,
            self.root / "restore",
            self.profile,
            reserve=0,
            preserve_settings=True,
        )
        cloned = json.loads(Path(prepared["workspace"]).read_text())
        self.assertEqual(cloned["target_faces_data"], saved["target_faces_data"])
        self.assertEqual(
            cloned["current_widget_parameters"], saved["current_widget_parameters"]
        )
        self.assertEqual(
            cloned["markers"]["90"]["parameters"], saved["markers"]["90"]["parameters"]
        )
        self.assertEqual(cloned["control"]["ProvidersPrioritySelection"], "TensorRT")
        self.assertNotEqual(
            cloned["control"]["OutputMediaFolder"],
            saved["control"].get("OutputMediaFolder"),
        )
        self.assertEqual(benchmark.sha256(project), prepared["original_project_sha256"])

    def test_gpu_safety_boundary_tracks_only_the_native_child(self):
        for own_memory, should_stop in [(850, False), (851, True), (None, False)]:
            with self.subTest(own_memory=own_memory):
                stop_path = self.root / "safety.json"
                stop_path.unlink(missing_ok=True)
                sampler = benchmark.ResourceSampler(
                    0, max_native_gpu_percent=85, stop_request=stop_path
                )
                sampler.pid = 999999999  # Absent /proc entry: CPU-only observation.
                process_rows = "233, 990\n"
                if own_memory is not None:
                    process_rows += f"{sampler.pid}, {own_memory}\n"
                with (
                    mock.patch.object(
                        sampler.stopping, "is_set", side_effect=[False, True]
                    ),
                    mock.patch.object(sampler.stopping, "wait"),
                    mock.patch.object(
                        benchmark,
                        "command",
                        side_effect=[
                            "GPU-id, Test GPU, 1, 1000, 999, 50",
                            process_rows,
                        ],
                    ),
                ):
                    sampler._sample()
                self.assertFalse(sampler.errors)
                self.assertEqual(stop_path.is_file(), should_stop)
                self.assertEqual(sampler.stop_evidence is not None, should_stop)
                if should_stop:
                    evidence = json.loads(stop_path.read_text())
                    self.assertEqual(
                        evidence["sample"]["native_process_gpu_used_mib"], 851
                    )
                self.assertEqual(len(sampler.samples), 1)

    def test_percentiles_do_not_invent_missing_measurements(self):
        self.assertIsNone(benchmark.percentile([], 0.95))
        self.assertEqual(benchmark.percentile([1, 2, 3], 0.5), 2)


if __name__ == "__main__":
    unittest.main()
