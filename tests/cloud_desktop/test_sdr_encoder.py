"""Protect default encoding and explicit compatibility-mode failures."""

import subprocess
import unittest
from unittest.mock import Mock

from app.processors.video_utils.encoder_policy import (
    sdr_encoder_arguments,
    sdr_encoder_settings,
)

encoder_class: type | None = None
try:
    from app.processors.video_utils.video_encoding import (
        FFmpegEncoder as _FFmpegEncoder,
    )

    encoder_class = _FFmpegEncoder
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise


class SDRPolicyTests(unittest.TestCase):
    def test_default_preserves_stock_nvenc_arguments(self):
        self.assertEqual(
            sdr_encoder_arguments({}, 20, {}),
            [
                "-c:v",
                "hevc_nvenc",
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
                "-tag:v",
                "hvc1",
            ],
        )

    def test_explicit_cpu_mode_ignores_nvenc_controls_and_hdr(self):
        args = sdr_encoder_arguments(
            {
                "FFPresetsSDRSelection": "p7",
                "FFSpatialAQToggle": True,
                "FFTemporalAQToggle": True,
            },
            18,
            {"VISOMASTER_SDR_ENCODER": "libx265"},
        )
        self.assertIn("veryfast", args)
        self.assertIn("-crf", args)
        self.assertNotIn("-cq", args)
        self.assertNotIn("-spatial-aq", args)
        self.assertNotIn("p7", args)
        self.assertNotIn("smpte2084", args)
        self.assertEqual(args[args.index("-colorspace") + 1], "bt709")

    def test_invalid_configuration_does_not_fall_back(self):
        with self.assertRaises(ValueError):
            sdr_encoder_settings({"VISOMASTER_SDR_ENCODER": "typo"})
        with self.assertRaises(ValueError):
            sdr_encoder_settings(
                {
                    "VISOMASTER_SDR_ENCODER": "libx265",
                    "VISOMASTER_SDR_X265_PRESET": "p4",
                }
            )


@unittest.skipIf(encoder_class is None, "Native NumPy dependency unavailable")
class EncoderFinalizationTests(unittest.TestCase):
    def encoder(self, exit_code):
        assert encoder_class is not None
        encoder = encoder_class()
        process = Mock()
        process.stdin.closed = False
        process.returncode = exit_code
        process.wait.return_value = exit_code
        encoder.recording_sp = process
        return encoder, process

    def test_failed_already_exited_encoder_cannot_become_success(self):
        encoder, process = self.encoder(187)
        process.poll.return_value = 187
        self.assertFalse(encoder.is_running())
        self.assertFalse(encoder.close_process())
        self.assertFalse(encoder.close_process())
        self.assertEqual(encoder.last_exit_code, 187)
        process.stdin.close.assert_called_once()

    def test_clean_exit_is_idempotently_successful(self):
        encoder, process = self.encoder(0)
        self.assertTrue(encoder.close_process())
        self.assertTrue(encoder.close_process())
        process.wait.assert_called_once_with(timeout=120)

    def test_timeout_followed_by_zero_exit_remains_failure(self):
        encoder, process = self.encoder(0)
        process.wait.side_effect = [subprocess.TimeoutExpired("ffmpeg", 120), 0]
        self.assertFalse(encoder.close_process())
        self.assertFalse(encoder.close_process())
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
