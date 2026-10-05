"""Explicit SDR encoding compatibility settings; native NVENC remains default."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


X265_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
}


def sdr_encoder_settings(environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    env = os.environ if environ is None else environ
    encoder = env.get("VISOMASTER_SDR_ENCODER", "hevc_nvenc")
    if encoder not in {"hevc_nvenc", "libx265"}:
        raise ValueError("VISOMASTER_SDR_ENCODER must be hevc_nvenc or libx265")
    preset = env.get("VISOMASTER_SDR_X265_PRESET", "veryfast")
    if encoder == "libx265" and preset not in X265_PRESETS:
        raise ValueError("Invalid VISOMASTER_SDR_X265_PRESET")
    return encoder, preset


def sdr_encoder_arguments(
    control: Mapping[str, Any],
    quality: int,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    encoder, preset = sdr_encoder_settings(environ)
    if encoder == "libx265":
        # Retain native SDR range/scale conversion in the caller. CQ and CRF
        # are distinct quality scales; this branch is an explicit CPU mode.
        return [
            "-c:v",
            "libx265",
            "-preset",
            preset,
            "-profile:v",
            "main10",
            "-crf",
            str(quality),
            "-pix_fmt",
            "yuv420p10le",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-tag:v",
            "hvc1",
        ]
    # Preserve every stock SDR encoder argument and its ordering.
    return [
        "-c:v",
        "hevc_nvenc",
        "-preset",
        str(control.get("FFPresetsSDRSelection", "p4")),
        "-profile:v",
        "main10",
        "-cq",
        str(quality),
        "-pix_fmt",
        "yuv420p10le",
        "-colorspace",
        "rgb",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-spatial-aq",
        str(int(control.get("FFSpatialAQToggle", 0))),
        "-temporal-aq",
        str(int(control.get("FFTemporalAQToggle", 0))),
        "-tier",
        "high",
        "-tag:v",
        "hvc1",
    ]
