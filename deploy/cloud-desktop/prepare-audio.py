#!/usr/bin/env python3
"""Configure a private Pulse server and verify the installed Xpra audio stack."""

import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys

from runtime import atomic_json, validate_owner


def prepare(root, run_dir):
    if os.environ.get("FUSION_ACCESS_MODE", "tunnel") != "tunnel":
        raise RuntimeError("Xpra is supported only through a localhost SSH tunnel")
    for binary in ("xpra", "pulseaudio", "pactl", "paplay"):
        if not shutil.which(binary):
            raise RuntimeError(f"Audio transport dependency missing: {binary}")
    # Use Ubuntu's system interpreter, not the application's isolated venv.
    probe = subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; "
            "Gst.init(None); "
            "from xpra.sound.gstreamer_util import import_gst, get_encoders, get_source_plugins; "
            "assert import_gst() is not None; "
            "assert 'pulsesrc' in get_source_plugins(), 'Pulse capture unavailable'; "
            "assert any(x in get_encoders() for x in ('opus+ogg','vorbis+ogg','mp3')), "
            "'No HTML5 audio encoder available'",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        raise RuntimeError(
            "Xpra/GStreamer audio probe failed; install the documented Ubuntu audio packages"
        )
    if (
        not Path("/usr/share/xpra/www/index.html").is_file()
        or not Path("/usr/share/xpra/www/js/lib/jquery.js").is_file()
    ):
        raise RuntimeError("Xpra HTML5 client is missing")
    # Only generated paths are interpolated into Pulse's configuration language.
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(run_dir)):
        raise RuntimeError("Unsupported audio runtime path")
    pulse = run_dir / "pulse"
    pulse.mkdir(parents=True, exist_ok=True)
    pulse.chmod(0o700)
    cookie = pulse / "cookie"
    if cookie.is_symlink():
        raise RuntimeError("Pulse cookie must not be a symlink")
    if not cookie.exists():
        with cookie.open("xb") as target:
            target.write(secrets.token_bytes(256))
    if len(cookie.read_bytes()) != 256:
        raise RuntimeError("Invalid existing Pulse authentication cookie")
    cookie.chmod(0o600)
    config = pulse / "default.pa"
    config.write_text(
        ".fail\n"
        "load-module module-null-sink sink_name=fusion_preview "
        "rate=48000 channels=2 sink_properties=device.description=FusionPreview\n"
        f"load-module module-native-protocol-unix socket={pulse}/native "
        f"auth-cookie={cookie} auth-cookie-enabled=yes auth-anonymous=no\n"
        "set-default-sink fusion_preview\n"
        "set-default-source fusion_preview.monitor\n"
    )
    config.chmod(0o600)
    for child in (run_dir / "pulse-state", run_dir / "xpra", run_dir / "xpra-config"):
        child.mkdir(parents=True, exist_ok=True)
        child.chmod(0o700)
    version = subprocess.run(
        ["xpra", "--version"], capture_output=True, text=True, check=True
    )
    atomic_json(
        root / "receipts" / "audio-transport.json",
        {
            "transport": "xpra-html5-shadow",
            "xpra_version": version.stdout.strip(),
            "listen_address": "127.0.0.1",
            "listen_port": int(os.environ.get("FUSION_XPRA_PORT", "6082")),
            "pulse_server": f"unix:{pulse}/native",
            "sink": "fusion_preview",
            "capture_source": "fusion_preview.monitor",
            "microphone_forwarding": False,
            "browser_audio_playback_validated": False,
            "native_preview_audio_validated": False,
            "browser_disconnect_stops_app": False,
        },
    )


def main():
    if os.environ.get("FUSION_TRANSPORT", "xpra") != "xpra":
        raise RuntimeError("Audio setup requires FUSION_TRANSPORT=xpra")
    root = Path(os.environ.get("FUSION_ROOT", "/workspace/fusion")).resolve()
    validate_owner(root)
    prepare(
        root, Path(os.environ.get("FUSION_RUN_DIR", "/run/fusion-desktop")).resolve()
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Fusion audio transport refused: {error}", file=sys.stderr)
        sys.exit(78)
