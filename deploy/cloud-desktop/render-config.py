#!/usr/bin/env python3
"""Render fixed templates with validated runtime paths/ports; never print secrets."""

import os
from pathlib import Path
import re

directory = Path(os.environ["FUSION_DEPLOY_DIR"])
values = {
    name: os.environ[name]
    for name in (
        "FUSION_ROOT",
        "FUSION_RUN_DIR",
        "FUSION_DEPLOY_DIR",
        "FUSION_HTTP_BIND",
        "FUSION_HTTP_PORT",
        "FUSION_WEBSOCKIFY_PORT",
        "FUSION_VNC_PORT",
        "FUSION_DESKTOP_GEOMETRY",
        "DISPLAY",
        "XAUTHORITY",
        "FUSION_SSH_AUTOSTART",
        "FUSION_AUDIO_AUTOSTART",
    )
}
for name, value in values.items():
    if not re.fullmatch(r"[A-Za-z0-9_./:-]+", value):
        raise RuntimeError(
            f"Unsupported character in {name}; refusing unsafe config interpolation"
        )
values["FUSION_AUTH_DIRECTIVES"] = os.environ["FUSION_AUTH_DIRECTIVES"]
values["FUSION_NGINX_USER"] = "user root;" if os.geteuid() == 0 else ""
for source, destination in (
    ("nginx.conf", "FUSION_NGINX_CONFIG"),
    ("supervisord.conf", "FUSION_SUPERVISOR_CONFIG"),
):
    content = (directory / source).read_text()
    for name, value in values.items():
        content = content.replace("@" + name + "@", value)
    if re.search(r"@[A-Z_]+@", content):
        raise RuntimeError("Unresolved service configuration token")
    path = Path(os.environ[destination])
    path.write_text(content)
    path.chmod(0o600)
