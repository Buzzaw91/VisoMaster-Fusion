"""Cooperative startup exclusion for staged cloud trees; ordinary installs are unchanged."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import IO


def _ancestor_pids() -> set[int]:
    result = set()
    pid = os.getpid()
    for _ in range(100):
        result.add(pid)
        if pid <= 1:
            break
        stat = Path(f"/proc/{pid}/stat")
        if stat.is_file():
            pid = int(stat.read_text().rsplit(")", 1)[1].split()[1])
        else:
            pid = int(
                subprocess.check_output(
                    ["ps", "-o", "ppid=", "-p", str(pid)], text=True
                ).strip()
            )
    return result


def acquire_cloud_instance(app_root: Path) -> IO[str] | None:
    """Return an application-held lock, or reject an unguarded managed launch.

    This protects cooperative native launches and survives a wrapper crash. It
    does not restrict arbitrary code run by a user with filesystem/root access.
    Keep the returned handle open for the entire native application's lifetime.
    """
    app_root = app_root.resolve()
    if not (app_root / ".fusion-source.json").is_file():
        return None
    root = app_root.parents[1]
    try:
        marker = json.loads((root / ".writer-state.json").read_text())
        owner = int(os.environ.get("FUSION_LOCK_OWNER_PID", "-1"))
        wrapper = int(os.environ.get("FUSION_NATIVE_GUARD_PID", "-1"))
        ancestors = _ancestor_pids()
        if (
            marker.get("active") is not True
            or marker.get("unclean_fusion")
            or marker.get("role") not in {"desktop", "fusion"}
            or marker.get("pid") != owner
            or marker.get("token") != os.environ.get("FUSION_LOCK_TOKEN")
            # A container entrypoint guard can legitimately be PID 1.
            or owner < 1
            or wrapper < 1
            or owner not in ancestors
            or wrapper not in ancestors
        ):
            raise RuntimeError("Missing live Fusion wrapper authorization")
        import fcntl

        handle = (root / ".native-instance.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            handle.close()
            raise
        handle.seek(0)
        handle.truncate()
        json.dump(
            {"pid": os.getpid(), "wrapper_pid": wrapper, "owner_pid": owner}, handle
        )
        handle.flush()
        return handle
    except Exception as exc:
        raise RuntimeError(
            "Managed cloud Fusion must start through launch-fusion.sh; "
            "another instance or an unrecovered failure may own this storage root."
        ) from exc
