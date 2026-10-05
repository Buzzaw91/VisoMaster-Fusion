"""Atomic writes for native state, including paths mapped through symlinks."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any


def _sync_directory(directory: Path) -> None:
    """Persist the rename where directory fsync is supported by the platform."""
    if os.name == "nt":
        return
    unsupported = {errno.EINVAL, errno.ENOTSUP, errno.EBADF}
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError as exc:
        if exc.errno in unsupported:
            return
        raise
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in unsupported:
                raise
    finally:
        os.close(fd)


def atomic_write_bytes(destination: str | Path, content: bytes) -> None:
    """Flush a temporary file beside the resolved target, then replace it.

    Resolving first preserves existing and dangling symlinks used to map native
    state onto persistent storage. The destination's parent must already exist.
    Failures before replacement leave the previous file intact. A failure to
    sync the directory after replacement is reported because durability is then
    uncertain, although the new file may already be visible.
    """
    target = Path(destination).resolve(strict=False)
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        mode = None

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            if mode is not None:
                os.chmod(temporary, mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _sync_directory(target.parent)
    finally:
        # Only our unpublished temporary file is eligible for removal.
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    destination: str | Path, payload: Any, *, indent: int = 4
) -> None:
    """Serialize fully before touching the destination or creating a temp file."""
    content = json.dumps(payload, indent=indent).encode("utf-8")
    atomic_write_bytes(destination, content)
