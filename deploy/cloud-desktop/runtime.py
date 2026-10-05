#!/usr/bin/env python3
"""Deployment-only state, source, and single-writer guard (Python 3.12 stdlib).

No face-processing code is patched. A failed/terminated writer keeps a recovery
marker even after flock is released. Recovery is an explicit operator action.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

SOURCE_SHA = "f2d6f5ebe190b631ee5c8972332083cf7f8a3e72"
DEFAULT_ROOT = "/workspace/fusion"
MAPPED = (
    "model_assets",
    "last_workspace.json",
    "jobs",
    ".jobs",
    "presets",
    "tensorrt-engines",
    "temp_files",
    "crash_logs",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
        temp = Path(out.name)
    os.chmod(temp, 0o600)
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path):
    return json.loads(Path(path).read_text())


def root_path(value):
    root = Path(value).expanduser().absolute()
    if root.is_symlink():
        raise RuntimeError("Storage root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def probe_flock(root):
    """Prove that another process cannot acquire this volume's same lock."""
    path = root / ".flock-probe"
    with path.open("a+") as out:
        fcntl.flock(out, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = "import fcntl,sys; f=open(sys.argv[1],'a+');\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(73)\nsys.exit(0)"
        result = subprocess.run([sys.executable, "-c", code, str(path)], close_fds=True)
        if result.returncode != 73:
            raise RuntimeError(
                "Storage flock exclusion probe failed; refusing concurrent state writers"
            )


def process_identity(pid):
    """Linux identity survives numeric PID reuse across container recreation."""
    if sys.platform != "linux":
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid_namespace_inode": os.stat(f"/proc/{pid}/ns/pid").st_ino,
            "start_ticks": int(stat[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except FileNotFoundError:
        return None


def validate_process_identity(identity):
    if (
        not isinstance(identity, dict)
        or set(identity) != {"pid_namespace_inode", "start_ticks", "boot_id"}
        or type(identity.get("pid_namespace_inode")) is not int
        or identity["pid_namespace_inode"] <= 0
        or type(identity.get("start_ticks")) is not int
        or identity["start_ticks"] < 0
        or not isinstance(identity.get("boot_id"), str)
        or not re.fullmatch(
            r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", identity["boot_id"]
        )
    ):
        raise RuntimeError(
            "Invalid recorded writer process identity; inspect before recovery"
        )


def recorded_owner_is_alive(info):
    pid = info.get("pid")
    if type(pid) is not int or pid <= 0:
        raise RuntimeError("Invalid recorded writer PID")
    recorded = info.get("process_identity")
    if recorded is not None:
        validate_process_identity(recorded)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if recorded is None:
        # Older markers and non-Linux CPU tests retain conservative PID checks.
        return True
    current = process_identity(pid)
    if current is None:
        if sys.platform != "linux":
            raise RuntimeError(
                "Linux writer identity cannot be verified on this platform"
            )
        # The process disappeared between kill(0) and the procfs read.
        return False
    validate_process_identity(current)
    return current == recorded


def validate_owner(root):
    """Validate a live guard and process ancestry even if supervisor closes FDs."""
    info = read_json(root / ".writer-state.json")
    if info.get("unclean_fusion"):
        raise RuntimeError(
            "Unclean Fusion child; shut down the desktop, inspect snapshots and recover the root lock before relaunching"
        )
    owner = int(os.environ.get("FUSION_LOCK_OWNER_PID", "-1"))
    if (
        info.get("active") is not True
        or info.get("pid") != owner
        or info.get("token") != os.environ.get("FUSION_LOCK_TOKEN")
    ):
        raise RuntimeError("Invalid inherited writer authorization")
    os.kill(owner, 0)
    if info.get("process_identity") is not None:
        validate_process_identity(info["process_identity"])
        if process_identity(owner) != info["process_identity"]:
            raise RuntimeError("Inherited writer process identity changed")
    pid = os.getpid()
    for _ in range(100):
        if pid == owner:
            return info
        if pid <= 1:
            break
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            pid = int(stat.read_text().rsplit(")", 1)[1].split()[1])
        else:
            pid = int(
                subprocess.check_output(
                    ["ps", "-o", "ppid=", "-p", str(pid)], text=True
                ).strip()
            )
    raise RuntimeError("Writer authorization is not from a live ancestor guard")


def probe_native_instance(root):
    """A live native app keeps this lock even if its launcher is killed."""
    with (root / ".native-instance.lock").open("a+") as native_lock:
        try:
            fcntl.flock(native_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "A live native Fusion process holds its instance lock; refusing setup, launch or recovery"
            ) from exc


def guard(root, role, command):
    root = root_path(root)
    marker = root / ".writer-state.json"
    lock = root / ".writer.lock"
    inherited = os.environ.get("FUSION_LOCK_FD")
    if inherited:
        info = validate_owner(root)
        # supervisord closes inherited descriptors in children. The live ancestor
        # owns flock for their whole lifetime; its unguessable token + ancestry
        # authorizes one Fusion child, guarded independently below.
        fd = int(inherited)
        try:
            valid_fd = (
                os.fstat(fd).st_ino == lock.stat().st_ino
                and os.fstat(fd).st_dev == lock.stat().st_dev
            )
        except OSError:
            valid_fd = False
        if role != "fusion":
            raise RuntimeError("Only Fusion may use the desktop owner's inherited lock")
        if info.get("role") != "desktop":
            raise RuntimeError("Inherited Fusion launch requires a desktop guard owner")
        probe_native_instance(root)
        # Independently reject two Fusion children of the same desktop owner.
        with (root / ".fusion-process.lock").open("a+") as child_lock:
            fcntl.flock(child_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run_command(
                command, pass_fd=fd if valid_fd else None, native_guard=True
            )
            if result != 0:
                info = read_json(marker)
                info["unclean_fusion"] = True
                atomic_json(marker, info)
            return result
    with lock.open("a+") as out:
        try:
            fcntl.flock(out, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another installer/desktop/Fusion owns this entire storage root"
            ) from exc
        probe_native_instance(root)
        if marker.exists():
            previous = read_json(marker)
            if (
                previous.get("active") is not False
                or previous.get("source_sha") != SOURCE_SHA
                or not isinstance(previous.get("token"), str)
                or len(previous["token"]) != 32
                or previous.get("role") not in ("setup", "desktop", "fusion")
            ):
                raise RuntimeError(
                    "Unclean or unknown writer state; inspect .writer-state.json and use recover-lock with its exact token"
                )
        probe_flock(root)
        token = uuid.uuid4().hex
        identity = process_identity(os.getpid())
        if sys.platform == "linux" and identity is None:
            raise RuntimeError("Cannot record Linux writer process identity")
        if identity is not None:
            validate_process_identity(identity)
        info = {
            "active": True,
            "pid": os.getpid(),
            "role": role,
            "token": token,
            "started_unix": time.time(),
            "source_sha": SOURCE_SHA,
            "process_identity": identity,
        }
        atomic_json(marker, info)
        os.set_inheritable(out.fileno(), True)
        env = os.environ.copy()
        env.update(
            FUSION_LOCK_FD=str(out.fileno()),
            FUSION_LOCK_TOKEN=token,
            FUSION_LOCK_OWNER_PID=str(os.getpid()),
        )
        result = run_command(
            command, env=env, pass_fd=out.fileno(), native_guard=role == "fusion"
        )
        current = read_json(marker)
        if result == 0 and not current.get("unclean_fusion"):
            current.update(active=False, ended_unix=time.time())
            atomic_json(marker, current)
        return result


def run_command(command, env=None, pass_fd=None, native_guard=False):
    if not command:
        raise RuntimeError("A guarded command is required")
    environment = (env if env is not None else os.environ).copy()
    if native_guard:
        environment["FUSION_NATIVE_GUARD_PID"] = str(os.getpid())
    else:
        environment.pop("FUSION_NATIVE_GUARD_PID", None)
    process = subprocess.Popen(
        command, env=environment, pass_fds=(() if pass_fd is None else (pass_fd,))
    )
    interrupted = []

    def forward(sig, _frame):
        interrupted.append(sig)
        process.send_signal(sig)

    old = {
        sig: signal.signal(sig, forward)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    try:
        result = process.wait()
        return (
            128 + interrupted[0]
            if interrupted
            else (result if result >= 0 else 128 - result)
        )
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def recover_lock(root, token):
    root = root_path(root)
    with (root / ".writer.lock").open("a+") as out:
        fcntl.flock(out, fcntl.LOCK_EX | fcntl.LOCK_NB)
        probe_native_instance(root)
        path = root / ".writer-state.json"
        info = read_json(path)
        if info.get("token") != token:
            raise RuntimeError(
                "Recovery token does not match the inspected writer marker"
            )
        # Exact-token recovery may distinguish a new container's unrelated PID1.
        # Kernel writer/native locks still exclude an owner in another namespace.
        if recorded_owner_is_alive(info):
            raise RuntimeError("Recorded writer PID is alive; refusing lock recovery")
        shutil.copy2(path, root / f"writer-recovery-{time.time_ns()}.json")
        info.update(
            active=False,
            recovered_unix=time.time(),
            recovery_required_snapshot_review=True,
            recovery_actor_identity=process_identity(os.getpid()),
        )
        atomic_json(path, info)


def stage_source(root, repository, source_patch=None, replace_source=False):
    root = root_path(root)
    target = root / "source" / SOURCE_SHA
    patch_sha = digest(source_patch) if source_patch else None
    if target.exists():
        verify_source(target)
        previous = read_json(target / ".fusion-source.json")
        if previous.get("source_patch_sha256") == patch_sha:
            return target
        if not replace_source:
            raise RuntimeError(
                "Requested source overlay differs; explicit --replace-source required after clean shutdown/snapshot review"
            )
    sha = subprocess.check_output(
        ["git", "-C", repository, "rev-parse", SOURCE_SHA], text=True
    ).strip()
    if sha != SOURCE_SHA:
        raise RuntimeError("Source repository does not contain the pinned commit")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".stage-", dir=target.parent))
    try:
        archive = subprocess.Popen(
            ["git", "-C", repository, "archive", SOURCE_SHA], stdout=subprocess.PIPE
        )
        extraction = subprocess.run(
            ["tar", "-xf", "-", "-C", str(temporary)], stdin=archive.stdout
        )
        archive.stdout.close()
        if extraction.returncode or archive.wait():
            raise RuntimeError("Pinned source archive failed")
        if source_patch:
            source_patch = str(Path(source_patch).resolve())
            numstat = subprocess.check_output(
                ["git", "apply", "--numstat", "-z", source_patch]
            )
            for entry in numstat.split(b"\0"):
                if not entry:
                    continue
                fields = entry.decode().split("\t", 2)
                if len(fields) != 3:
                    raise RuntimeError("Unrecognized source patch path record")
                relative = Path(fields[2])
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0] in (*MAPPED, ".git")
                    or fields[2].startswith(".fusion-")
                ):
                    raise RuntimeError(
                        f"Source patch may not modify native state or unsafe paths: {relative}"
                    )
            subprocess.run(
                ["git", "-C", str(temporary), "apply", "--check", source_patch],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(temporary), "apply", "--binary", source_patch],
                check=True,
            )
        files = {
            str(p.relative_to(temporary)): digest(p)
            for p in temporary.rglob("*")
            if p.is_file() and p.relative_to(temporary).parts[0] not in MAPPED
        }
        seeds = {
            str(p.relative_to(temporary)): digest(p)
            for p in (temporary / "model_assets").rglob("*")
            if p.is_file()
        }
        atomic_json(
            temporary / ".fusion-source.json",
            {
                "source_sha": SOURCE_SHA,
                "source_patch_sha256": patch_sha,
                "files": files,
                "seed_files": seeds,
            },
        )
        if target.exists():
            archives = root / "source-archives"
            archives.mkdir(exist_ok=True)
            os.rename(target, archives / f"{SOURCE_SHA}-{time.time_ns()}")
        os.rename(temporary, target)
        if source_patch:
            receipt_path = root / "receipts" / f"source-patch-{patch_sha}.patch"
            receipt_path.parent.mkdir(exist_ok=True)
            if not receipt_path.exists():
                shutil.copy2(source_patch, receipt_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def verify_source(app):
    metadata = read_json(app / ".fusion-source.json")
    if metadata.get("source_sha") != SOURCE_SHA or not metadata.get("files"):
        raise RuntimeError(
            "Existing source has unknown provenance; refusing to overwrite"
        )
    for relative, expected in metadata["files"].items():
        path = app / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise RuntimeError(f"Pinned source was changed or lost: {relative}")
    if not (app / "model_assets").is_symlink():
        for relative, expected in metadata.get("seed_files", {}).items():
            if digest(app / relative) != expected:
                raise RuntimeError(f"Pinned seed asset was changed: {relative}")


def map_link(path, target):
    target = target.absolute()
    if path.is_symlink():
        if Path(os.readlink(path)).absolute() != target:
            raise RuntimeError(f"Unexpected existing native mapping: {path}")
        return
    if path.exists():
        raise RuntimeError(
            f"Unmapped native state exists at {path}; migrate explicitly before setup"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=target.is_dir())


def map_state(root, cache_key):
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", cache_key):
        raise RuntimeError("Invalid cache compatibility key")
    root = root_path(root)
    app = root / "source" / SOURCE_SHA
    verify_source(app)
    native = root / "native" / SOURCE_SHA
    assets = root / "assets" / SOURCE_SHA / "model_assets"
    cache = root / "cache" / cache_key
    for folder in (
        native,
        assets,
        cache,
        root / "projects",
        root / "profile",
        root / "logs",
    ):
        folder.mkdir(parents=True, exist_ok=True)
    seed = app / "model_assets"
    if not seed.is_symlink():
        # Tracked seeds only; existing user/reference/model files are never replaced.
        if not seed.is_dir():
            raise RuntimeError("Pinned source model seed folder missing")
        for item in seed.rglob("*"):
            relative = item.relative_to(seed)
            dest = assets / relative
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            elif item.is_file():
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest)
                elif digest(item) != digest(dest):
                    raise RuntimeError(
                        f"Existing asset conflicts with pinned seed: {relative}"
                    )
        # Removal affects only the freshly verified pinned seed copy, never assets.
        shutil.rmtree(seed)
    map_link(seed, assets)
    for name in ("jobs", ".jobs", "presets"):
        destination = native / name
        destination.mkdir(exist_ok=True)
        map_link(app / name, destination)
    map_link(app / "last_workspace.json", native / "last_workspace.json")
    for name in ("tensorrt-engines", "torch_compile_cache", "temp_files"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    # Cache changes are permitted only while the root writer lock is held.
    for path, dest in (
        (app / "tensorrt-engines", cache / "tensorrt-engines"),
        (assets / "torch_compile_cache", cache / "torch_compile_cache"),
        (app / "temp_files", cache / "temp_files"),
    ):
        if path.is_symlink() and Path(os.readlink(path)).absolute() != dest.absolute():
            path.unlink()
        map_link(path, dest)
    (root / "logs" / "crashes").mkdir(exist_ok=True)
    map_link(app / "crash_logs", root / "logs" / "crashes")
    (assets / "reference_kv_data").mkdir(exist_ok=True)
    atomic_json(
        root / "runtime-state.json",
        {
            "source_sha": SOURCE_SHA,
            "app": str(app),
            "cache_key": cache_key,
            "preview_audio": "not supplied by noVNC",
        },
    )
    return app


def check_storage(root, minimum_gb):
    root = root_path(root)
    with tempfile.NamedTemporaryFile(dir=root) as probe:
        probe.write(b"fusion-storage-probe")
        probe.flush()
        os.fsync(probe.fileno())
        probe.seek(0)
        if probe.read() != b"fusion-storage-probe":
            raise RuntimeError("Persistent storage readback failed")
    backing_free = shutil.disk_usage(root).free
    effective_free = backing_free
    quota = None
    if os.environ.get("FUSION_VOLUME_QUOTA_FREE_GB") is not None:
        declared_gb = float(os.environ["FUSION_VOLUME_QUOTA_FREE_GB"])
        if not math.isfinite(declared_gb) or declared_gb < 0:
            raise RuntimeError(
                "FUSION_VOLUME_QUOTA_FREE_GB must be a finite nonnegative operator observation"
            )
        declared = int(declared_gb * 1024**3)
        observed_at = os.environ.get(
            "FUSION_VOLUME_QUOTA_CHECKED_AT", "operator time not supplied"
        )
        ledger = root / "receipts" / "quota-consumption.json"
        consumed = 0
        if ledger.exists():
            previous = read_json(ledger)
            if (
                previous.get("declared_free_bytes") == declared
                and previous.get("operator_checked_at") == observed_at
            ):
                consumed = int(previous.get("download_bytes_since_observation", 0))
        quota = {
            "declared_free_bytes_upper_bound": declared,
            "operator_checked_at": observed_at,
            "download_bytes_since_observation": consumed,
            "remaining_bytes_upper_bound": max(0, declared - consumed),
            "authority": "manual static observation; other writers/allocations can reduce real remaining quota",
        }
        effective_free = min(backing_free, quota["remaining_bytes_upper_bound"])
    if effective_free < minimum_gb * 1024**3:
        raise RuntimeError(
            f"Storage bound has {effective_free / 1024**3:.2f} GiB available; requires {minimum_gb} GiB (backing free is not volume quota)"
        )
    probe_flock(root)
    return {
        "root": str(root),
        "backing_filesystem_free_bytes": backing_free,
        "backing_free_is_not_allocated_volume_quota": True,
        "allocated_volume_quota": quota
        or "unknown; inspect actual provider quota before new allocations",
        "effective_available_bytes_upper_bound": effective_free,
        "checked_unix": time.time(),
        "flock_exclusion": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("FUSION_ROOT", DEFAULT_ROOT))
    sub = parser.add_subparsers(dest="action", required=True)
    guarded = sub.add_parser("guard")
    guarded.add_argument(
        "--role", choices=("setup", "desktop", "fusion"), required=True
    )
    guarded.add_argument("command", nargs=argparse.REMAINDER)
    assertion = sub.add_parser("assert-guard")
    assertion.add_argument(
        "--owner-role",
        choices=("setup", "desktop", "fusion"),
        action="append",
        required=True,
    )
    recovery = sub.add_parser("recover-lock")
    recovery.add_argument("--expect-token", required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("--repository", required=True)
    stage.add_argument("--source-patch", type=Path)
    stage.add_argument("--replace-source", action="store_true")
    mapping = sub.add_parser("map")
    mapping.add_argument("--cache-key", required=True)
    verify = sub.add_parser("verify-source")
    verify.add_argument("--source-patch", type=Path)
    storage = sub.add_parser("storage")
    storage.add_argument("--minimum-gb", type=float, default=2)
    args = parser.parse_args()
    if args.action in ("stage", "map"):
        # Mutations require the same live root guard acquired by install/launch.
        validate_owner(Path(args.root).resolve())
    if args.action == "guard":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        return guard(args.root, args.role, command)
    if args.action == "assert-guard":
        if validate_owner(Path(args.root).resolve()).get("role") not in args.owner_role:
            raise RuntimeError("Unexpected root guard role for this internal operation")
        return 0
    if args.action == "recover-lock":
        recover_lock(args.root, args.expect_token)
    elif args.action == "stage":
        print(
            stage_source(
                args.root, args.repository, args.source_patch, args.replace_source
            )
        )
    elif args.action == "map":
        print(map_state(args.root, args.cache_key))
    elif args.action == "verify-source":
        app = Path(args.root) / "source" / SOURCE_SHA
        verify_source(app)
        if args.source_patch:
            if read_json(app / ".fusion-source.json").get(
                "source_patch_sha256"
            ) != digest(args.source_patch):
                raise RuntimeError(
                    "Existing source overlay differs from the required runtime/image patch; explicit restaging required"
                )
    elif args.action == "storage":
        print(json.dumps(check_storage(args.root, args.minimum_gb), indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Fusion runtime refused: {error}", file=sys.stderr)
        sys.exit(73)
