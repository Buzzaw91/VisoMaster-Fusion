#!/usr/bin/env python3
"""Prepare a dedicated key-only sshd without touching an existing SSH service."""

import ipaddress
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
import tempfile

from runtime import atomic_json, validate_owner

KEY_TYPES = {
    "ssh-ed25519",
    "ssh-rsa",
    "ecdsa-sha2-nistp256",
    "ecdsa-sha2-nistp384",
    "ecdsa-sha2-nistp521",
    "sk-ssh-ed25519@openssh.com",
    "sk-ecdsa-sha2-nistp256@openssh.com",
}


def validated_keys(text, folder):
    """Validate plain public-key lines without echoing supplied material."""
    keys = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if (
            len(fields) < 2
            or fields[0] not in KEY_TYPES
            or not re.fullmatch(r"[A-Za-z0-9+/=]+", fields[1])
        ):
            raise RuntimeError(
                "Invalid SSH public key; private keys and authorized-key options are not accepted"
            )
        key = " ".join(fields[:2])
        with tempfile.NamedTemporaryFile(mode="w", dir=folder) as probe:
            probe.write(key + "\n")
            probe.flush()
            result = subprocess.run(
                ["ssh-keygen", "-l", "-f", probe.name], capture_output=True, text=True
            )
            if result.returncode:
                raise RuntimeError("SSH public key failed OpenSSH validation")
        keys.append(key)
    if not keys:
        raise RuntimeError("Managed SSH requires at least one valid runtime public key")
    return list(dict.fromkeys(keys))


def prepare(root, run_dir):
    if os.geteuid() != 0:
        raise RuntimeError(
            "Managed image SSH requires root; use the existing endpoint in a nonroot deployment"
        )
    port = int(os.environ.get("FUSION_SSH_PORT", "22"))
    if not 1 <= port <= 65535:
        raise RuntimeError("Invalid FUSION_SSH_PORT")
    bind = str(ipaddress.ip_address(os.environ.get("FUSION_SSH_BIND", "0.0.0.0")))
    supplied_file = os.environ.get("FUSION_SSH_PUBLIC_KEY_FILE")
    if supplied_file:
        text, source = Path(supplied_file).read_text(), "FUSION_SSH_PUBLIC_KEY_FILE"
    elif os.environ.get("PUBLIC_KEY"):
        text, source = os.environ["PUBLIC_KEY"], "PUBLIC_KEY"
    elif os.environ.get("SSH_PUBLIC_KEY"):
        text, source = os.environ["SSH_PUBLIC_KEY"], "SSH_PUBLIC_KEY"
    else:
        raise RuntimeError(
            "Managed SSH refused: runtime public key file or PUBLIC_KEY is required"
        )
    ssh_dir = run_dir / "ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    ssh_dir.chmod(0o700)
    keys = validated_keys(text, ssh_dir)
    authorized = ssh_dir / "authorized_keys"
    authorized.write_text("\n".join(keys) + "\n")
    authorized.chmod(0o600)
    host_dir = root / "profile" / "ssh-hostkeys"
    if host_dir.is_symlink():
        raise RuntimeError("SSH host-key directory must not be a symlink")
    host_dir.mkdir(parents=True, exist_ok=True)
    host_dir.chmod(0o700)
    host_key = host_dir / "ssh_host_ed25519_key"
    if host_key.is_symlink():
        raise RuntimeError("SSH host key must not be a symlink")
    if not host_key.exists():
        result = subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(host_key)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError("Runtime SSH host-key generation failed")
    host_key.chmod(0o600)
    user = pwd.getpwuid(os.getuid()).pw_name
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", user):
        raise RuntimeError("Unsupported SSH account name")
    for path in (authorized, host_key, ssh_dir):
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(path)):
            raise RuntimeError(
                "SSH runtime paths contain unsupported configuration characters"
            )
    config = ssh_dir / "sshd_config"
    config.write_text(f"""Port {port}
ListenAddress {bind}
HostKey {host_key}
PidFile {ssh_dir}/sshd.pid
AuthorizedKeysFile {authorized}
StrictModes yes
AllowUsers {user}
PermitRootLogin prohibit-password
PubkeyAuthentication yes
AuthenticationMethods publickey
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitEmptyPasswords no
UsePAM yes
AllowTcpForwarding local
GatewayPorts no
X11Forwarding no
PermitTunnel no
PermitUserEnvironment no
PrintMotd no
LogLevel INFO
Subsystem sftp internal-sftp
""")
    config.chmod(0o600)
    # Ubuntu's compiled privilege-separation directory; no daemon starts here.
    Path("/run/sshd").mkdir(mode=0o755, exist_ok=True)
    result = subprocess.run(
        ["/usr/sbin/sshd", "-t", "-f", str(config)], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(
            "Dedicated SSH configuration validation failed; no SSH endpoint started"
        )
    atomic_json(
        root / "receipts" / "ssh-endpoint.json",
        {
            "managed": True,
            "listen_port": port,
            "listen_address": bind,
            "authentication": "publickey-only",
            "key_source": source,
            "authorized_key_count": len(keys),
            "host_key_location": str(host_key),
            "existing_host_sshd_modified": False,
            "endpoint_connection_validated": False,
        },
    )
    print(f"Dedicated key-only SSH configured on TCP {port}; key values omitted")


def main():
    if os.environ.get("FUSION_MANAGED_SSH", "0") != "1":
        raise RuntimeError("SSH preparation requires explicit FUSION_MANAGED_SSH=1")
    root = Path(os.environ.get("FUSION_ROOT", "/workspace/fusion")).resolve()
    validate_owner(root)
    prepare(
        root, Path(os.environ.get("FUSION_RUN_DIR", "/run/fusion-desktop")).resolve()
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Fusion SSH refused: {error}", file=sys.stderr)
        sys.exit(78)
