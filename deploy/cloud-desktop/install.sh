#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FUSION_ROOT="${FUSION_ROOT:-/workspace/fusion}"
PIN=f2d6f5ebe190b631ee5c8972332083cf7f8a3e72
MODE="${1:-check}"
if [[ "$MODE" == --help || "$MODE" == -h ]]; then
    cat <<'HELP'
Usage: install.sh check | install [--source-repo PATH] [--skip-apt] [--skip-models]
                   [--models all|NAME,NAME] [--reuse-model-root PATH] [--resolve-lock]
                   [--borrow-python /absolute/existing/venv/bin/python]
                   [--source-patch PATH] [--replace-source]
                   [--resolve-only | --resolved-lock /path/reviewed-hashed.lock]
                   [--defer-import-check]
check prints prerequisites and never installs or starts services.
install requires Linux x86_64/Ubuntu 24.04 and never starts desktop/Fusion.
Default persistent root is /workspace/fusion; it must be a mounted durable volume.
Dependency pins are provisional until a Linux --resolve-lock receipt is captured.
--defer-import-check explicitly records unchecked setup imports for cross-builds.
It never disables the launcher's required runtime preflight.
HELP
    exit 0
fi
shift || true
if [[ "$MODE" == check ]]; then
    uname -a
    for command in python3 git ffmpeg ffprobe nvidia-smi Xtigervnc supervisord nginx; do
        if command -v "$command" >/dev/null; then command -v "$command"; else printf 'MISSING: %s\n' "$command"; fi
    done
    if command -v nvidia-smi >/dev/null; then nvidia-smi; fi
    printf 'Storage target: %s\nNo installation or services started.\n' "$FUSION_ROOT"
    exit 0
fi
[[ "$MODE" == install ]] || { printf 'Unknown mode: %s\n' "$MODE" >&2; exit 64; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { printf 'This installer requires Linux x86_64.\n' >&2; exit 65; }
SOURCE_REPO="${FUSION_SOURCE_REPO:-}"
SKIP_APT=0; SKIP_MODELS=0; RESOLVE_LOCK=0; RESOLVE_ONLY=0; RESOLVED_LOCK=""; MODEL_NAMES=all; REUSE_ROOT=""; BORROW_PYTHON=""; SOURCE_PATCH="${FUSION_SOURCE_PATCH:-}"; REPLACE_SOURCE=0
LOCKED=0; DEFER_IMPORT_CHECK=0
while (($#)); do
    case "$1" in
        --source-repo) SOURCE_REPO="$2"; shift 2 ;;
        --skip-apt) SKIP_APT=1; shift ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --models) MODEL_NAMES="$2"; shift 2 ;;
        --reuse-model-root) REUSE_ROOT="$2"; shift 2 ;;
        --resolve-lock) RESOLVE_LOCK=1; shift ;;
        --resolve-only) RESOLVE_LOCK=1; RESOLVE_ONLY=1; shift ;;
        --resolved-lock) RESOLVED_LOCK="$2"; shift 2 ;;
        --borrow-python) BORROW_PYTHON="$2"; shift 2 ;;
        --source-patch) SOURCE_PATCH="$2"; shift 2 ;;
        --replace-source) REPLACE_SOURCE=1; shift ;;
        --defer-import-check) DEFER_IMPORT_CHECK=1; shift ;;
        --locked) LOCKED=1; shift ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done
if [[ -n "$RESOLVED_LOCK" ]] && (( RESOLVE_LOCK )); then
    printf 'Choose supplied --resolved-lock or fresh --resolve-lock/--resolve-only.\n' >&2; exit 64
fi
if [[ -n "$BORROW_PYTHON" ]] && { (( RESOLVE_LOCK )) || [[ -n "$RESOLVED_LOCK" ]]; }; then
    printf 'Borrowed runtime is read-only; dependency resolution/installation requires the separate runtime path.\n' >&2; exit 64
fi
[[ -z "$RESOLVED_LOCK" || -s "$RESOLVED_LOCK" ]] || { printf 'Reviewed resolved lock is missing or empty.\n' >&2; exit 64; }
if (( ! LOCKED )); then
    args=(install --locked --models "$MODEL_NAMES")
    [[ -z "$SOURCE_REPO" ]] || args+=(--source-repo "$SOURCE_REPO")
    (( ! SKIP_APT )) || args+=(--skip-apt)
    (( ! SKIP_MODELS )) || args+=(--skip-models)
    (( ! RESOLVE_LOCK )) || args+=(--resolve-lock)
    (( ! RESOLVE_ONLY )) || args+=(--resolve-only)
    [[ -z "$RESOLVED_LOCK" ]] || args+=(--resolved-lock "$RESOLVED_LOCK")
    [[ -z "$REUSE_ROOT" ]] || args+=(--reuse-model-root "$REUSE_ROOT")
    [[ -z "$BORROW_PYTHON" ]] || args+=(--borrow-python "$BORROW_PYTHON")
    [[ -z "$SOURCE_PATCH" ]] || args+=(--source-patch "$SOURCE_PATCH")
    (( ! REPLACE_SOURCE )) || args+=(--replace-source)
    (( ! DEFER_IMPORT_CHECK )) || args+=(--defer-import-check)
    exec python3 "$HERE/runtime.py" --root "$FUSION_ROOT" guard --role setup -- "$HERE/install.sh" "${args[@]}"
fi
python3 "$HERE/runtime.py" --root "$FUSION_ROOT" assert-guard --owner-role setup
if (( ! SKIP_APT )); then
    [[ $EUID -eq 0 ]] || { printf 'System packages require root; provision packages first or run --skip-apt.\n' >&2; exit 77; }
    . /etc/os-release
    [[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || { printf 'Apt setup is scoped to Ubuntu 24.04.\n' >&2; exit 65; }
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
      ca-certificates curl git python3.12 python3.12-venv python3-pip \
      ffmpeg libgl1 libegl1 libopengl0 libglu1-mesa libglib2.0-0 libdbus-1-3 \
      libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
      libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 \
      libxcb-shape0 libxcb-randr0 libxcb-sync1 libxcb-xfixes0 libxcb1 \
      libx11-xcb1 libxrender1 libxi6 libxrandr2 libxtst6 libsm6 libice6 \
      libasound2t64 libpulse0 libnss3 libfontconfig1 libfreetype6 fonts-dejavu-core \
      gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
      gstreamer1.0-libav gstreamer1.0-pulseaudio gstreamer1.0-plugins-ugly \
      libgstreamer-plugins-base1.0-0 python3-gst-1.0 gir1.2-gstreamer-1.0 \
      gir1.2-gst-plugins-base-1.0 xpra libjs-jquery pulseaudio pulseaudio-utils \
      xfce4 xfce4-terminal dbus-x11 xauth x11-utils tigervnc-standalone-server \
      tigervnc-tools novnc websockify supervisor nginx apache2-utils procps openssh-server
fi
command -v python3.12 >/dev/null || { printf 'Python 3.12 is required.\n' >&2; exit 69; }
export FUSION_RUNTIME_ROOT="${FUSION_RUNTIME_ROOT:-$FUSION_ROOT/runtime}"
mkdir -p "$FUSION_RUNTIME_ROOT" "$FUSION_ROOT/logs" "$FUSION_ROOT/receipts"
if [[ -z "$SOURCE_REPO" ]]; then
    SOURCE_REPO="$FUSION_RUNTIME_ROOT/source.git"
    if [[ ! -d "$SOURCE_REPO" ]]; then git init --bare "$SOURCE_REPO"; fi
    if ! git -C "$SOURCE_REPO" cat-file -e "$PIN^{commit}" 2>/dev/null; then
        git -C "$SOURCE_REPO" fetch --no-tags https://github.com/VisoMasterFusion/VisoMaster-Fusion.git "$PIN"
    fi
fi
python3.12 "$HERE/runtime.py" --root "$FUSION_ROOT" storage --minimum-gb "${FUSION_MIN_FREE_GB:-2}"
stage_args=(stage --repository "$SOURCE_REPO")
[[ -z "$SOURCE_PATCH" ]] || stage_args+=(--source-patch "$SOURCE_PATCH")
(( ! REPLACE_SOURCE )) || stage_args+=(--replace-source)
python3.12 "$HERE/runtime.py" --root "$FUSION_ROOT" "${stage_args[@]}"
APP="$FUSION_ROOT/source/$PIN"
LOCK="$HERE/requirements-linux.lock"
STATUS=provisional-direct-pins
if [[ -n "$BORROW_PYTHON" ]]; then
    [[ "$BORROW_PYTHON" == /* && -x "$BORROW_PYTHON" ]] || { printf 'Borrowed Python must be an existing absolute executable.\n' >&2; exit 65; }
    "$BORROW_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Borrowed runtime must use Python 3.12"'
    APP_PYTHON="$BORROW_PYTHON"
    STATUS=borrowed-unmodified-runtime-compatibility-trial
else
  # Bootstrap is separate from the application venv; no host pip packages are replaced.
  if [[ ! -x "$FUSION_RUNTIME_ROOT/bootstrap/bin/python" ]]; then
      python3.12 -m venv "$FUSION_RUNTIME_ROOT/bootstrap"
  fi
  "$FUSION_RUNTIME_ROOT/bootstrap/bin/python" -m pip install --require-hashes -r "$HERE/bootstrap-linux.lock"
  UV="$FUSION_RUNTIME_ROOT/bootstrap/bin/uv"
  export UV_CACHE_DIR="$FUSION_RUNTIME_ROOT/uv-cache"
  export UV_LINK_MODE="${FUSION_INSTALL_UV_LINK_MODE:-copy}"
  [[ "$UV_LINK_MODE" == copy || "$UV_LINK_MODE" == hardlink ]] || { printf 'FUSION_INSTALL_UV_LINK_MODE must be copy or hardlink.\n' >&2; exit 64; }
  VENV="$FUSION_RUNTIME_ROOT/venv"
  if [[ ! -x "$VENV/bin/python" ]]; then "$UV" venv --python python3.12 "$VENV"; fi
  APP_PYTHON="$VENV/bin/python"
  if (( RESOLVE_LOCK )); then
    LOCK="$FUSION_ROOT/receipts/requirements-linux.resolved.lock"
    "$UV" pip compile --python "$VENV/bin/python" --generate-hashes \
      --index-strategy unsafe-best-match --index-url https://pypi.org/simple \
      --extra-index-url https://pypi.nvidia.com \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \
      "$HERE/requirements-linux.lock" -o "$LOCK"
    STATUS=resolved-hashed-linux
    if (( RESOLVE_ONLY )); then
        FUSION_DEPENDENCY_STATUS=resolved-hashed-not-installed FUSION_INSTALL_LOCK="$LOCK" \
            "$APP_PYTHON" "$HERE/receipt.py" --root "$FUSION_ROOT" --source-repo "$SOURCE_REPO" --resolution-only
        printf 'Resolution complete; application packages, native mappings, models and services were not installed or started.\n'
        exit 0
    fi
    "$UV" pip sync --python "$VENV/bin/python" --require-hashes --index-strategy unsafe-best-match \
      --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ "$LOCK"
  elif [[ -n "$RESOLVED_LOCK" ]]; then
    LOCK="$RESOLVED_LOCK"
    STATUS=reviewed-hashed-linux-lock
    "$UV" pip sync --python "$VENV/bin/python" --require-hashes --index-strategy unsafe-best-match \
      --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ "$LOCK"
  else
    "$UV" pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
      --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ -r "$LOCK"
  fi
  "$UV" pip check --python "$VENV/bin/python"
  "$UV" pip freeze --python "$VENV/bin/python" > "$FUSION_ROOT/receipts/pip-freeze.txt"
fi
printf '%s\n' "$APP_PYTHON" > "$FUSION_ROOT/receipts/python-path.txt"
python3.12 "$HERE/runtime.py" --root "$FUSION_ROOT" map --cache-key unprobed
export FUSION_PYTHON="$APP_PYTHON"
source "$HERE/gpu-env.sh"
if (( ! SKIP_MODELS )); then
    args=(--root "$FUSION_ROOT" --mode download --models "$MODEL_NAMES")
    [[ -z "$REUSE_ROOT" ]] || args+=(--reuse-root "$REUSE_ROOT")
    "$APP_PYTHON" "$HERE/models.py" "${args[@]}"
fi
export FUSION_IMPORT_CHECK_STATUS=passed-setup-import-check
if (( DEFER_IMPORT_CHECK )); then
    export FUSION_IMPORT_CHECK_STATUS=deferred-no-import-check
    python3.12 - "$FUSION_ROOT" "$HERE" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from runtime import atomic_json
atomic_json(Path(sys.argv[1]) / 'logs/setup-imports.json', {
    'status': 'deferred-no-import-check', 'imports_checked': False,
    'reason': 'explicit --defer-import-check for setup; launch runtime preflight remains required',
    'native_functionality': 'unvalidated', 'gpu_execution_verified': False,
})
PY
    printf 'Setup import checks explicitly deferred; native/GPU functionality remains unvalidated.\n' >&2
else
    "$APP_PYTHON" "$HERE/preflight.py" --root "$FUSION_ROOT" --mode imports \
        --output "$FUSION_ROOT/logs/setup-imports.json"
fi
FUSION_DEPENDENCY_STATUS="$STATUS" FUSION_INSTALL_LOCK="$LOCK" \
    "$APP_PYTHON" "$HERE/receipt.py" --root "$FUSION_ROOT" --source-repo "$SOURCE_REPO"
printf 'Setup complete; desktop and Fusion were not started.\nDependency status: %s\nImport check status: %s\n' "$STATUS" "$FUSION_IMPORT_CHECK_STATUS"
