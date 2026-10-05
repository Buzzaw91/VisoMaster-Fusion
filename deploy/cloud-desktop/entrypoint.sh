#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FUSION_ROOT="${FUSION_ROOT:-/workspace/fusion}"
export FUSION_ACCESS_MODE="${FUSION_ACCESS_MODE:-tunnel}"
export FUSION_TRANSPORT="${FUSION_TRANSPORT:-xpra}"
export DISPLAY="${DISPLAY:-:100}"
export FUSION_HTTP_PORT="${FUSION_HTTP_PORT:-6080}"
export FUSION_WEBSOCKIFY_PORT="${FUSION_WEBSOCKIFY_PORT:-6081}"
export FUSION_VNC_PORT="${FUSION_VNC_PORT:-5901}"
export FUSION_XPRA_PORT="${FUSION_XPRA_PORT:-6082}"
export FUSION_DESKTOP_GEOMETRY="${FUSION_DESKTOP_GEOMETRY:-1920x1080}"
export FUSION_PREFLIGHT_MODE="${FUSION_PREFLIGHT_MODE:-full}"
export FUSION_START_APP="${FUSION_START_APP:-1}"
export FUSION_MANAGED_SSH="${FUSION_MANAGED_SSH:-0}"
export FUSION_SSH_PORT="${FUSION_SSH_PORT:-22}"
export FUSION_MODEL_SELECTION="${FUSION_MODEL_SELECTION-RetinaFace,Inswapper128ArcFace,Inswapper128,FaceLandmark5,FaceLandmark68,FaceLandmark106,GFPGANv1.4,CodeFormer}"
if [[ "${1:-}" != --locked ]]; then
    if [[ "${1:-}" == --help ]]; then
        printf 'entrypoint.sh: localhost Xpra HTML5/audio on 6082 by default; noVNC fallback on 6080.\nFUSION_TRANSPORT=novnc disables Xpra/audio. Public mode supports only authenticated noVNC.\nFUSION_PREFLIGHT_MODE=imports explicitly selects a GUI compatibility pilot.\n'
        exit 0
    fi
    exec python3 "$HERE/runtime.py" --root "$FUSION_ROOT" guard --role desktop -- "$HERE/entrypoint.sh" --locked "$@"
fi
shift
python3 "$HERE/runtime.py" --root "$FUSION_ROOT" assert-guard --owner-role desktop
for value in "$FUSION_HTTP_PORT" "$FUSION_WEBSOCKIFY_PORT" "$FUSION_VNC_PORT" "$FUSION_XPRA_PORT"; do
    [[ "$value" =~ ^[0-9]+$ && "$value" -gt 1024 && "$value" -lt 65536 ]] || { printf 'Invalid desktop port\n' >&2; exit 64; }
done
[[ "$DISPLAY" =~ ^:[0-9]+$ && "$FUSION_DESKTOP_GEOMETRY" =~ ^[0-9]+x[0-9]+$ ]] || { printf 'Invalid display or geometry\n' >&2; exit 64; }
[[ "$FUSION_START_APP" == 0 || "$FUSION_START_APP" == 1 ]] || exit 64
case "$FUSION_TRANSPORT" in
  xpra)
    [[ "$FUSION_ACCESS_MODE" == tunnel ]] || { printf 'Public Xpra refused; use SSH tunnel or authenticated noVNC.\n' >&2; exit 78; }
    export FUSION_AUDIO_AUTOSTART=true
    ;;
  novnc) export FUSION_AUDIO_AUTOSTART=false ;;
  *) printf 'FUSION_TRANSPORT must be xpra or novnc.\n' >&2; exit 64 ;;
esac
if [[ "$FUSION_ACCESS_MODE" == public ]]; then
    [[ -n "${FUSION_HTPASSWD_FILE:-}" && -s "$FUSION_HTPASSWD_FILE" ]] || { printf 'Public desktop refused: runtime FUSION_HTPASSWD_FILE required.\n' >&2; exit 78; }
fi
if [[ -z "${FUSION_PYTHON:-}" && -f "$FUSION_ROOT/receipts/python-path.txt" ]]; then
    FUSION_PYTHON="$(cat "$FUSION_ROOT/receipts/python-path.txt")"
fi
export FUSION_PYTHON="${FUSION_PYTHON:-$FUSION_ROOT/runtime/venv/bin/python}"
# Container images ship an immutable bare source repository and installed runtime.
if [[ ! -d "$FUSION_ROOT/source/f2d6f5ebe190b631ee5c8972332083cf7f8a3e72" ]]; then
    [[ -n "${FUSION_SOURCE_REPO:-}" ]] || { printf 'Source not staged. Run install.sh first.\n' >&2; exit 69; }
    stage_args=(stage --repository "$FUSION_SOURCE_REPO")
    [[ -z "${FUSION_SOURCE_PATCH:-}" ]] || stage_args+=(--source-patch "$FUSION_SOURCE_PATCH")
    python3 "$HERE/runtime.py" --root "$FUSION_ROOT" "${stage_args[@]}"
    python3 "$HERE/runtime.py" --root "$FUSION_ROOT" map --cache-key unprobed
elif [[ -n "${FUSION_SOURCE_PATCH:-}" ]]; then
    python3 "$HERE/runtime.py" --root "$FUSION_ROOT" verify-source --source-patch "$FUSION_SOURCE_PATCH"
fi
[[ -x "$FUSION_PYTHON" ]] || { printf 'Python runtime missing.\n' >&2; exit 69; }
export FUSION_RUN_DIR="${FUSION_RUN_DIR:-/run/fusion-desktop}"
export FUSION_SSH_AUTOSTART=false
case "$FUSION_MANAGED_SSH" in
  0) ;;
  1)
    python3 "$HERE/prepare-ssh.py"
    export FUSION_SSH_AUTOSTART=true
    ;;
  *) printf 'FUSION_MANAGED_SSH must be 0 or 1.\n' >&2; exit 64 ;;
esac
if [[ "${FUSION_DOWNLOAD_MODELS:-0}" == 1 ]]; then
    model_args=(--root "$FUSION_ROOT" --mode download --models "$FUSION_MODEL_SELECTION")
    [[ -z "${FUSION_REUSE_MODEL_ROOT:-}" ]] || model_args+=(--reuse-root "$FUSION_REUSE_MODEL_ROOT")
    "$FUSION_PYTHON" "$HERE/models.py" "${model_args[@]}"
fi
for binary in Xtigervnc websockify nginx supervisord dbus-run-session startxfce4 xdpyinfo; do
    command -v "$binary" >/dev/null || { printf 'Desktop dependency missing: %s\n' "$binary" >&2; exit 69; }
done
python3 "$HERE/runtime.py" --root "$FUSION_ROOT" storage
mkdir -p "$FUSION_RUN_DIR" "$FUSION_ROOT/logs" "$FUSION_ROOT/profile/config" "$FUSION_ROOT/profile/data"
chmod 700 "$FUSION_RUN_DIR"
export XAUTHORITY="$FUSION_RUN_DIR/Xauthority"
touch "$XAUTHORITY"
chmod 600 "$XAUTHORITY"
python3 - <<'PY'
import os, secrets, subprocess
subprocess.run(['xauth', '-f', os.environ['XAUTHORITY'], 'add', os.environ['DISPLAY'],
                'MIT-MAGIC-COOKIE-1', secrets.token_hex(16)], check=True, capture_output=True)
PY
export XDG_RUNTIME_DIR="$FUSION_RUN_DIR/xdg-runtime"
export XDG_CONFIG_HOME="$FUSION_ROOT/profile/config"
export XDG_DATA_HOME="$FUSION_ROOT/profile/data"
export XDG_CACHE_HOME="$FUSION_ROOT/profile/cache"
mkdir -p "$XDG_RUNTIME_DIR" "$XDG_CACHE_HOME"
chmod 700 "$XDG_RUNTIME_DIR"
if [[ "$FUSION_TRANSPORT" == xpra ]]; then
    python3 "$HERE/prepare-audio.py"
    export PULSE_SERVER="unix:$FUSION_RUN_DIR/pulse/native"
    export PULSE_COOKIE="$FUSION_RUN_DIR/pulse/cookie"
    export PULSE_SINK=fusion_preview
    export PULSE_SOURCE=fusion_preview.monitor
    export PULSE_RUNTIME_PATH="$FUSION_RUN_DIR/pulse"
    export PULSE_STATE_PATH="$FUSION_RUN_DIR/pulse-state"
fi
export QT_QPA_PLATFORM=xcb
# Default noVNC is reachable only via an SSH tunnel; public mode fails closed.
case "$FUSION_ACCESS_MODE" in
  tunnel) export FUSION_HTTP_BIND=127.0.0.1; AUTH_DIRECTIVES='auth_basic off;' ;;
  public)
    [[ -n "${FUSION_HTPASSWD_FILE:-}" && -s "$FUSION_HTPASSWD_FILE" ]] || { printf 'Public desktop refused: runtime FUSION_HTPASSWD_FILE required.\n' >&2; exit 78; }
    export FUSION_HTTP_BIND=0.0.0.0
    cp -- "$FUSION_HTPASSWD_FILE" "$FUSION_RUN_DIR/htpasswd"
    chmod 600 "$FUSION_RUN_DIR/htpasswd"
    AUTH_DIRECTIVES="auth_basic \"Fusion desktop\"; auth_basic_user_file $FUSION_RUN_DIR/htpasswd;"
    ;;
  *) printf 'FUSION_ACCESS_MODE must be tunnel or public.\n' >&2; exit 64 ;;
esac
export FUSION_DEPLOY_DIR="$HERE"
export FUSION_NGINX_CONFIG="$FUSION_RUN_DIR/nginx.conf"
export FUSION_SUPERVISOR_CONFIG="$FUSION_RUN_DIR/supervisord.conf"
FUSION_AUTH_DIRECTIVES="$AUTH_DIRECTIVES" python3 "$HERE/render-config.py"
nginx -t -c "$FUSION_NGINX_CONFIG" -p "$FUSION_RUN_DIR"
printf 'Fusion desktop starting: transport %s, display %s; noVNC fallback %s:%s (no audio).\n' "$FUSION_TRANSPORT" "$DISPLAY" "$FUSION_HTTP_BIND" "$FUSION_HTTP_PORT"
[[ "$FUSION_TRANSPORT" != xpra ]] || printf 'Xpra HTML5/audio: 127.0.0.1:%s through SSH tunnel; enable browser sound after connecting.\n' "$FUSION_XPRA_PORT"
exec supervisord -n -c "$FUSION_SUPERVISOR_CONFIG"
