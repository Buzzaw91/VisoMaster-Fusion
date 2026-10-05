#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FUSION_ROOT="${FUSION_ROOT:-/workspace/fusion}"
PIN=f2d6f5ebe190b631ee5c8972332083cf7f8a3e72
if [[ "${1:-}" != --locked ]]; then
    exec python3 "$HERE/runtime.py" --root "$FUSION_ROOT" guard --role fusion -- "$HERE/launch-fusion.sh" --locked "$@"
fi
shift
python3 "$HERE/runtime.py" --root "$FUSION_ROOT" assert-guard --owner-role desktop --owner-role fusion
if [[ -z "${FUSION_PYTHON:-}" && -f "$FUSION_ROOT/receipts/python-path.txt" ]]; then
    FUSION_PYTHON="$(cat "$FUSION_ROOT/receipts/python-path.txt")"
fi
export FUSION_PYTHON="${FUSION_PYTHON:-$FUSION_ROOT/runtime/venv/bin/python}"
[[ -x "$FUSION_PYTHON" ]] || { printf 'Isolated/borrowed Python runtime is missing: %s\n' "$FUSION_PYTHON" >&2; exit 69; }
export DISPLAY="${DISPLAY:-:100}"
# Explicitly override inherited ComfyUI/offscreen platform selection.
export QT_QPA_PLATFORM=xcb
export XDG_CONFIG_HOME="$FUSION_ROOT/profile/config"
export XDG_DATA_HOME="$FUSION_ROOT/profile/data"
export XDG_CACHE_HOME="$FUSION_ROOT/profile/cache"
export FUSION_PREFLIGHT_MODE="${FUSION_PREFLIGHT_MODE:-full}"
source "$HERE/gpu-env.sh"
python3 "$HERE/runtime.py" --root "$FUSION_ROOT" verify-source
"$FUSION_PYTHON" "$HERE/preflight.py" --root "$FUSION_ROOT" --mode "$FUSION_PREFLIGHT_MODE" \
    --output "$FUSION_ROOT/logs/launch-preflight.json"
if [[ "$FUSION_PREFLIGHT_MODE" == imports ]]; then
    printf 'Compatibility-only GUI pilot: GPU, encoder, native inference and browser/audio gates remain unvalidated.\n' >&2
    CACHE_KEY=unprobed
else
    CACHE_KEY="$("$FUSION_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cache_key"])' "$FUSION_ROOT/logs/launch-preflight.json")"
fi
python3 "$HERE/runtime.py" --root "$FUSION_ROOT" map --cache-key "$CACHE_KEY"
export TMPDIR="$FUSION_ROOT/cache/$CACHE_KEY/temp_files"
export TORCHINDUCTOR_CACHE_DIR="$FUSION_ROOT/cache/$CACHE_KEY/torch_compile_cache"
export CUDA_CACHE_PATH="$FUSION_ROOT/cache/$CACHE_KEY/cuda"
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
cd -- "$FUSION_ROOT/source/$PIN"
exec "$FUSION_PYTHON" main.py --gpu-id "${FUSION_GPU_ID:-0}" "$@"
