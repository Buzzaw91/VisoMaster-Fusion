#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "$FUSION_START_APP" == 0 ]]; then
    printf 'Fusion auto-start disabled; start with launch-fusion.sh inside this desktop session.\n'
    exit 0
fi
for attempt in {1..100}; do
    audio_ready=1
    if [[ "${FUSION_TRANSPORT:-novnc}" == xpra ]] && ! pactl info >/dev/null 2>&1; then
        audio_ready=0
    fi
    if [[ "$audio_ready" == 1 ]] && xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        exec "$FUSION_DEPLOY_DIR/launch-fusion.sh"
    fi
    sleep 0.2
done
printf 'Fusion refused: visible display or selected session audio did not become ready.\n' >&2
exit 1
