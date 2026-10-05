#!/usr/bin/env bash
set -Eeuo pipefail
# Xfce and Fusion are independent supervised children; browser lifetimes do not
# start/stop either process. A separate display avoids the existing ComfyUI app.
for attempt in {1..100}; do
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        exec dbus-run-session -- startxfce4
    fi
    sleep 0.2
done
printf 'Visible desktop display did not become ready.\n' >&2
exit 1
