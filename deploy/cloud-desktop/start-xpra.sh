#!/usr/bin/env bash
set -Eeuo pipefail
[[ "${FUSION_ACCESS_MODE:-tunnel}" == tunnel ]] || { printf 'Public Xpra refused.\n' >&2; exit 78; }
# Ignore inherited/system startup commands: this server only shadows our display.
export XPRA_SYSTEM_CONF_DIRS="$FUSION_RUN_DIR/xpra-config"
export XPRA_USER_CONF_DIRS="$FUSION_RUN_DIR/xpra-config"
export XPRA_DEFAULT_CONF_DIRS="$FUSION_RUN_DIR/xpra-config"
# Ubuntu's Xpra 3.1.5 icon scaler uses removed Pillow.Image.ANTIALIAS.
# Disable transport window icons; the actual desktop and native UI stay visible.
export XPRA_PNG_ICONS=0 XPRA_ARGB_ICONS=0 XPRA_DEFAULT_ICONS=0
for attempt in {1..100}; do
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 && pactl info >/dev/null 2>&1; then
        # The explicit source prevents Xpra's synthetic test-source fallback.
        # All sockets are localhost/private; the existing TigerVNC owns DISPLAY.
        exec /usr/bin/xpra shadow "$DISPLAY" --daemon=no --systemd-run=no \
          --bind-tcp="127.0.0.1:$FUSION_XPRA_PORT" --html=on --ssh-upgrade=no \
          --socket-dir="$FUSION_RUN_DIR/xpra" --socket-dirs="$FUSION_RUN_DIR/xpra" \
          --speaker=on --microphone=disabled --pulseaudio=no \
          --sound-source=pulse:device=fusion_preview.monitor \
          --dbus-launch= --dbus-proxy=no --dbus-control=no --mdns=no \
          --printing=no --file-transfer=no --open-files=no --open-url=no \
          --start-new-commands=no --webcam=no --exit-with-client=no \
          --tray=no --notifications=no --resize-display=no
    fi
    sleep 0.2
done
printf 'Xpra display or private Pulse server did not become ready.\n' >&2
exit 1
