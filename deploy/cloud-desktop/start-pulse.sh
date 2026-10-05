#!/usr/bin/env bash
set -Eeuo pipefail
# Private session only: never autospawn, kill, or configure a host Pulse daemon.
exec /usr/bin/pulseaudio --daemonize=no --system=no --exit-idle-time=-1 \
  --use-pid-file=no --disable-shm=yes --log-target=stderr \
  -n --file="$FUSION_RUN_DIR/pulse/default.pa"
