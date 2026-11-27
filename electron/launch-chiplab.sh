#!/usr/bin/env bash
# ChipLab Desktop launcher — Linux (GNOME/Wayland, KDE, X11)
#
# Why every flag here exists:
#   --no-sandbox / --disable-gpu-sandbox  Chromium's SUID sandbox helper is not
#                                         setuid-root in a dev checkout and aborts
#                                         with SIGTRAP before any window appears.
#   --ozone-platform=x11                  On Wayland + NVIDIA the Wayland Ozone
#                                         backend kills the GPU process (exit 15)
#                                         and the window never maps. XWayland works.
#   --enable-unsafe-swiftshader           Software WebGL2 fallback when the NVIDIA
#                                         driver refuses a context in Electron.
#   --disable-dev-shm-usage               /dev/shm is small in containers/Flatpak.
set -uo pipefail

APP_DIR="/home/everett/Chip Lab/electron"
STATE_DIR="${HOME}/.local/state/chiplab"
LOG="${STATE_DIR}/launch.log"
mkdir -p "$STATE_DIR" 2>/dev/null || true

exec >>"$LOG" 2>&1
echo "===== $(date -Is) launch requested ====="

notify_failure() {
  local msg="$1"
  echo "FAILURE: $msg"
  if command -v notify-send >/dev/null 2>&1; then
    notify-send -u critical -a ChipLab "ChipLab failed to start" "$msg" || true
  fi
}

if [[ ! -d "$APP_DIR" ]]; then
  notify_failure "App directory not found: $APP_DIR"
  exit 1
fi
cd "$APP_DIR" || { notify_failure "Cannot cd into $APP_DIR"; exit 1; }

ELECTRON_BIN="$APP_DIR/node_modules/electron/dist/electron"
if [[ ! -x "$ELECTRON_BIN" ]]; then
  notify_failure "Electron binary missing — run: cd \"$APP_DIR\" && npm install"
  exit 1
fi

# A SIGKILLed run can leave a stale Chromium profile lock behind, which makes
# every later launch exit immediately with no window.
PROFILE_DIR="${HOME}/.config/chiplab-desktop"
if [[ -L "$PROFILE_DIR/SingletonLock" ]]; then
  OWNER_PID="$(readlink "$PROFILE_DIR/SingletonLock" | sed 's/.*-//')"
  if ! kill -0 "$OWNER_PID" 2>/dev/null; then
    echo "Removing stale SingletonLock (pid $OWNER_PID is gone)"
    rm -f "$PROFILE_DIR/SingletonLock" "$PROFILE_DIR/SingletonSocket" "$PROFILE_DIR/SingletonCookie"
  fi
fi

# Fall back to the local X display if launched from a bare environment.
if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X0 ]]; then
  export DISPLAY=:0
fi

export ELECTRON_DISABLE_SECURITY_WARNINGS=1

"$ELECTRON_BIN" \
  --no-sandbox \
  --disable-gpu-sandbox \
  --enable-unsafe-swiftshader \
  --disable-dev-shm-usage \
  --ozone-platform=x11 \
  --disable-features=WaylandWindowDecorations \
  --class=ChipLab \
  "$APP_DIR" "$@"
CODE=$?

if [[ $CODE -ne 0 ]]; then
  notify_failure "Electron exited with code $CODE. Last log lines:
$(tail -n 12 "$LOG" 2>/dev/null)"
fi
exit $CODE
