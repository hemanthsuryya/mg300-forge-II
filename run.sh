#!/usr/bin/env bash
# Start Tone Sear on http://127.0.0.1:8000
set -e
cd "$(dirname "$0")"

# The app needs Python 3.10+. A bare `python3` is often older (macOS system
# Python, or an old pyenv default), so pick the first interpreter that
# actually qualifies rather than failing halfway through the install.
pick_python() {
  for c in "$PYTHON" python3.13 python3.12 python3.11 python3.10 python3; do
    [ -n "$c" ] || continue
    p="$(command -v "$c" 2>/dev/null)" || continue
    if "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      echo "$p"; return 0
    fi
  done
  return 1
}

if [ ! -d .venv ]; then
  PY="$(pick_python)" || {
    echo "Python 3.10 or newer is required and none was found on PATH." >&2
    echo "Install one (brew install python@3.12) or set PYTHON=/path/to/python3." >&2
    exit 1
  }
  echo "First run: creating a virtual environment with $PY and installing dependencies."
  "$PY" -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

exec ./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8000}"
