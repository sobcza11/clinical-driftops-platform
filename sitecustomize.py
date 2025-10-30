# sitecustomize.py  (repo root)
# Ensures stdout/stderr are UTF-8 so any Unicode (emoji, long dashes, etc.)
# prints cleanly on Windows terminals and in subprocesses spawned by tests.

import os, sys

# Respect PYTHONIOENCODING if user set it; otherwise force UTF-8.
if not os.environ.get("PYTHONIOENCODING"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
