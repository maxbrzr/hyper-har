"""Camera-ready paper experiment stages."""

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / "artifacts" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_ROOT / "artifacts" / ".cache"))
