from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PHASE1_SCRIPT = ROOT / "style-scripts" / "phase1_contrastive_set_encoder.py"


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    phase1 = _load_module_from_path("phase1_contrastive_wrapper", PHASE1_SCRIPT)
    print("Redirecting to Phase 1 script: style-scripts/phase1_contrastive_set_encoder.py")
    phase1.main()


if __name__ == "__main__":
    main()
