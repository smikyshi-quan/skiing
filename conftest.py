"""Root conftest.py — ensure canonical packages resolve in the cleaned layout."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PLATFORM_ROOT = PROJECT_ROOT / "platform"
SRC_ROOT = PLATFORM_ROOT / "src"

for path in (str(SRC_ROOT), str(PLATFORM_ROOT), str(PROJECT_ROOT)):
    try:
        sys.path.remove(path)
    except ValueError:
        pass
    sys.path.insert(0, path)
