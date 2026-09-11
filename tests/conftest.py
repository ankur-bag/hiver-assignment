"""Test import configuration; remote services are replaced by explicit fakes."""

import sys
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1] / "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
