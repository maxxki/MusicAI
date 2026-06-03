"""
conftest.py — Pytest Configuration für MAXXKI Tests

Stellt sicher, dass alle maxxki-Module aus dem gleichen Verzeichnis importiert werden,
unabhängig davon, wo pytest gestartet wird.
"""

import sys
from pathlib import Path

# Finde das Verzeichnis, in dem diese conftest.py liegt
TEST_DIR = Path(__file__).parent.resolve()

# Füge Test-Verzeichnis zu sys.path hinzu (vor allem anderen)
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

# Wenn Module in einem Unterverzeichnis liegen, auch das hinzufügen
# (z.B. für src-layout)
for subdir in ["src", "maxxki", "."]:
    candidate = TEST_DIR / subdir
    if candidate.exists() and candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
