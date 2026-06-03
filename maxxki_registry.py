"""
MAXXKI Registry v1.4 — Singleton File-Lock (Deadlock-Fix)
Append-only Hash-Chain Registry für Audio-Manifeste.

Änderungen v1.4:
- File-Lock wird einmal pro Registry-Instanz gehalten (Singleton)
- _read_all() ohne Lock für interne Nutzung (Caller hält bereits)
- append() nutzt _read_all_unsafe() statt last_entry()

Manipulation eines Eintrags bricht die Kette → verify_chain() schlägt an.
"""

import hashlib
import json
import copy
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ═══ Lockfile Import ════════════════════════════════════════════════════════
try:
    from maxxki_registry_lock import get_lock
    _HAS_LOCK = True
except ImportError:
    _HAS_LOCK = False
    import warnings
    warnings.warn("maxxki_registry_lock nicht verfügbar — parallele Writes unsicher")

REGISTRY_PATH = Path.home() / ".maxxki" / "registry.chain"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _canonical(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Registry ─────────────────────────────────────────────────────────────────

class MAXXKIRegistry:

    GENESIS_HASH = "0" * 64

    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        # ═══ NEU v1.4: File-Lock Singleton pro Instanz ═══════════════════════
        self._file_lock = None
        self._thread_lock = threading.Lock()
        # ═════════════════════════════════════════════════════════════════════

    def _ensure_lock(self):
        """Lazy-Initialisierung des File-Locks (einmal pro Instanz)."""
        if self._file_lock is None and _HAS_LOCK:
            self._file_lock = get_lock(self.path.with_suffix(".lock"), timeout=30.0)
        return self._file_lock

    def _read_all(self) -> list[dict]:
        """RAW read — ohne Lock. Caller muss Lock halten."""
        lines = self.path.read_text().splitlines()
        return [json.loads(l) for l in lines if l.strip()]

    def count(self) -> int:
        with self._thread_lock:
            lock = self._ensure_lock()
            with lock:
                return len(self._read_all())

    def last_entry(self) -> Optional[dict]:
        with self._thread_lock:
            lock = self._ensure_lock()
            with lock:
                entries = self._read_all()
                return entries[-1] if entries else None

    def find_by_session(self, session_id: str) -> Optional[dict]:
        with self._thread_lock:
            lock = self._ensure_lock()
            with lock:
                for e in self._read_all():
                    if e.get("manifest", {}).get("content", {}).get("session_id") == session_id:
                        return e
                return None

    def find_by_hash(self, sha256: str) -> Optional[dict]:
        with self._thread_lock:
            lock = self._ensure_lock()
            with lock:
                for e in self._read_all():
                    content = e.get("manifest", {}).get("content", {})
                    if content.get("final_sha256") == sha256 or content.get("raw_sha256") == sha256:
                        return e
                return None

    # ── Schreiben ─────────────────────────────────────────────────────────────

    def append(self, manifest: dict) -> dict:
        """Thread-sicheres Append mit Lock."""
        with self._thread_lock:
            lock = self._ensure_lock()
            with lock:
                # ═══ FIX v1.4: Direkter _read_all() statt last_entry() ════════
                # Vermeidet nested File-Lock
                entries = self._read_all()
                last = entries[-1] if entries else None
                # ═════════════════════════════════════════════════════════════════

                prev_hash = last["entry_hash"] if last else self.GENESIS_HASH
                seq       = (last["seq"] + 1) if last else 0

                entry = {
                    "seq":       seq,
                    "prev_hash": prev_hash,
                    "timestamp": _utc_now(),
                    "manifest":  manifest,
                }
                entry["entry_hash"] = _sha256(_canonical(entry))

                with open(self.path, "a") as f:
                    f.write(json.dumps(entry, separators=(",", ":")) + "\n")
                    f.flush()
                    import os
                    os.fsync(f.fileno())

                return entry

    # ── Verifikation ──────────────────────────────────────────────────────────

    def verify_chain(self) -> tuple[bool, str]:
        """Prüft Hash-Kette. Deepcopy statt Mutation."""
        with self._thread_lock:
            lock = self._ensure_lock()
            with lock:
                entries = self._read_all()
                if not entries:
                    return True, "Registry leer."

                prev_hash = self.GENESIS_HASH

                for e in entries:
                    check_entry = copy.deepcopy(e)

                    stored_hash = check_entry.pop("entry_hash", None)
                    if stored_hash is None:
                        return False, f"Seq {check_entry['seq']}: Kein entry_hash."

                    expected = _sha256(_canonical(check_entry))

                    if stored_hash != expected:
                        return False, f"Seq {check_entry['seq']}: Entry-Hash manipuliert."
                    if check_entry["prev_hash"] != prev_hash:
                        return False, f"Seq {check_entry['seq']}: Chain-Link gebrochen."

                    prev_hash = stored_hash

                return True, f"Chain OK — {len(entries)} Einträge verifiziert."

    # ── Export ────────────────────────────────────────────────────────────────

    def list_all(self) -> list[dict]:
        with self._thread_lock:
            lock = self._ensure_lock()
            with lock:
                return self._read_all()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MAXXKI Registry")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list",   help="Alle Einträge anzeigen")
    sub.add_parser("verify", help="Hash-Chain verifizieren")
    p_find = sub.add_parser("find", help="Nach Session-ID oder Hash suchen")
    p_find.add_argument("query")

    args = parser.parse_args()
    reg  = MAXXKIRegistry()

    if args.cmd == "list":
        for e in reg.list_all():
            c = e["manifest"]["content"]
            print(f"  [{e['seq']:04d}] {e['timestamp']}  {c.get('session_id')}  {c.get('final_sha256','')[:16]}…")

    elif args.cmd == "verify":
        ok, msg = reg.verify_chain()
        print(f"{'✓' if ok else '✗'} {msg}")

    elif args.cmd == "find":
        e = reg.find_by_session(args.query) or reg.find_by_hash(args.query)
        if e:
            print(json.dumps(e, indent=2))
        else:
            print(f"✗ Nicht gefunden: {args.query}")

    else:
        parser.print_help()
