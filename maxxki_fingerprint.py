"""
MAXXKI Fingerprint v1.0
Hash + Sign + Manifest für jeden generierten Sound.

Kein externer Service. Kein API-Key. 100% lokal.

Manifest enthält:
- raw_sha256     : Hash über Roh-Audio (vor Mastering)
- final_sha256   : Hash über finale WAV
- prompt         : Generator-Prompt
- generator      : Modell-ID
- session_id     : MAXXKI Session-ID
- provenance     : Zeitstempel, Host, User
- signature      : HMAC-SHA256 über das kanonische Manifest
"""

import hashlib
import hmac
import json
import os
import socket
import getpass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np

MANIFEST_VERSION = "1.0"
KEY_FILE = Path.home() / ".maxxki" / "signing.key"


# ─── Signing Key ──────────────────────────────────────────────────────────────

def _load_or_create_key() -> bytes:
    """Lokaler HMAC-Key. Einmalig generiert, lokal gespeichert."""
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    key = os.urandom(32)
    KEY_FILE.write_bytes(key)
    KEY_FILE.chmod(0o600)
    return key


# ─── Hashing ──────────────────────────────────────────────────────────────────

def hash_audio(audio: np.ndarray) -> str:
    """SHA256 über rohen Audio-Buffer (numpy array)."""
    return hashlib.sha256(audio.tobytes()).hexdigest()


def hash_file(path: Path) -> str:
    """SHA256 über Datei-Bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Manifest ─────────────────────────────────────────────────────────────────

def _canonical(d: dict) -> str:
    """Kanonisches JSON — sortierte Keys, kein Whitespace."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sign(manifest_content: dict) -> str:
    """HMAC-SHA256 über kanonisches Manifest."""
    key = _load_or_create_key()
    msg = _canonical(manifest_content).encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def create_manifest(
    session_id: str,
    prompt: str,
    generator: str,
    duration_s: float,
    sample_rate: int,
    raw_audio: Optional[np.ndarray] = None,
    final_path: Optional[Path] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Erstellt ein vollständiges, signiertes Manifest.

    raw_audio  : numpy array vor Mastering (für raw_sha256)
    final_path : Pfad zur fertigen WAV (für final_sha256)
    """
    content = {
        "manifest_version": MANIFEST_VERSION,
        "session_id": session_id,
        "prompt": prompt,
        "generator": generator,
        "duration_s": round(duration_s, 3),
        "sample_rate": sample_rate,
        "raw_sha256":   hash_audio(raw_audio) if raw_audio is not None else None,
        "final_sha256": hash_file(final_path) if final_path is not None else None,
    }
    if extra:
        content.update(extra)

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
    }

    manifest = {
        "content": content,
        "provenance": provenance,
    }

    manifest["signature"] = _sign(manifest)
    return manifest


def verify_signature(manifest: dict) -> bool:
    """Prüft HMAC-Signatur eines Manifests."""
    sig = manifest.get("signature")
    if not sig:
        return False
    check = {k: v for k, v in manifest.items() if k != "signature"}
    expected = _sign(check)
    return hmac.compare_digest(sig, expected)


# ─── Save / Load ──────────────────────────────────────────────────────────────

def save_manifest(manifest: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    return out_path


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import scipy.io.wavfile

    parser = argparse.ArgumentParser(description="MAXXKI Fingerprint — WAV signieren")
    parser.add_argument("wav", help="WAV-Datei")
    parser.add_argument("--prompt", default="", help="Prompt")
    parser.add_argument("--generator", default="unknown")
    parser.add_argument("--session-id", default="manual")
    args = parser.parse_args()

    wav_path = Path(args.wav)
    sr, data = scipy.io.wavfile.read(str(wav_path))
    audio = data.astype(np.float32)

    manifest = create_manifest(
        session_id=args.session_id,
        prompt=args.prompt,
        generator=args.generator,
        duration_s=len(audio) / sr,
        sample_rate=sr,
        raw_audio=audio,
        final_path=wav_path,
    )

    out = wav_path.with_suffix(".manifest.json")
    save_manifest(manifest, out)
    print(f"✓ Manifest: {out}")
    print(f"  SHA256:    {manifest['content']['final_sha256']}")
    print(f"  Signed:    {verify_signature(manifest)}")
