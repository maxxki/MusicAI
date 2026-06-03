"""
MAXXKI Sign Session v1.0
Wrapper — nach jeder MAXXKI-Produktion automatisch fingerprinting + registry.

Verwendung:
    from maxxki_sign_session import sign_and_register

    # Nach generator.generate() oder orchestrator.produce():
    entry = sign_and_register(
        wav_path=Path("output.wav"),
        session_id="mxk_20260603_134722",
        prompt="dark dancehall riddim, heavy 808",
        generator="musicgen-small",
        raw_audio=audio_array,   # optional, numpy
    )
    print(entry["entry_hash"])
"""

import scipy.io.wavfile
import numpy as np
from pathlib import Path
from typing import Optional

from maxxki_fingerprint import create_manifest, save_manifest
from maxxki_registry import MAXXKIRegistry


def sign_and_register(
    wav_path: Path,
    session_id: str,
    prompt: str,
    generator: str,
    raw_audio: Optional[np.ndarray] = None,
    registry: Optional[MAXXKIRegistry] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    1. Liest WAV (für sample_rate + duration)
    2. Erstellt signiertes Manifest
    3. Speichert .manifest.json neben der WAV
    4. Trägt in Registry ein
    5. Gibt Registry-Eintrag zurück
    """
    reg = registry or MAXXKIRegistry()

    sr, data = scipy.io.wavfile.read(str(wav_path))
    audio    = data.astype(np.float32)
    duration = len(audio) / sr

    manifest = create_manifest(
        session_id=session_id,
        prompt=prompt,
        generator=generator,
        duration_s=duration,
        sample_rate=sr,
        raw_audio=raw_audio,
        final_path=wav_path,
        extra=extra,
    )

    manifest_path = wav_path.with_suffix(".manifest.json")
    save_manifest(manifest, manifest_path)

    entry = reg.append(manifest)

    return entry


# ─── CLI: Batch-Signierung bestehender WAVs ───────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MAXXKI Sign Session — WAV(s) nachträglich signieren & registrieren"
    )
    parser.add_argument("wavs", nargs="+", help="WAV-Dateien")
    parser.add_argument("--prompt",    default="", help="Prompt")
    parser.add_argument("--generator", default="unknown")
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()

    reg = MAXXKIRegistry()

    for wav in args.wavs:
        wav_path   = Path(wav)
        session_id = args.session_id or f"manual_{wav_path.stem}"
        entry      = sign_and_register(
            wav_path=wav_path,
            session_id=session_id,
            prompt=args.prompt,
            generator=args.generator,
            registry=reg,
        )
        print(f"✓ {wav_path.name}  →  seq={entry['seq']}  hash={entry['entry_hash'][:16]}…")
