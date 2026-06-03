"""
MAXXKI Verify v1.0
Echtheit und Provenance eines Audio-Files prüfen.

Checks:
1. Datei-Hash stimmt mit Registry-Eintrag überein
2. Manifest-Signatur valide (HMAC)
3. Hash-Chain intakt (kein Eintrag wurde nachträglich manipuliert)
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from maxxki_fingerprint import hash_file, verify_signature, load_manifest
from maxxki_registry import MAXXKIRegistry


@dataclass
class VerificationResult:
    authentic:       bool
    chain_intact:    bool
    signature_valid: bool
    file_hash_match: bool
    session_id:      Optional[str]
    created_at:      Optional[str]
    prompt:          Optional[str]
    generator:       Optional[str]
    message:         str

    def print(self):
        icon = "✓" if self.authentic else "✗"
        print(f"\n{icon} MAXXKI Verify")
        print(f"  Authentic:       {self.authentic}")
        print(f"  Signature:       {'✓' if self.signature_valid else '✗'}")
        print(f"  File Hash Match: {'✓' if self.file_hash_match else '✗'}")
        print(f"  Chain Intact:    {'✓' if self.chain_intact else '✗'}")
        if self.session_id:
            print(f"  Session:         {self.session_id}")
        if self.created_at:
            print(f"  Created:         {self.created_at}")
        if self.prompt:
            print(f"  Prompt:          {self.prompt}")
        if self.generator:
            print(f"  Generator:       {self.generator}")
        print(f"  Message:         {self.message}\n")


def verify_file(wav_path: Path, registry: Optional[MAXXKIRegistry] = None) -> VerificationResult:
    """
    Verifiziert eine WAV-Datei gegen Registry + Manifest.

    Sucht zuerst nach einer .manifest.json im selben Verzeichnis,
    dann in der Registry per Datei-Hash.
    """
    reg = registry or MAXXKIRegistry()

    if not wav_path.exists():
        return VerificationResult(
            authentic=False, chain_intact=False, signature_valid=False,
            file_hash_match=False, session_id=None, created_at=None,
            prompt=None, generator=None, message=f"Datei nicht gefunden: {wav_path}"
        )

    file_hash = hash_file(wav_path)

    # ── 1. Manifest suchen ────────────────────────────────────────────────────
    manifest_path = wav_path.with_suffix(".manifest.json")
    manifest = None

    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
    else:
        entry = reg.find_by_hash(file_hash)
        if entry:
            manifest = entry.get("manifest")

    if manifest is None:
        return VerificationResult(
            authentic=False, chain_intact=False, signature_valid=False,
            file_hash_match=False, session_id=None, created_at=None,
            prompt=None, generator=None,
            message="Kein Manifest gefunden — Datei unbekannt oder nicht registriert."
        )

    content    = manifest.get("content", {})
    provenance = manifest.get("provenance", {})

    # ── 2. Datei-Hash prüfen ──────────────────────────────────────────────────
    file_hash_match = (content.get("final_sha256") == file_hash)

    # ── 3. Signatur prüfen ────────────────────────────────────────────────────
    signature_valid = verify_signature(manifest)

    # ── 4. Hash-Chain prüfen ──────────────────────────────────────────────────
    chain_ok, chain_msg = reg.verify_chain()

    authentic = file_hash_match and signature_valid and chain_ok

    return VerificationResult(
        authentic=authentic,
        chain_intact=chain_ok,
        signature_valid=signature_valid,
        file_hash_match=file_hash_match,
        session_id=content.get("session_id"),
        created_at=provenance.get("created_at"),
        prompt=content.get("prompt"),
        generator=content.get("generator"),
        message="OK" if authentic else (
            "Datei manipuliert." if not file_hash_match else
            "Signatur ungültig." if not signature_valid else
            chain_msg
        )
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MAXXKI Verify — Audio-Echtheit prüfen")
    parser.add_argument("wav", nargs="+", help="WAV-Datei(en)")
    args = parser.parse_args()

    reg = MAXXKIRegistry()
    all_ok = True

    for wav in args.wav:
        result = verify_file(Path(wav), reg)
        result.print()
        if not result.authentic:
            all_ok = False

    sys.exit(0 if all_ok else 1)
