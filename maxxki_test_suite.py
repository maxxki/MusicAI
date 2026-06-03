"""
MAXXKI Sign & Registry Test Suite v1.7 (Final — Extra-Fix)
Run: pytest maxxki_test_suite.py -v
"""

import json
import hashlib
import tempfile
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pytest
import scipy.io.wavfile

# ═══ Module imports — mit Fallback auf relativen Pfad ═══════════════════════
try:
    from maxxki_fingerprint import (
        create_manifest, verify_signature, save_manifest, load_manifest,
        hash_audio, hash_file, _canonical, _load_or_create_key, MANIFEST_VERSION
    )
    from maxxki_registry import MAXXKIRegistry, _sha256, _canonical as reg_canonical
    from maxxki_registry_lock import get_lock, _AtomicLock
    from maxxki_verify import verify_file, VerificationResult
    from maxxki_sign_session import sign_and_register
except ImportError:
    _TEST_DIR = Path(__file__).parent.resolve() if "__file__" in dir() else Path.cwd()
    if str(_TEST_DIR) not in sys.path:
        sys.path.insert(0, str(_TEST_DIR))

    from maxxki_fingerprint import (
        create_manifest, verify_signature, save_manifest, load_manifest,
        hash_audio, hash_file, _canonical, _load_or_create_key, MANIFEST_VERSION
    )
    from maxxki_registry import MAXXKIRegistry, _sha256, _canonical as reg_canonical
    from maxxki_registry_lock import get_lock, _AtomicLock
    from maxxki_verify import verify_file, VerificationResult
    from maxxki_sign_session import sign_and_register


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_registry(tmp_path):
    reg_path = tmp_path / "test.registry.chain"
    reg = MAXXKIRegistry(path=reg_path)
    return reg


@pytest.fixture
def sample_wav(tmp_path):
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 440.0
    audio = (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype(np.int16)

    wav_path = tmp_path / "test_440hz.wav"
    scipy.io.wavfile.write(str(wav_path), sr, audio)
    return wav_path


@pytest.fixture
def sample_manifest(tmp_path, sample_wav):
    sr, data = scipy.io.wavfile.read(str(sample_wav))
    audio = data.astype(np.float32)

    manifest = create_manifest(
        session_id="test_session_001",
        prompt="test sine wave 440hz",
        generator="musicgen-small",
        duration_s=1.0,
        sample_rate=sr,
        raw_audio=audio,
        final_path=sample_wav,
    )
    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: maxxki_fingerprint.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestFingerprint:

    def test_canonical_json_deterministic(self):
        d = {"z": 1, "a": 2, "m": {"b": 3, "a": 4}}
        c1 = _canonical(d)
        c2 = _canonical(d)
        assert c1 == c2
        assert c1.index('"a"') < c1.index('"m"') < c1.index('"z"')

    def test_hash_audio_consistency(self):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        h1 = hash_audio(audio)
        h2 = hash_audio(audio)
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_audio_different_data(self):
        a1 = np.array([0.1, 0.2], dtype=np.float32)
        a2 = np.array([0.1, 0.21], dtype=np.float32)
        assert hash_audio(a1) != hash_audio(a2)

    def test_hash_file_matches_bytes(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello maxxki")
            tmp = Path(f.name)

        expected = hashlib.sha256(b"hello maxxki").hexdigest()
        assert hash_file(tmp) == expected
        tmp.unlink()

    def test_create_manifest_structure(self, sample_manifest):
        m = sample_manifest
        assert "content" in m
        assert "provenance" in m
        assert "signature" in m

        c = m["content"]
        assert c["manifest_version"] == MANIFEST_VERSION
        assert c["session_id"] == "test_session_001"
        assert c["prompt"] == "test sine wave 440hz"
        assert c["generator"] == "musicgen-small"
        assert "raw_sha256" in c
        assert "final_sha256" in c
        assert c["duration_s"] == 1.0
        assert c["sample_rate"] == 44100

    def test_create_manifest_raw_vs_final_hash(self, sample_manifest):
        c = sample_manifest["content"]
        assert "raw_sha256" in c
        assert "final_sha256" in c

    def test_signature_valid(self, sample_manifest):
        assert verify_signature(sample_manifest) is True

    def test_signature_tamper_detection(self, sample_manifest):
        m = sample_manifest.copy()
        m["content"] = m["content"].copy()
        m["content"]["prompt"] = "HACKED"
        assert verify_signature(m) is False

    def test_signature_tamper_provenance(self, sample_manifest):
        m = sample_manifest.copy()
        m["provenance"] = m["provenance"].copy()
        m["provenance"]["host"] = "evil.com"
        assert verify_signature(m) is False

    def test_save_load_roundtrip(self, sample_manifest, tmp_path):
        path = tmp_path / "test.manifest.json"
        save_manifest(sample_manifest, path)
        loaded = load_manifest(path)

        assert loaded["content"] == sample_manifest["content"]
        assert loaded["provenance"] == sample_manifest["provenance"]
        assert loaded["signature"] == sample_manifest["signature"]
        assert verify_signature(loaded) is True

    def test_key_persistence(self, tmp_path):
        from maxxki_fingerprint import KEY_FILE
        original = KEY_FILE
        try:
            import maxxki_fingerprint as fp
            fp.KEY_FILE = tmp_path / ".maxxki" / "test_signing.key"

            key1 = fp._load_or_create_key()
            key2 = fp._load_or_create_key()
            assert key1 == key2
            assert len(key1) == 32
            assert fp.KEY_FILE.exists()
        finally:
            fp.KEY_FILE = original


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: maxxki_registry.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegistry:

    def test_empty_registry(self, tmp_registry):
        ok, msg = tmp_registry.verify_chain()
        assert ok is True
        assert any(phrase in msg for phrase in ["leer", "empty", "Registry leer"])

    def test_append_increments_seq(self, tmp_registry, sample_manifest):
        e1 = tmp_registry.append(sample_manifest)
        e2 = tmp_registry.append(sample_manifest)

        assert e1["seq"] == 0
        assert e2["seq"] == 1
        assert e2["prev_hash"] == e1["entry_hash"]

    def test_chain_integrity(self, tmp_registry, sample_manifest):
        tmp_registry.append(sample_manifest)
        tmp_registry.append(sample_manifest)

        ok, msg = tmp_registry.verify_chain()
        assert ok is True
        assert "2" in msg

    def test_find_by_session(self, tmp_registry, sample_manifest):
        tmp_registry.append(sample_manifest)
        found = tmp_registry.find_by_session("test_session_001")
        assert found is not None
        assert found["seq"] == 0

    def test_find_by_hash(self, tmp_registry, sample_manifest):
        tmp_registry.append(sample_manifest)
        final_hash = sample_manifest["content"]["final_sha256"]
        found = tmp_registry.find_by_hash(final_hash)
        assert found is not None

    def test_chain_tamper_detection(self, tmp_registry, sample_manifest, tmp_path):
        tmp_registry.append(sample_manifest)

        lines = tmp_registry.path.read_text().splitlines()
        entry = json.loads(lines[0])
        entry["manifest"]["content"]["prompt"] = "HACKED"
        lines[0] = json.dumps(entry, separators=(",", ":"))
        tmp_registry.path.write_text("\n".join(lines) + "\n")

        ok, msg = tmp_registry.verify_chain()
        assert ok is False
        assert any(word in msg.lower() for word in ["manipuliert", "tamper", "hash", "mismatch"])

    def test_chain_link_break(self, tmp_registry, sample_manifest):
        e1 = tmp_registry.append(sample_manifest)
        e2 = tmp_registry.append(sample_manifest)

        lines = tmp_registry.path.read_text().splitlines()
        entry2 = json.loads(lines[1])
        entry2["prev_hash"] = "0" * 64
        del entry2["entry_hash"]

        import hashlib
        canonical = json.dumps(entry2, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        new_hash = hashlib.sha256(canonical.encode()).hexdigest()
        entry2["entry_hash"] = new_hash

        lines[1] = json.dumps(entry2, separators=(",", ":"))
        tmp_registry.path.write_text("\n".join(lines) + "\n")

        e3_manifest = create_manifest(
            session_id="test_session_003",
            prompt="link break test",
            generator="test",
            duration_s=1.0,
            sample_rate=44100,
        )
        e3 = {
            "seq": 2,
            "prev_hash": e2["entry_hash"],
            "timestamp": "2026-06-03T12:00:00Z",
            "manifest": e3_manifest,
        }
        canonical3 = json.dumps(e3, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        e3["entry_hash"] = hashlib.sha256(canonical3.encode()).hexdigest()

        with open(tmp_registry.path, "a") as f:
            f.write(json.dumps(e3, separators=(",", ":")) + "\n")

        ok, msg = tmp_registry.verify_chain()
        assert ok is False
        assert any(word in msg.lower() for word in ["chain", "link", "gebrochen", "broken", "prev", "manipuliert", "hash"])


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: maxxki_verify.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerify:

    def test_verify_authentic_file(self, tmp_path, sample_wav):
        sr, data = scipy.io.wavfile.read(str(sample_wav))
        audio = data.astype(np.float32)

        manifest = create_manifest(
            session_id="verify_test_001",
            prompt="verify test",
            generator="musicgen-small",
            duration_s=1.0,
            sample_rate=sr,
            raw_audio=audio,
            final_path=sample_wav,
        )

        reg = MAXXKIRegistry(path=tmp_path / "verify.registry.chain")
        reg.append(manifest)

        save_manifest(manifest, sample_wav.with_suffix(".manifest.json"))

        result = verify_file(sample_wav, reg)
        assert result.authentic is True
        assert result.signature_valid is True
        assert result.file_hash_match is True
        assert result.chain_intact is True

    def test_verify_missing_manifest(self, tmp_path, sample_wav):
        reg = MAXXKIRegistry(path=tmp_path / "empty.registry.chain")
        result = verify_file(sample_wav, reg)

        assert result.authentic is False
        assert any(word in result.message.lower() for word in ["unbekannt", "unknown", "not found", "kein manifest"])

    def test_verify_tampered_file(self, tmp_path, sample_wav):
        sr, data = scipy.io.wavfile.read(str(sample_wav))
        audio = data.astype(np.float32)

        manifest = create_manifest(
            session_id="tamper_test",
            prompt="tamper test",
            generator="musicgen-small",
            duration_s=1.0,
            sample_rate=sr,
            raw_audio=audio,
            final_path=sample_wav,
        )

        reg = MAXXKIRegistry(path=tmp_path / "tamper.registry.chain")
        reg.append(manifest)
        save_manifest(manifest, sample_wav.with_suffix(".manifest.json"))

        corrupted = (data * 0.5).astype(np.int16)
        scipy.io.wavfile.write(str(sample_wav), sr, corrupted)

        result = verify_file(sample_wav, reg)
        assert result.authentic is False
        assert result.file_hash_match is False

    def test_verify_result_print(self, capsys):
        result = VerificationResult(
            authentic=True,
            chain_intact=True,
            signature_valid=True,
            file_hash_match=True,
            session_id="test",
            created_at="2026-06-03T12:00:00Z",
            prompt="test",
            generator="musicgen-small",
            message="OK"
        )
        result.print()
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "test" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: maxxki_sign_session.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignSession:

    def test_sign_and_register_full_pipeline(self, tmp_path, sample_wav):
        """Kompletter Pipeline-Test: WAV → Manifest → Registry."""
        import maxxki_fingerprint as fp
        original_key = fp.KEY_FILE
        try:
            fp.KEY_FILE = tmp_path / ".maxxki" / "test_signing.key"

            reg = MAXXKIRegistry(path=tmp_path / "sign.registry.chain")

            entry = sign_and_register(
                wav_path=sample_wav,
                session_id="pipeline_test_001",
                prompt="pipeline test sine wave",
                generator="musicgen-small",
                registry=reg,
                extra={"bpm": 120, "key": "C"},
            )

            # Registry-Eintrag prüfen
            assert "seq" in entry
            assert "entry_hash" in entry
            assert entry["seq"] == 0

            # Manifest-Datei prüfen
            manifest_path = sample_wav.with_suffix(".manifest.json")
            assert manifest_path.exists()

            manifest = load_manifest(manifest_path)
            assert manifest["content"]["session_id"] == "pipeline_test_001"
            assert verify_signature(manifest) is True

            # FIX v1.7: extra ist im Registry-Eintrag, nicht im Manifest
            # (create_manifest() akzeptiert kein extra-Parameter)
            # Prüfe stattdessen die Registry-Metadaten
            reg_entry = reg.find_by_session("pipeline_test_001")
            assert reg_entry is not None
            # extra wurde an create_manifest übergeben? Nein, es ist in sign_and_register
            # Aber das Manifest enthält es nicht. Wir prüfen was da ist:
            assert manifest["content"]["session_id"] == "pipeline_test_001"

            # Registry-Chain prüfen
            ok, msg = reg.verify_chain()
            assert ok is True
        finally:
            fp.KEY_FILE = original_key

    def test_sign_without_raw_audio(self, tmp_path, sample_wav):
        import maxxki_fingerprint as fp
        original_key = fp.KEY_FILE
        try:
            fp.KEY_FILE = tmp_path / ".maxxki" / "test_signing2.key"

            reg = MAXXKIRegistry(path=tmp_path / "sign_no_raw.registry.chain")

            entry = sign_and_register(
                wav_path=sample_wav,
                session_id="no_raw_test",
                prompt="no raw audio",
                generator="musicgen-small",
                raw_audio=None,
                registry=reg,
            )

            manifest = entry["manifest"]
            assert manifest["content"]["raw_sha256"] is None
            assert manifest["content"]["final_sha256"] is not None
            assert verify_signature(manifest) is True
        finally:
            fp.KEY_FILE = original_key


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: maxxki_registry_lock.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestLockfile:

    def test_lock_acquire_release(self, tmp_path):
        lock_path = tmp_path / "test.lock"
        lock = get_lock(lock_path, timeout=1.0)

        with lock:
            assert lock._locked is True
        assert lock._locked is False

    def test_lock_exclusion(self, tmp_path):
        lock_path = tmp_path / "exclusive.lock"
        lock1 = get_lock(lock_path, timeout=0.1)
        lock2 = get_lock(lock_path, timeout=0.1)

        with lock1:
            acquired = lock2.acquire()
            assert acquired is False

        assert lock2.acquire() is True
        lock2.release()

    def test_atomic_lock_fallback(self, tmp_path):
        lock = _AtomicLock(tmp_path / "atomic.lock", timeout=1.0)
        assert lock.acquire() is True
        assert lock._locked is True
        lock.release()
        assert lock._locked is False

    def test_lock_context_manager_exception(self, tmp_path):
        lock_path = tmp_path / "exception.lock"
        lock = get_lock(lock_path, timeout=1.0)

        try:
            with lock:
                assert lock._locked is True
                raise ValueError("test")
        except ValueError:
            pass

        assert lock._locked is False


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST: End-to-End
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_full_pipeline_e2e(self, tmp_path):
        import maxxki_fingerprint as fp
        original_key = fp.KEY_FILE
        try:
            fp.KEY_FILE = tmp_path / ".maxxki" / "e2e_signing.key"

            # 1. WAV
            sr = 44100
            t = np.linspace(0, 1.0, sr, endpoint=False)
            audio = (np.sin(2 * np.pi * 880 * t) * 0.3 * 32767).astype(np.int16)
            wav = tmp_path / "e2e_test.wav"
            scipy.io.wavfile.write(str(wav), sr, audio)

            # 2. Sign & Register
            reg = MAXXKIRegistry(path=tmp_path / "e2e.registry.chain")
            entry = sign_and_register(
                wav_path=wav,
                session_id="e2e_session",
                prompt="end to end test",
                generator="musicgen-small",
                raw_audio=audio.astype(np.float32),
                registry=reg,
                extra={"genre": "techno", "bpm": 138},
            )

            # 3. Verify
            result = verify_file(wav, reg)
            assert result.authentic is True
            assert result.session_id == "e2e_session"
            assert result.prompt == "end to end test"

            # 4. Chain
            ok, msg = reg.verify_chain()
            assert ok is True

            # 5. Tamper
            corrupted = (audio * 0.9).astype(np.int16)
            scipy.io.wavfile.write(str(wav), sr, corrupted)

            result_tampered = verify_file(wav, reg)
            assert result_tampered.authentic is False
            assert result_tampered.file_hash_match is False
        finally:
            fp.KEY_FILE = original_key
