# ═══════════════════════════════════════════════════════════════════════════════
# MAXXKI Orchestrator v1.2 — Sign & Registry Integration (Fixed)
# ═══════════════════════════════════════════════════════════════════════════════
"""
MAXXKI Orchestrator v1.2 — Sign & Registry Integration (Fixed v1.2.1)

Änderungen v1.2:
- Post-Mastering: automatische Fingerprint-Erzeugung + Registry-Eintrag
- raw_audio Capture vor Mastering für raw_sha256
- Registry-Metadaten erweitert (BPM, Key, Genre, Profile, Track-Liste)
- Lockfile-Schutz für parallele Session-Writes
- ProductionSession um registry_entry erweitert

FIX v1.2.1:
- Graceful degradation: MusicGen/MIDI-Renderer/Registry optional
- Timeout für async_generate (kein Hang bei fehlendem Model)
- fluidsynth-Check vor MIDI-Rendering
- Registry-Fehler führt nicht zu Session-Abort

Pipeline:
    produce() → _mix_and_master() → _sign_and_register() → _cleanup()
"""

import os
import sys
import json
import logging
import asyncio
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

import numpy as np
import soundfile as sf
import scipy.signal
import scipy.io.wavfile

# ═══ Optional imports — graceful degradation ════════════════════════════════
try:
    from maxxki_master_ear import (
        MasterEarEngine, VibeProfile, MasteringProfile, 
        PostProcessor, VIBE_PROFILES
    )
    _HAS_MASTER_EAR = True
except ImportError:
    _HAS_MASTER_EAR = False
    logging.warning("maxxki_master_ear nicht verfügbar — rule-basierte Profile")

try:
    from maxxki_beat_gen import (
        PROGRESSIONS, GENRE_SCALE_MAP, GENRE_DRUM_MAP,
        SCALES, CHORD_TYPES, add_chords, add_bass, add_melody, add_drums,
        humanize, scale_notes, chord_notes_absolute
    )
    _HAS_BEAT_GEN = True
except ImportError:
    _HAS_BEAT_GEN = False
    logging.warning("maxxki_beat_gen nicht verfügbar — MIDI-Generation deaktiviert")

try:
    from maxxki_midi_gen import build_prompt, generate_tokens, tokens_to_midi
    _HAS_MIDI_GEN = True
except ImportError:
    _HAS_MIDI_GEN = False
    logging.warning("maxxki_midi_gen nicht verfügbar — AI-Melodie deaktiviert")

try:
    from maxxki_music_gen import MusicGenerator, MusicGeneratorConfig, GENRE_PRESETS
    _HAS_MUSIC_GEN = True
except ImportError:
    _HAS_MUSIC_GEN = False
    logging.warning("maxxki_music_gen nicht verfügbar — MusicGen deaktiviert")

# ═══ NEU v1.2: Sign & Registry Imports (optional) ══════════════════════════
try:
    from maxxki_sign_session import sign_and_register
    from maxxki_registry import MAXXKIRegistry
    _HAS_REGISTRY = True
except ImportError:
    _HAS_REGISTRY = False
    logging.warning("Sign/Registry nicht verfügbar — Signierung deaktiviert")
# ═════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("orchestrator.log"),
    ],
)
logger = logging.getLogger("MAXXKI-Orchestrator")

OUTPUT_DIR = Path.home() / "projects/mk/music-output"
FINAL_DIR = OUTPUT_DIR / "final"
STEM_DIR = OUTPUT_DIR / "stems"
TEMP_DIR = Path(tempfile.gettempdir()) / "maxxki_sessions"

# SoundFont-Suche erweitert
SOUNDFONT_PATHS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/GeneralUser_GS.sf2",
    "/usr/share/sounds/sf2/default.sf2",
    "/var/home/mk/projects/mk/music/VintageDreamsWaves-v2.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
]

def find_soundfont() -> Optional[Path]:
    for p in SOUNDFONT_PATHS:
        if Path(p).exists():
            return Path(p)
    return None

# ═══ FIX v1.2.1: Prüfe ob fluidsynth verfügbar ═══════════════════════════════
def _has_fluidsynth() -> bool:
    try:
        subprocess.run(["fluidsynth", "--version"], 
                      capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

_HAS_FLUIDSYNTH = _has_fluidsynth()
if not _HAS_FLUIDSYNTH:
    logger.warning("fluidsynth nicht verfügbar — MIDI-Rendering deaktiviert")
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackSpec:
    name: str
    source: str
    channel: int
    instrument: int
    pan: float = 0.0
    gain_db: float = 0.0
    eq_profile: str = "flat"
    send_reverb: float = 0.0
    send_delay: float = 0.0
    midi_path: Optional[Path] = None
    audio_path: Optional[Path] = None

@dataclass
class SessionConfig:
    user_prompt: str
    vibe_hint: Optional[str] = None
    duration_seconds: int = 60
    bpm: Optional[int] = None
    key: str = "C"
    genre: Optional[str] = None
    use_beat_gen: bool = True
    use_midi_gen: bool = True
    use_music_gen: bool = True
    use_claude: bool = True

@dataclass
class ProductionSession:
    session_id: str
    config: SessionConfig
    profile: Optional[Any] = None  # FIX: Optional für graceful degradation
    track_plan: Dict[str, Any] = field(default_factory=dict)
    tracks: List[TrackSpec] = field(default_factory=list)
    stems: Dict[str, Path] = field(default_factory=dict)
    final_mix: Optional[Path] = None
    # ═══ NEU v1.2 ═════════════════════════════════════════════════════════════
    registry_entry: Optional[Dict] = None
    raw_audio_pre_master: Optional[np.ndarray] = None
    # ═════════════════════════════════════════════════════════════════════════
    created_at: datetime = field(default_factory=datetime.now)

class MIDIRenderer:
    def __init__(self, soundfont: Optional[Path] = None):
        self.sf = soundfont or find_soundfont()
        if self.sf is None or not _HAS_FLUIDSYNTH:
            logger.warning("MIDI-Rendering nicht verfügbar (kein SoundFont oder fluidsynth)")
            self.sf = None

    def render_track(self, midi_path: Path, output_path: Path, 
                     track_idx: int = 0, duration: Optional[float] = None) -> Path:
        if self.sf is None:
            raise RuntimeError("MIDI-Rendering nicht verfügbar")
        cmd = [
            "fluidsynth", "-ni", "-g", "0.8", "-r", "44100",
            "-o", "synth.chorus.active=0",
            str(self.sf), str(midi_path), "-F", str(output_path),
        ]
        if duration:
            cmd.extend(["-T", str(duration), "-E", "alsa"])
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        logger.info(f"Rendered: {output_path}")
        return output_path

    def render_stems(self, midi_path: Path, output_dir: Path,
                     tracks: List[TrackSpec]) -> Dict[str, Path]:
        if self.sf is None:
            return {}
        output_dir.mkdir(parents=True, exist_ok=True)
        stems = {}
        for track in tracks:
            if track.source in ("beat_gen", "midi_gen"):
                wav_path = output_dir / f"{track.name}_{midi_path.stem}.wav"
                self.render_track(midi_path, wav_path, track.channel)
                track.audio_path = wav_path
                stems[track.name] = wav_path
        return stems

class StemMixer:
    def __init__(self, profile: Any, bpm: int, sr: int = 44100):
        self.profile = profile
        self.bpm = bpm
        self.sr = sr
        self.beat_duration = 60.0 / bpm
        self.bar_duration = 4 * self.beat_duration
        # FIX: PostProcessor nur wenn verfügbar
        if _HAS_MASTER_EAR and profile is not None:
            self.master_chain = PostProcessor(profile)
        else:
            self.master_chain = None

    def mix_stems(self, stems: Dict[str, np.ndarray], 
                  track_specs: List[TrackSpec],
                  duration_seconds: float) -> np.ndarray:
        total_samples = int(duration_seconds * self.sr)
        mix = np.zeros((total_samples, 2), dtype=np.float64)
        reverb_bus = np.zeros((total_samples, 2), dtype=np.float64)
        delay_bus = np.zeros((total_samples, 2), dtype=np.float64)

        for spec in track_specs:
            if spec.name not in stems:
                continue
            audio = stems[spec.name]
            audio = self._normalize_length(audio, total_samples)
            if audio.ndim == 1:
                audio = np.column_stack([audio, audio])
            audio = self._apply_gain_pan(audio, spec.gain_db, spec.pan)
            audio = self._apply_eq(audio, spec.eq_profile)
            mix += audio
            if spec.send_reverb > 0:
                reverb_bus += audio * spec.send_reverb
            if spec.send_delay > 0:
                delay_bus += audio * spec.send_delay

        if np.max(np.abs(reverb_bus)) > 0:
            wet = self._simple_reverb(reverb_bus)
            mix += wet * 0.3
        if np.max(np.abs(delay_bus)) > 0:
            wet = self._simple_delay(delay_bus)
            mix += wet * 0.25

        mix = np.tanh(mix * 0.8)
        return mix.astype(np.float32)

    def _normalize_length(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        if len(audio) > target_samples:
            return audio[:target_samples]
        elif len(audio) < target_samples:
            pad_shape = (target_samples - len(audio),) + audio.shape[1:]
            pad = np.zeros(pad_shape, dtype=audio.dtype)
            return np.concatenate([audio, pad])
        return audio

    def _apply_gain_pan(self, audio: np.ndarray, gain_db: float, pan: float) -> np.ndarray:
        gain = 10 ** (gain_db / 20)
        left = gain * np.cos((pan + 1) * np.pi / 4)
        right = gain * np.sin((pan + 1) * np.pi / 4)
        audio[:, 0] *= left
        audio[:, 1] *= right
        return audio

    def _apply_eq(self, audio: np.ndarray, profile: str) -> np.ndarray:
        ax = 0
        if profile == "drums":
            sos = scipy.signal.butter(2, 30/(self.sr/2), btype='high', output='sos')
            audio = scipy.signal.sosfilt(sos, audio, axis=ax)
        elif profile == "bass":
            sos_hp = scipy.signal.butter(2, 40/(self.sr/2), btype='high', output='sos')
            sos_lp = scipy.signal.butter(2, 800/(self.sr/2), btype='low', output='sos')
            audio = scipy.signal.sosfilt(sos_hp, audio, axis=ax)
            audio = scipy.signal.sosfilt(sos_lp, audio, axis=ax)
        elif profile == "melodic":
            sos = scipy.signal.butter(2, 200/(self.sr/2), btype='high', output='sos')
            audio = scipy.signal.sosfilt(sos, audio, axis=ax)
        return audio

    def _simple_reverb(self, audio: np.ndarray) -> np.ndarray:
        delays = [int(self.sr * d) for d in [0.0297, 0.0371, 0.0411, 0.0437]]
        wet = np.zeros_like(audio)
        for d in delays:
            if d < len(audio):
                wet[d:] += audio[:-d] * 0.5
        return wet * 0.3

    def _simple_delay(self, audio: np.ndarray) -> np.ndarray:
        delay_s = (60/self.bpm) * 0.75
        delay_samples = int(self.sr * delay_s)
        wet = np.zeros_like(audio)
        if delay_samples < len(audio):
            wet[delay_samples:] += audio[:-delay_samples] * 0.4
        return wet

    def master(self, mix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mastered + raw_mono zurückgeben.
        FIX v1.2.1: Fallback wenn PostProcessor nicht verfügbar.
        """
        if mix.ndim == 2:
            mono = mix.mean(axis=1)
            raw_mono = mono.copy()

            if self.master_chain is not None:
                mastered_mono = self.master_chain.process(mono, self.sr)
                scale = np.max(np.abs(mastered_mono)) / (np.max(np.abs(raw_mono)) + 1e-9)
                result = (mix * scale).astype(np.float32)
            else:
                # Kein Mastering verfügbar — raw = final
                mastered_mono = raw_mono
                result = mix.astype(np.float32)

            return result, raw_mono.astype(np.float32)

        if self.master_chain is not None:
            mastered = self.master_chain.process(mix, self.sr)
        else:
            mastered = mix
        return mastered.astype(np.float32), mix.astype(np.float32)

class MAXXKIProducer:
    def __init__(self):
        self.master_ear = None
        self.music_gen = None
        self.midi_renderer = None
        self.registry = None
        self._init_engines()

    def _init_engines(self):
        # Master-Ear (optional)
        if _HAS_MASTER_EAR:
            try:
                self.master_ear = MasterEarEngine()
                logger.info("Master-Ear initialisiert")
            except Exception as e:
                logger.warning(f"Master-Ear nicht verfügbar: {e}")

        # MusicGen (optional)
        if _HAS_MUSIC_GEN:
            try:
                self.music_gen = MusicGenerator.get_instance(MusicGeneratorConfig())
                logger.info("MusicGen initialisiert")
            except Exception as e:
                logger.warning(f"MusicGen nicht verfügbar: {e}")
                self.music_gen = None

        # MIDI-Renderer (optional)
        if _HAS_BEAT_GEN:
            try:
                self.midi_renderer = MIDIRenderer()
                if self.midi_renderer.sf is not None:
                    logger.info("MIDI-Renderer initialisiert")
            except Exception as e:
                logger.warning(f"MIDI-Renderer nicht verfügbar: {e}")

        # Registry (optional)
        if _HAS_REGISTRY:
            try:
                self.registry = MAXXKIRegistry()
                logger.info("Registry initialisiert")
            except Exception as e:
                logger.warning(f"Registry nicht verfügbar: {e}")

    async def produce(self, config: SessionConfig) -> ProductionSession:
        session_id = f"mxk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"=== Session {session_id} START ===")

        # Profile bestimmen (mit Fallback)
        profile = None
        musicgen_prompt = config.user_prompt
        analysis = {}

        if config.use_claude and self.master_ear and _HAS_MASTER_EAR:
            try:
                profile, musicgen_prompt, analysis = self.master_ear.analyze(
                    config.user_prompt, vibe_hint=config.vibe_hint
                )
            except Exception as e:
                logger.warning(f"Master-Ear Analyse fehlgeschlagen: {e}")

        if profile is None:
            # Fallback-Profile
            if _HAS_MASTER_EAR and config.vibe_hint:
                try:
                    profile = VIBE_PROFILES.get(
                        VibeProfile(config.vibe_hint),
                        VIBE_PROFILES[VibeProfile.THE_PRODUCK]
                    )
                except:
                    profile = None
            if profile is None:
                # Minimal-Fallback
                profile = type('obj', (object,), {
                    'name': 'Fallback',
                    'midi_bpm': 120,
                    'midi_density': 'sparse',
                    'sub_boost_db': 4.0,
                    'mid_scoop_db': -2.0,
                    'high_shelf_db': 2.0,
                    'compression_ratio': 4.0,
                    'compression_threshold': -12.0,
                    'attack_ms': 10.0,
                    'release_ms': 100.0,
                    'sidechain_enabled': True,
                    'sidechain_amount': 0.4,
                    'reverb_size': 0.3,
                    'reverb_damp': 0.8,
                    'reverb_wet': 0.15,
                })()

        bpm = config.bpm or (getattr(profile, 'midi_bpm', 120))
        genre = config.genre or analysis.get("vibe_profile", "dark")

        session = ProductionSession(
            session_id=session_id,
            config=config,
            profile=profile,
            track_plan=analysis,
        )

        # Track-Plan (nur wenn beat_gen verfügbar)
        if _HAS_BEAT_GEN:
            tracks = self._create_track_plan(genre, profile, analysis)
            session.tracks = tracks

            midi_stems = self._generate_midi_foundation(session, bpm, genre)

            if self.midi_renderer and midi_stems:
                stem_dir = STEM_DIR / session_id
                try:
                    rendered = self.midi_renderer.render_stems(
                        midi_stems["combined"], stem_dir, tracks
                    )
                    session.stems.update(rendered)
                except Exception as e:
                    logger.warning(f"MIDI-Rendering fehlgeschlagen: {e}")
        else:
            session.tracks = []

        # MusicGen Texture (optional, mit Timeout)
        if config.use_music_gen and self.music_gen and _HAS_MUSIC_GEN:
            try:
                texture_path = await asyncio.wait_for(
                    self._generate_texture(musicgen_prompt, profile),
                    timeout=300  # 5 Min Timeout
                )
                if texture_path:
                    texture_track = TrackSpec(
                        name="texture", source="music_gen", channel=-1, instrument=-1,
                        pan=0.0, gain_db=-6, eq_profile="atmospheric", send_reverb=0.4,
                        audio_path=texture_path
                    )
                    session.tracks.append(texture_track)
                    session.stems["texture"] = texture_path
            except asyncio.TimeoutError:
                logger.warning("MusicGen Timeout — Texture übersprungen")
            except Exception as e:
                logger.warning(f"MusicGen fehlgeschlagen: {e}")

        # Mix & Master
        if session.stems:
            try:
                final_path, raw_audio = await self._mix_and_master(session, bpm)
                session.final_mix = final_path
                session.raw_audio_pre_master = raw_audio

                # Sign & Register
                await self._sign_and_register(session, bpm, genre)
            except Exception as e:
                logger.error(f"Mix/Master fehlgeschlagen: {e}")
                # Versuche trotzdem einen Output
                if session.stems:
                    first_stem = list(session.stems.values())[0]
                    if first_stem.exists():
                        session.final_mix = first_stem
                        logger.info(f"Fallback: erster Stem als Final {first_stem}")

        self._cleanup(session)
        logger.info(f"=== Session {session_id} COMPLETE → {session.final_mix} ===")
        return session

    def _create_track_plan(self, genre: str, profile: Any, 
                          analysis: Dict) -> List[TrackSpec]:
        tracks = []
        tracks.append(TrackSpec(
            name="drums", source="beat_gen", channel=9, instrument=0,
            pan=0.0, gain_db=-2, eq_profile="drums", send_reverb=0.1
        ))
        tracks.append(TrackSpec(
            name="bass", source="beat_gen", channel=1, instrument=33,
            pan=0.1, gain_db=-1, eq_profile="bass", send_reverb=0.0
        ))
        tracks.append(TrackSpec(
            name="chords", source="beat_gen", channel=0, instrument=88,
            pan=-0.2, gain_db=-3, eq_profile="harmonic", send_reverb=0.3
        ))
        density = getattr(profile, 'midi_density', 'sparse') if profile else 'sparse'
        if analysis.get("structure_plan", {}).get("midi_density") != "none" and density != "none":
            tracks.append(TrackSpec(
                name="melody", source="midi_gen", channel=2, instrument=80,
                pan=0.3, gain_db=-2, eq_profile="melodic", send_reverb=0.2, send_delay=0.15
            ))
        return tracks

    def _generate_midi_foundation(self, session: ProductionSession, 
                                   bpm: int, genre: str) -> Optional[Dict[str, Path]]:
        if not _HAS_BEAT_GEN:
            return None

        from midiutil import MIDIFile
        bars = max(4, int(session.config.duration_seconds * bpm / 240))

        midi = MIDIFile(4)
        for t in range(4):
            midi.addTempo(t, 0, bpm)

        key_map = {"C":60,"C#":61,"D":62,"D#":63,"E":64,
                   "F":65,"F#":66,"G":67,"G#":68,"A":69,"A#":70,"B":71}
        key_midi = key_map.get(session.config.key, 60)

        scale = SCALES.get(GENRE_SCALE_MAP.get(genre, "minor"), SCALES["minor"])
        progression = PROGRESSIONS.get(genre, PROGRESSIONS["dark"])
        drum_pattern = GENRE_DRUM_MAP.get(genre, "basic")

        add_chords(midi, 0, 0, progression, key_midi, scale, bars, bpm, "block")
        add_bass(midi, 1, 1, progression, key_midi, scale, bars, "root")
        add_drums(midi, 3, 9, drum_pattern, bars, bpm)

        if session.config.use_midi_gen and _HAS_MIDI_GEN:
            try:
                density = getattr(session.profile, 'midi_density', 'sparse') if session.profile else 'sparse'
                prompt = build_prompt("synth-lead", density, bars=4)
                tokens = generate_tokens(prompt, max_new_tokens=512)
                melody_midi = TEMP_DIR / f"{session.session_id}_melody.mid"
                tokens_to_midi(tokens, melody_midi, bpm=bpm)
                melody_tracks = [t for t in session.tracks if t.name == "melody"]
                if melody_tracks:
                    melody_tracks[0].midi_path = melody_midi
            except Exception as e:
                logger.warning(f"MIDI-Gen Melodie fehlgeschlagen: {e}")
                add_melody(midi, 2, 2, progression, key_midi, scale, bars, 0.5)
        else:
            add_melody(midi, 2, 2, progression, key_midi, scale, bars, 0.5)

        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        combined_path = TEMP_DIR / f"{session.session_id}_foundation.mid"
        with open(combined_path, "wb") as f:
            midi.writeFile(f)

        return {"combined": combined_path}

    async def _generate_texture(self, prompt: str, profile: Any) -> Optional[Path]:
        if not _HAS_MUSIC_GEN or self.music_gen is None:
            return None
        try:
            bpm = getattr(profile, 'midi_bpm', 120) if profile else 120
            scoop = getattr(profile, 'mid_scoop_db', -2) if profile else -2
            enhanced_prompt = (
                f"{prompt}, {bpm} BPM, "
                f"{'dark' if scoop < -2 else 'bright'} atmosphere, "
                f"minimalist, sparse texture"
            )
            path = await self.music_gen.async_generate(enhanced_prompt, duration=30)
            return path
        except Exception as e:
            logger.error(f"MusicGen Texture fehlgeschlagen: {e}")
            return None

    async def _mix_and_master(self, session: ProductionSession, bpm: int) -> Tuple[Path, np.ndarray]:
        stem_audio = {}
        for name, path in session.stems.items():
            if path.exists():
                try:
                    audio, sr = sf.read(str(path))
                    if sr != 44100:
                        audio = scipy.signal.resample(audio, int(len(audio) * 44100 / sr))
                    stem_audio[name] = audio
                except Exception as e:
                    logger.warning(f"Stem {name} laden fehlgeschlagen: {e}")

        mixer = StemMixer(session.profile, bpm, sr=44100)
        duration = session.config.duration_seconds
        mix = mixer.mix_stems(stem_audio, session.tracks, duration)
        mastered, raw_mono = mixer.master(mix)

        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        final_path = FINAL_DIR / f"{session.session_id}_mastered.wav"
        sf.write(str(final_path), mastered, 44100)

        return final_path, raw_mono

    async def _sign_and_register(self, session: ProductionSession, bpm: int, genre: str) -> None:
        if not _HAS_REGISTRY or self.registry is None:
            logger.debug("Signierung deaktiviert (Registry nicht verfügbar)")
            return

        try:
            extra = {
                "bpm": bpm,
                "key": session.config.key,
                "genre": genre,
                "mastering_profile": getattr(session.profile, 'name', 'unknown') if session.profile else 'unknown',
                "vibe": session.config.vibe_hint,
                "tracks": [t.name for t in session.tracks],
                "track_sources": {t.name: t.source for t in session.tracks},
                "midi_density": getattr(session.profile, 'midi_density', 'unknown') if session.profile else 'unknown',
                "duration_configured": session.config.duration_seconds,
            }

            entry = sign_and_register(
                wav_path=session.final_mix,
                session_id=session.session_id,
                prompt=session.config.user_prompt,
                generator="musicgen-small",
                raw_audio=session.raw_audio_pre_master,
                registry=self.registry,
                extra=extra,
            )

            session.registry_entry = entry
            logger.info(f"✓ Signiert & registriert: seq={entry['seq']} hash={entry['entry_hash'][:16]}…")

        except Exception as e:
            logger.error(f"Signierung fehlgeschlagen: {e}")
            # Nicht fatal — Production läuft weiter

    def _cleanup(self, session: ProductionSession):
        pass

# ═══ CLI ══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MAXXKI Autonomous Producer v1.2.1",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("prompt", nargs="?", default="", 
                       help="Beschreibung des gewünschten Tracks")
    parser.add_argument("--vibe", default="the_produck", help="Vibe-Profile")
    parser.add_argument("--bpm", type=int, help="BPM")
    parser.add_argument("--key", default="C", help="Tonart")
    parser.add_argument("--duration", type=int, default=60, help="Dauer in Sekunden")
    parser.add_argument("--no-claude", action="store_true", help="Master-Ear deaktivieren")
    parser.add_argument("--no-midi-gen", action="store_true", help="AI-Melodie deaktivieren")
    parser.add_argument("--no-music-gen", action="store_true", help="Atmosphäre deaktivieren")
    parser.add_argument("--no-sign", action="store_true", 
                       help="Signierung & Registry deaktivieren")
    parser.add_argument("--verify", type=str, metavar="WAV_PATH",
                       help="Bestehende WAV verifizieren (statt produzieren)")
    parser.add_argument("--list-engines", action="store_true",
                       help="Verfügbare Engines anzeigen und beenden")
    args = parser.parse_args()

    # Engine-Status anzeigen
    if args.list_engines:
        print("\n⬡ MAXXKI Engine Status:")
        print(f"  Master-Ear:   {'✓' if _HAS_MASTER_EAR else '✗'}")
        print(f"  Beat-Gen:     {'✓' if _HAS_BEAT_GEN else '✗'}")
        print(f"  MIDI-Gen:     {'✓' if _HAS_MIDI_GEN else '✗'}")
        print(f"  MusicGen:     {'✓' if _HAS_MUSIC_GEN else '✗'}")
        print(f"  fluidsynth:   {'✓' if _HAS_FLUIDSYNTH else '✗'}")
        print(f"  Registry:     {'✓' if _HAS_REGISTRY else '✗'}")
        print()
        sys.exit(0)

    # Verify-Modus
    if args.verify:
        if not _HAS_REGISTRY:
            print("✗ Registry nicht verfügbar — Verifikation nicht möglich")
            sys.exit(1)
        from maxxki_verify import verify_file
        result = verify_file(Path(args.verify))
        result.print()
        sys.exit(0 if result.authentic else 1)

    # Production-Modus
    if not args.prompt:
        print("\n⬡ MAXXKI Autonomous Producer v1.2.1")
        print("  Usage: python maxxki_orchestrator.py \"dark dancehall riddim\"")
        print("  --list-engines für verfügbare Engines")
        sys.exit(0)

    config = SessionConfig(
        user_prompt=args.prompt,
        vibe_hint=args.vibe,
        bpm=args.bpm,
        key=args.key,
        duration_seconds=args.duration,
        use_claude=not args.no_claude,
        use_midi_gen=not args.no_midi_gen,
        use_music_gen=not args.no_music_gen,
    )

    producer = MAXXKIProducer()

    if args.no_sign:
        producer.registry = None
        logger.info("Signierung deaktiviert (--no-sign)")

    try:
        session = asyncio.run(producer.produce(config))
        print(f"\n{'='*60}")
        print(f"  MAXXKI Production Complete")
        print(f"  Session:    {session.session_id}")
        print(f"  Profile:    {getattr(session.profile, 'name', 'Fallback')}")
        print(f"  BPM:        {getattr(session.profile, 'midi_bpm', 'N/A')}")
        print(f"  Tracks:     {len(session.tracks)}")
        print(f"  Stems:      {len(session.stems)}")
        print(f"  FINAL MIX:  {session.final_mix}")
        if session.registry_entry:
            print(f"  REGISTRY:   seq={session.registry_entry['seq']} "
                  f"hash={session.registry_entry['entry_hash'][:16]}…")
            print(f"  MANIFEST:   {session.final_mix.with_suffix('.manifest.json')}")
        print(f"{'='*60}")
    except Exception as e:
        logger.critical(f"Production failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
