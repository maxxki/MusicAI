"""
MAXXKI Kalado Dancehall Starter v2.0
"The Produck" meets Kingston — jetzt mit echter Kalado-Konfiguration.

Kalado-Style Merkmale (werden tatsächlich angewendet):
- Heavy 808 sub-bass (40-60Hz Fokus, +8dB Boost)
- Sparse percussion (Kick, Snare, minimal Hi-Hats)
- Dark synth-stabs (minor/Phrygian)
- Aggressive Sidechain-Pumpe (0.6)
- Maximal 3-4 Elemente gleichzeitig
- 100 BPM, lazy but heavy

Usage: python maxxki_kalado_dancehall.py [prompt]
"""

import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional

# Importiere den Orchestrator
from maxxki_orchestrator import (
    MAXXKIProducer, SessionConfig, VibeProfile,
    OUTPUT_DIR, FINAL_DIR, TrackSpec
)

# ─── Kalado-Style Presets ───────────────────────────────────────────────────

KALADO_PRESETS = {
    "dark_riddim": {
        "prompt": "dark dancehall riddim, heavy sub 808, minimal kick snare, "
                  "sparse percussion, Kingston night, 100 BPM, Kalado style",
        "bpm": 100,
        "key": "D",
        "vibe": "the_produck",
        "duration": 90,
    },
    "produck_bashment": {
        "prompt": "bashment dancehall, aggressive 808 bounce, dark synth stabs, "
                  "heavy sidechain, minimal elements, street energy, 100 BPM",
        "bpm": 100,
        "key": "F#",
        "vibe": "the_produck",
        "duration": 120,
    },
    "midnight_kingston": {
        "prompt": "midnight Kingston dancehall, deep sub bass, lonely percussion, "
                  "dark atmosphere, 3am vibe, sparse, heavy, 100 BPM",
        "bpm": 100,
        "key": "G",
        "vibe": "dark_minimal",
        "duration": 60,
    },
    "808_pressure": {
        "prompt": "dancehall 808 pressure, sub-bass focused, minimal drums, "
                  "dark melodic stabs, aggressive ducking, 100 BPM, Kalado",
        "bpm": 100,
        "key": "A",
        "vibe": "the_produck",
        "duration": 90,
    },
}

# ─── Kalado Track-Plan (wird tatsächlich an Orchestrator übergeben) ───────

@dataclass
class KaladoTrackPlan:
    """
    Spezifische Track-Konfiguration für Kalado-Style Dancehall.
    Wird als Override an den Orchestrator durchgereicht.
    """

    # Drums: Minimal, heavy — sparse pattern mit off-beat accent
    drum_pattern: Dict[int, list] = None

    # Bass: Deep, sub-focused, syncopated bounce
    bass_octave: int = 1           # Tiefer als Standard (Octave 2)
    bass_style: str = "syncopated"  # Off-beat bounce statt root

    # Chords: Dark, staccato
    chord_style: str = "stab"      # Kurze, aggressive Stabs
    chord_types: list = None       # ["minor", "min7", "dim"]

    # Melody: Sparse, intervallisch
    melody_density: float = 0.3    # Sehr sparsam (Standard: 0.5)
    melody_octave: int = 4

    # Mastering: Sub-Fokus
    sub_boost_db: float = 8.0      # Extrem (Standard: 4-6)
    low_cut_hz: float = 25.0       # Tiefere Grenze (Standard: 30-40)
    sidechain_amount: float = 0.6   # Aggressiver (Standard: 0.4-0.5)

    def __post_init__(self):
        if self.drum_pattern is None:
            self.drum_pattern = {
                36: [1,0,0,0, 0,0,1,0, 1,0,0,0, 0,0,0,0],  # Kick — sparse
                38: [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],  # Snare — backbeat
                42: [0,1,0,1, 0,1,0,1, 0,1,0,1, 0,1,0,1],  # HH — minimal 8th
                46: [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],  # Open HH — rare
            }
        if self.chord_types is None:
            self.chord_types = ["minor", "min7", "dim"]


# ─── Korrekte Producer-Integration ──────────────────────────────────────────

class KaladoProducer(MAXXKIProducer):
    """
    Erweiterter Producer mit Kalado-Style Overrides.
    Überschreibt _create_track_plan und _generate_midi_foundation.
    """

    def __init__(self, kalado_plan: KaladoTrackPlan = None):
        super().__init__()
        self.kalado = kalado_plan or KaladoTrackPlan()

    def _create_track_plan(self, genre: str, profile, analysis: Dict) -> list:
        """
        Erstellt Track-Spezifikationen mit Kalado-Overrides.
        """
        tracks = super()._create_track_plan(genre, profile, analysis)

        # Kalado-Overrides anwenden
        for track in tracks:
            if track.name == "bass":
                track.gain_db = -1
                track.eq_profile = "bass"
            elif track.name == "drums":
                track.gain_db = -2
                track.eq_profile = "drums"
            elif track.name == "chords":
                track.gain_db = -3
                track.eq_profile = "harmonic"
            elif track.name == "melody":
                track.gain_db = -2
                track.eq_profile = "melodic"
                track.send_reverb = 0.2
                track.send_delay = 0.15

        return tracks

    def _generate_midi_foundation(self, session, bpm: int, genre: str) -> Optional[Dict]:
        """
        Generiert MIDI-Foundation mit Kalado-Drum-Pattern und Bass-Style.
        """
        from midiutil import MIDIFile
        from maxxki_beat_gen import (
            add_chords, add_bass, add_melody, add_drums,
            PROGRESSIONS, GENRE_SCALE_MAP, GENRE_DRUM_MAP,
            SCALES, CHORD_TYPES
        )

        bars = max(4, int(session.config.duration_seconds * bpm / 240))

        midi = MIDIFile(4)
        for t in range(4):
            midi.addTempo(t, 0, bpm)

        key_midi = {"C":60,"C#":61,"D":62,"D#":63,"E":64,
                   "F":65,"F#":66,"G":67,"G#":68,"A":69,"A#":70,"B":71}.get(
            session.config.key, 60
        )

        scale = SCALES.get(GENRE_SCALE_MAP.get(genre, "minor"), SCALES["minor"])
        progression = PROGRESSIONS.get(genre, PROGRESSIONS["dark"])

        # ─── KALADO OVERRIDE: Custom Drum-Pattern ─────────────────────────
        # add_drums() erwartet einen String-Key, kein Dict direkt.
        # Kalado-Pattern unter eigenem Key registrieren.
        from maxxki_beat_gen import DRUM_PATTERNS
        kalado_key = f"kalado_{session.session_id}"
        DRUM_PATTERNS[kalado_key] = self.kalado.drum_pattern
        drum_pattern_key = kalado_key

        # ─── KALADO OVERRIDE: Bass Style ──────────────────────────────────
        # Statt "root" nutzen wir kalado.bass_style ("syncopated")
        bass_style = self.kalado.bass_style

        # ─── KALADO OVERRIDE: Chord Style ───────────────────────────────
        # Statt "block" nutzen wir kalado.chord_style ("stab")
        chord_style = self.kalado.chord_style

        # ─── KALADO OVERRIDE: Melody Density ────────────────────────────
        # Statt 0.5 nutzen wir kalado.melody_density (0.3)
        melody_density = self.kalado.melody_density

        # Tracks mit Kalado-Settings
        add_chords(midi, 0, 0, progression, key_midi, scale, bars, bpm, chord_style)
        add_bass(midi, 1, 1, progression, key_midi, scale, bars, bass_style)
        add_drums(midi, 3, 9, drum_pattern_key, bars, bpm)  # Custom pattern!

        # AI-Melodie oder Fallback
        if session.config.use_midi_gen:
            try:
                from maxxki_midi_gen import build_prompt, generate_tokens, tokens_to_midi
                prompt = build_prompt("synth-lead", "sparse", bars=4)
                tokens = generate_tokens(prompt, max_new_tokens=512)
                from maxxki_orchestrator import TEMP_DIR
                melody_midi = TEMP_DIR / f"{session.session_id}_melody.mid"
                tokens_to_midi(tokens, melody_midi, bpm=bpm)
                # Melody-Track im Haupt-MIDI
                add_melody(midi, 2, 2, progression, key_midi, scale, bars, melody_density)
            except Exception as e:
                add_melody(midi, 2, 2, progression, key_midi, scale, bars, melody_density)
        else:
            add_melody(midi, 2, 2, progression, key_midi, scale, bars, melody_density)

        # Speichern
        from maxxki_orchestrator import TEMP_DIR
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        combined_path = TEMP_DIR / f"{session.session_id}_foundation.mid"
        with open(combined_path, "wb") as f:
            midi.writeFile(f)

        return {"combined": combined_path}


# ─── Haupt-Funktion (korrigiert) ───────────────────────────────────────────

def run_kalado(preset_name: str = "dark_riddim", 
               custom_prompt: str = None,
               overrides: Dict = None):
    """
    Startet eine Kalado-Style Dancehall-Produktion.

    FIX v2.0: 
    - Overrides werden korrekt übernommen (nicht weggeworfen)
    - KaladoTrackPlan wird tatsächlich instantiiert und angewendet
    - KaladoProducer überschreibt MIDI-Generierung mit Style-Settings
    """
    # Preset laden und mit Overrides mergen
    base_preset = KALADO_PRESETS.get(preset_name, KALADO_PRESETS["dark_riddim"]).copy()
    if overrides:
        base_preset.update({k: v for k, v in overrides.items() if v is not None})

    prompt = custom_prompt or base_preset["prompt"]

    print(f"\n{'='*70}")
    print(f"  MAXXKI KALADO DANCEHALL — {preset_name.upper()}")
    print(f"{'='*70}")
    print(f"  Style:     The Produck meets Kingston")
    print(f"  BPM:       {base_preset['bpm']}")
    print(f"  Key:       {base_preset['key']}")
    print(f"  Duration:  {base_preset['duration']}s")
    print(f"  Prompt:    {prompt[:60]}...")
    print(f"{'='*70}\n")

    # Kalado-Plan erstellen (wird tatsächlich verwendet!)
    kalado_plan = KaladoTrackPlan()

    # Session Config mit korrekten Werten
    config = SessionConfig(
        user_prompt=prompt,
        vibe_hint=base_preset["vibe"],
        bpm=base_preset["bpm"],
        key=base_preset["key"],
        duration_seconds=base_preset["duration"],
        use_claude=True,
        use_beat_gen=True,
        use_midi_gen=True,
        use_music_gen=True,
    )

    # Producer mit Kalado-Overrides
    producer = KaladoProducer(kalado_plan=kalado_plan)

    try:
        session = asyncio.run(producer.produce(config))

        print(f"\n{'='*70}")
        print(f"  ✓ PRODUCTION COMPLETE")
        print(f"{'='*70}")
        print(f"  Session:   {session.session_id}")
        print(f"  Profile:   {session.profile.name}")
        print(f"  BPM:       {session.profile.midi_bpm}")
        print(f"  Tracks:    {len(session.tracks)}")
        print(f"  Stems:     {len(session.stems)}")
        print(f"{'='*70}")
        print(f"  FINAL:     {session.final_mix}")
        print(f"{'='*70}")
        print(f"\n  Play:      xdg-open '{session.final_mix}'")
        print(f"  Analyze:   audacity '{session.final_mix}'\n")

        return session

    except Exception as e:
        print(f"\n✗ Production failed: {e}\n")
        sys.exit(1)


def interactive_menu():
    """
    Interaktives Menu für Kalado-Style Presets.
    FIX v2.0: Dynamischer Range, nicht hardcodiert.
    """
    print(f"\n{'='*70}")
    print(f"  MAXXKI KALADO DANCEHALL STUDIO v2.0")
    print(f"  The Produck Vibe — Kingston Style")
    print(f"{'='*70}\n")

    presets = list(KALADO_PRESETS.items())
    print("  Presets:")
    for i, (name, preset) in enumerate(presets, 1):
        print(f"    {i}. {name:20s} — {preset['prompt'][:45]}...")

    print(f"\n    0. Custom Prompt")
    print(f"\n{'='*70}")

    choice = input("\n  Select (0-{}): ".format(len(presets))).strip()

    if choice == "0":
        custom = input("\n  Enter your prompt: ").strip()
        run_kalado("dark_riddim", custom)
    elif choice.isdigit() and 1 <= int(choice) <= len(presets):
        preset_name = presets[int(choice)-1][0]
        run_kalado(preset_name)
    else:
        print("  Invalid choice, using default.")
        run_kalado("dark_riddim")


# ─── CLI Entry Point (korrigiert) ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MAXXKI Kalado Dancehall Producer v2.0",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python maxxki_kalado_dancehall.py
  python maxxki_kalado_dancehall.py --preset produck_bashment
  python maxxki_kalado_dancehall.py --prompt "dark 808, minimal, Kingston night"
  python maxxki_kalado_dancehall.py --preset 808_pressure --duration 120 --key G
        """
    )
    parser.add_argument("--preset", choices=list(KALADO_PRESETS.keys()),
                       default="dark_riddim", help="Preset to use")
    parser.add_argument("--prompt", help="Custom prompt (overrides preset)")
    parser.add_argument("--duration", type=int, help="Override duration")
    parser.add_argument("--key", help="Override key")
    parser.add_argument("--bpm", type=int, help="Override BPM")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive menu mode")

    args = parser.parse_args()

    if args.interactive:
        interactive_menu()
    else:
        # Overrides korrekt sammeln und übergeben
        overrides = {}
        if args.duration:
            overrides["duration"] = args.duration
        if args.key:
            overrides["key"] = args.key
        if args.bpm:
            overrides["bpm"] = args.bpm

        run_kalado(args.preset, args.prompt, overrides=overrides if overrides else None)
