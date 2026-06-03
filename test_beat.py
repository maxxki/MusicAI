#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from maxxki_beat_gen import (
    PROGRESSIONS, GENRE_SCALE_MAP, GENRE_DRUM_MAP,
    SCALES, add_chords, add_bass, add_melody, add_drums
)
from midiutil import MIDIFile

# ─── Config ───────────────────────────────────────────────────────────────────

BPM    = 100
KEY    = "D"
GENRE  = "dancehall"
BARS   = 8
SF2    = Path("/var/home/mk/projects/mk/music/VintageDreamsWaves-v2.sf2")

KEY_MIDI = {
    "C":60,"C#":61,"D":62,"D#":63,"E":64,
    "F":65,"F#":66,"G":67,"G#":68,"A":69,"A#":70,"B":71
}[KEY]

OUT_DIR = Path.home() / "projects/mk/music-output/midi"

# ─── MIDI ─────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
midi_path = OUT_DIR / "test_kalado.mid"
wav_path  = OUT_DIR / "test_kalado.wav"

midi = MIDIFile(4)
for t in range(4):
    midi.addTempo(t, 0, BPM)

scale       = SCALES[GENRE_SCALE_MAP[GENRE]]
progression = PROGRESSIONS[GENRE]
drum_pattern = GENRE_DRUM_MAP[GENRE]

add_chords(midi, 0, 0, progression, KEY_MIDI, scale, BARS, BPM, "block")
add_bass(  midi, 1, 1, progression, KEY_MIDI, scale, BARS, "root")
add_melody(midi, 2, 2, progression, KEY_MIDI, scale, BARS, 0.5)
add_drums( midi, 3, 9, drum_pattern, BARS, BPM)

with open(midi_path, "wb") as f:
    midi.writeFile(f)

print(f"✓ MIDI: {midi_path}")

# ─── FluidSynth Render ────────────────────────────────────────────────────────

if not SF2.exists():
    print(f"✗ SoundFont nicht gefunden: {SF2}")
    sys.exit(1)

try:
    result = subprocess.run([
        "fluidsynth",
        "-ni",
        "-g", "0.8",
        "-r", "44100",
        "-T", "wav",
        "-O", "s16",
        "-F", str(wav_path),
        str(SF2),
        str(midi_path),
    ], check=True, capture_output=True, text=True)

    print(f"✓ WAV:  {wav_path}")
    print(f"  Play: xdg-open '{wav_path}'")

except FileNotFoundError:
    print("✗ FluidSynth nicht gefunden.")
    print("  Install: sudo dnf install fluidsynth   # Fedora")
    print("           sudo apt install fluidsynth   # Debian/Ubuntu")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"✗ FluidSynth Fehler:\n{e.stderr}")
    sys.exit(1)
