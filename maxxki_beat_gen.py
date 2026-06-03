"""
MAXXKI Beat Generator
Interaktiver Multi-Track MIDI Generator für LMMS / ZenBeats

Tracks:
  0 — Chords      (Piano / Pad)
  1 — Bass        (syncopated root notes)
  2 — Melody      (scale-based, humanized)
  3 — Drums       (CH9, pattern-based)

Output: ~/projects/mk/music-output/midi/<name>.mid
"""

import random
import sys
from pathlib import Path
from midiutil import MIDIFile

OUTPUT_DIR = Path.home() / "projects/mk/music-output/midi"

# ─── Musik-Theorie ────────────────────────────────────────────────────────────

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

KEY_MIDI = {
    "C": 60, "C#": 61, "D": 62, "D#": 63, "E": 64,
    "F": 65, "F#": 66, "G": 67, "G#": 68, "A": 69,
    "A#": 70, "B": 71,
}

# Skalenmuster (Halbtonschritte vom Root)
SCALES = {
    "major":       [0, 2, 4, 5, 7, 9, 11],
    "minor":       [0, 2, 3, 5, 7, 8, 10],
    "dorian":      [0, 2, 3, 5, 7, 9, 10],
    "phrygian":    [0, 1, 3, 5, 7, 8, 10],
    "mixolydian":  [0, 2, 4, 5, 7, 9, 10],
    "pentatonic":  [0, 2, 4, 7, 9],
    "blues":       [0, 3, 5, 6, 7, 10],
}

# Chord-Typen: (Terz, Quinte [, Sept])
CHORD_TYPES = {
    "major":  [0, 4, 7],
    "minor":  [0, 3, 7],
    "maj7":   [0, 4, 7, 11],
    "min7":   [0, 3, 7, 10],
    "dom7":   [0, 4, 7, 10],
    "sus2":   [0, 2, 7],
    "sus4":   [0, 5, 7],
    "dim":    [0, 3, 6],
}

# Progressionen: Liste von (Skalenstufe, Chord-Typ)
PROGRESSIONS = {
    "pop":       [(0,"major"), (5,"major"), (2,"minor"), (4,"major")],
    "sad":       [(0,"minor"), (6,"major"), (3,"minor"), (4,"major")],
    "jazz":      [(0,"maj7"),  (5,"dom7"),  (1,"min7"),  (4,"dom7")],
    "blues":     [(0,"dom7"),  (0,"dom7"),  (5,"dom7"),  (0,"dom7")],
    "lofi":      [(0,"maj7"),  (4,"min7"),  (2,"min7"),  (5,"dom7")],
    "dark":      [(0,"minor"), (7,"major"), (6,"major"), (4,"minor")],
    "euphoric":  [(0,"major"), (4,"major"), (5,"major"), (3,"minor")],
    "drill":     [(0,"minor"), (6,"major"), (7,"major"), (4,"minor")],
    "dancehall": [(0,"major"), (5,"major"), (3,"minor"), (4,"major")],
    "techno":    [(0,"minor"), (0,"minor"), (7,"minor"), (6,"major")],
}

# Drum-Pattern (16 steps, 1=hit)
# GM Drum-Noten: Kick=36, Snare=38, Closed HH=42, Open HH=46, Clap=39
DRUM_PATTERNS = {
    "basic": {
        36: [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],  # Kick
        38: [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],  # Snare
        42: [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],  # HH closed
    },
    "trap": {
        36: [1,0,0,0, 0,0,1,0, 0,0,0,0, 1,0,0,0],
        38: [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,1,0],
        42: [1,1,0,1, 1,0,1,1, 0,1,1,0, 1,1,0,1],  # HH rolls
        46: [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],  # Open HH
    },
    "lofi": {
        36: [1,0,0,0, 0,0,1,0, 1,0,0,0, 0,0,0,0],
        38: [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        42: [1,0,1,0, 0,0,1,0, 1,0,1,0, 0,0,1,0],
    },
    "dancehall": {
        36: [1,0,0,1, 0,0,1,0, 1,0,0,0, 0,1,0,0],
        38: [0,0,0,0, 1,0,0,0, 0,0,1,0, 1,0,0,0],
        39: [0,0,1,0, 0,0,0,1, 0,0,1,0, 0,0,0,1],  # Clap
        42: [0,1,0,1, 0,1,0,1, 0,1,0,1, 0,1,0,1],
    },
    "techno": {
        36: [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
        38: [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        42: [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],  # 16th HH
    },
    "afrobeats": {
        36: [1,0,0,0, 0,1,0,0, 1,0,0,0, 0,0,1,0],
        38: [0,0,1,0, 1,0,0,0, 0,0,1,0, 1,0,0,0],
        42: [1,0,1,1, 0,1,1,0, 1,0,1,1, 0,1,1,0],
    },
}

GENRE_DRUM_MAP = {
    "pop": "basic", "sad": "lofi", "jazz": "lofi",
    "blues": "basic", "lofi": "lofi", "dark": "trap",
    "euphoric": "basic", "drill": "trap", "dancehall": "dancehall",
    "techno": "techno", "afrobeats": "afrobeats",
}

GENRE_SCALE_MAP = {
    "pop": "major", "sad": "minor", "jazz": "major",
    "blues": "blues", "lofi": "major", "dark": "phrygian",
    "euphoric": "major", "drill": "minor", "dancehall": "major",
    "techno": "dorian", "afrobeats": "pentatonic",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def humanize(value: int, amount: int = 8) -> int:
    """Leichte Velocity-Variation für organischen Sound."""
    return max(40, min(127, value + random.randint(-amount, amount)))


def scale_notes(root_midi: int, scale: list[int], octave: int = 4) -> list[int]:
    """Gibt absolute MIDI-Noten der Skala zurück."""
    base = (root_midi % 12) + (octave * 12)
    return [base + interval for interval in scale]


def chord_notes(root_midi: int, chord_type: list[int], octave: int = 4) -> list[int]:
    """Gibt Akkord-Noten zurück."""
    base = (root_midi % 12) + (octave * 12)
    return [base + interval for interval in chord_type]


def progression_to_roots(key_midi: int, progression: list, scale: list[int]) -> list[int]:
    """Wandelt Skalenstufen in absolute MIDI-Roots um."""
    roots = []
    for degree, _ in progression:
        idx = degree % len(scale)
        roots.append((key_midi % 12) + scale[idx] + 48)  # Octave 3 für Chords
    return roots


# ─── Track-Builder ────────────────────────────────────────────────────────────

def add_chords(midi: MIDIFile, track: int, channel: int,
               progression: list, key_midi: int, scale: list[int],
               bars: int, bpm: int, style: str = "block") -> None:
    """
    Fügt Chord-Track hinzu.
    style: 'block' = ganzer Takt, 'arp' = arpeggiert, 'stab' = kurze Stabs
    """
    beat_per_bar = 4
    chord_octave = 4

    for bar in range(bars):
        prog_idx = bar % len(progression)
        degree, chord_type_name = progression[prog_idx]
        chord_type = CHORD_TYPES[chord_type_name]

        idx = degree % len(scale)
        root = (key_midi % 12) + scale[idx] + (chord_octave * 12)
        notes = chord_notes(root, chord_type, octave=0)  # schon absolut

        bar_start = bar * beat_per_bar

        if style == "block":
            for note in notes:
                vel = humanize(72, 10)
                midi.addNote(track, channel, note, bar_start, beat_per_bar - 0.1, vel)

        elif style == "arp":
            step = beat_per_bar / len(notes)
            for i, note in enumerate(notes):
                vel = humanize(80, 12)
                midi.addNote(track, channel, note, bar_start + i * step, step * 0.8, vel)

        elif style == "stab":
            for note in notes:
                vel = humanize(90, 8)
                midi.addNote(track, channel, note, bar_start, 0.5, vel)
                midi.addNote(track, channel, note, bar_start + 2.0, 0.5, vel)


def add_bass(midi: MIDIFile, track: int, channel: int,
             progression: list, key_midi: int, scale: list[int],
             bars: int, style: str = "root") -> None:
    """
    Bass-Track.
    style: 'root' = nur Root, 'walking' = Root + Quinte, 'syncopated' = off-beat
    """
    beat_per_bar = 4
    bass_octave = 2

    for bar in range(bars):
        prog_idx = bar % len(progression)
        degree, chord_type_name = progression[prog_idx]

        idx = degree % len(scale)
        root = (key_midi % 12) + scale[idx] + (bass_octave * 12)
        fifth = root + 7

        bar_start = bar * beat_per_bar

        if style == "root":
            vel = humanize(85, 8)
            midi.addNote(track, channel, root, bar_start, 1.8, vel)
            midi.addNote(track, channel, root, bar_start + 2.0, 1.8, vel)

        elif style == "walking":
            for beat, note in enumerate([root, fifth, root, fifth]):
                vel = humanize(80, 10)
                midi.addNote(track, channel, note, bar_start + beat, 0.85, vel)

        elif style == "syncopated":
            hits = [0.0, 0.75, 2.0, 2.5, 3.5]
            for hit in hits:
                vel = humanize(88, 10)
                dur = 0.4 if hit % 1 != 0 else 0.7
                midi.addNote(track, channel, root, bar_start + hit, dur, vel)


def add_melody(midi: MIDIFile, track: int, channel: int,
               progression: list, key_midi: int, scale: list[int],
               bars: int, density: float = 0.5) -> None:
    """
    Melody-Track — scale-basiert, melodische Bögen, Pausen.
    density: 0.0–1.0, wie viele Steps eine Note haben
    """
    beat_per_bar = 4
    mel_octave = 5
    scale_abs = scale_notes(key_midi, scale, mel_octave)

    # Melodische Bewegung: bevorzuge Schritte, nicht Sprünge
    prev_idx = random.randint(0, len(scale_abs) - 1)

    for bar in range(bars):
        bar_start = bar * beat_per_bar
        # 8th notes pro Takt
        for step in range(8):
            if random.random() > density:
                continue  # Pause

            # Schritt-bevorzugend: ±1-2 Skalentöne
            move = random.choice([-2, -1, -1, 0, 1, 1, 2])
            prev_idx = max(0, min(len(scale_abs) - 1, prev_idx + move))
            note = scale_abs[prev_idx]

            beat_pos = bar_start + step * 0.5
            dur = random.choice([0.4, 0.5, 0.9, 1.4])
            vel = humanize(75, 15)

            midi.addNote(track, channel, note, beat_pos, dur, vel)


def add_drums(midi: MIDIFile, track: int, channel: int,
              pattern_name: str, bars: int, bpm: int) -> None:
    """Drum-Track auf Channel 9."""
    pattern = DRUM_PATTERNS.get(pattern_name, DRUM_PATTERNS["basic"])
    beat_per_bar = 4
    step_dur = beat_per_bar / 16  # 16th notes

    for bar in range(bars):
        bar_start = bar * beat_per_bar
        for drum_note, steps in pattern.items():
            for step, hit in enumerate(steps):
                if hit:
                    t = bar_start + step * step_dur
                    vel = humanize(90, 15) if drum_note != 42 else humanize(70, 10)
                    midi.addNote(track, channel, drum_note, t, step_dur * 0.9, vel)


# ─── Interactive CLI ──────────────────────────────────────────────────────────

def ask(prompt: str, options: list[str] | None = None, default: str = "") -> str:
    if options:
        opts_str = " / ".join(f"[{o}]" for o in options)
        print(f"  {prompt} {opts_str}")
        if default:
            print(f"  (Enter = {default})")
    else:
        print(f"  {prompt}" + (f" (Enter = {default})" if default else ""))

    while True:
        val = input("  → ").strip()
        if not val and default:
            return default
        if options and val.lower() not in [o.lower() for o in options]:
            print(f"  ⚠ Ungültig. Optionen: {', '.join(options)}")
            continue
        return val.lower() if val else default


def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   MAXXKI Beat Generator              ║")
    print("║   Multi-Track MIDI für LMMS/ZenBeats ║")
    print("╚══════════════════════════════════════╝\n")

    # ── Genre
    genres = list(PROGRESSIONS.keys())
    print(f"  Genres: {', '.join(genres)}")
    genre = ask("Genre:", genres, "lofi")

    # ── Key
    print(f"  Keys: {', '.join(NOTES)}")
    key_input = ask("Tonart:", NOTES, "C").upper()
    key_midi = KEY_MIDI.get(key_input, 60)

    # ── BPM
    bpm_input = ask("BPM:", default="90")
    try:
        bpm = int(bpm_input)
    except ValueError:
        bpm = 90

    # ── Bars
    bars_input = ask("Takte:", default="16")
    try:
        bars = int(bars_input)
    except ValueError:
        bars = 16

    # ── Chord Style
    chord_style = ask("Chord-Stil:", ["block", "arp", "stab"], "block")

    # ── Bass Style
    bass_style = ask("Bass-Stil:", ["root", "walking", "syncopated"], "root")

    # ── Melody Dichte
    density_input = ask("Melody-Dichte (0.0–1.0):", default="0.5")
    try:
        density = float(density_input)
    except ValueError:
        density = 0.5

    # ── Dateiname
    default_name = f"{genre}_{key_input}_{bpm}bpm_{bars}bars"
    name_input = ask(f"Dateiname:", default=default_name)
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name_input)

    print("\n  Generating...\n")

    # ── MIDI aufbauen
    midi = MIDIFile(4)  # 4 Tracks

    for t in range(4):
        midi.addTempo(t, 0, bpm)

    scale_name = GENRE_SCALE_MAP.get(genre, "minor")
    scale = SCALES[scale_name]
    progression = PROGRESSIONS[genre]
    drum_pattern = GENRE_DRUM_MAP.get(genre, "basic")

    # Track 0: Chords (CH 0)
    add_chords(midi, track=0, channel=0,
               progression=progression, key_midi=key_midi,
               scale=scale, bars=bars, bpm=bpm, style=chord_style)

    # Track 1: Bass (CH 1)
    add_bass(midi, track=1, channel=1,
             progression=progression, key_midi=key_midi,
             scale=scale, bars=bars, style=bass_style)

    # Track 2: Melody (CH 2)
    add_melody(midi, track=2, channel=2,
               progression=progression, key_midi=key_midi,
               scale=scale, bars=bars, density=density)

    # Track 3: Drums (CH 9)
    add_drums(midi, track=3, channel=9,
              pattern_name=drum_pattern, bars=bars, bpm=bpm)

    # ── Speichern
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{safe_name}.mid"

    with open(out_path, "wb") as f:
        midi.writeFile(f)

    print(f"╔══════════════════════════════════════╗")
    print(f"║  ✓ Fertig!                           ║")
    print(f"╠══════════════════════════════════════╣")
    print(f"║  Genre:    {genre:<27}║")
    print(f"║  Key:      {key_input:<27}║")
    print(f"║  Scale:    {scale_name:<27}║")
    print(f"║  BPM:      {bpm:<27}║")
    print(f"║  Takte:    {bars:<27}║")
    print(f"║  Drums:    {drum_pattern:<27}║")
    print(f"╠══════════════════════════════════════╣")
    print(f"║  Tracks:                             ║")
    print(f"║    CH 0 → Chords  (Piano/Pad)        ║")
    print(f"║    CH 1 → Bass    (Synth Bass)        ║")
    print(f"║    CH 2 → Melody  (Lead Synth)        ║")
    print(f"║    CH 9 → Drums   (GM Drums)          ║")
    print(f"╠══════════════════════════════════════╣")
    print(f"║  {str(out_path):<38}║")
    print(f"╚══════════════════════════════════════╝")
    print(f"\n  xdg-open '{out_path}'\n")


if __name__ == "__main__":
    main()

# Alias für Orchestrator-Import-Kompatibilität
chord_notes_absolute = chord_notes
