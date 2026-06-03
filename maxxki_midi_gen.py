"""
MAXXKI MIDI Generator — GPT-2 Music Model v1.1
Generiert MIDI-Dateien für LMMS, ZenBeats etc.
Model: ai-guru/lakhclean_mmmtrack_4bars_d-2048

Fixes v1.1:
- Multi-Track: Parser bricht nicht mehr beim ersten TRACK_END ab,
  alle Tracks werden vollständig geparst (bis PIECE_END oder Token-Ende)
- Channel-Zuweisung: Jeder Track bekommt seinen eigenen Channel (0,1,2,…),
  nicht mehr alles auf CH0
- BAR_END snap: Entfernt — hat Timing-Drift verursacht wenn TIME_DELTAs
  nicht sauber auf Bar-Grenzen aufgingen
- MIDIFile wird mit korrekter Track-Anzahl initialisiert (nach Scan)
- Offene Noten werden pro Track korrekt geschlossen
"""

import logging
import sys
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MAXXKI-MIDI")

MODEL_ID   = "ai-guru/lakhclean_mmmtrack_4bars_d-2048"
OUTPUT_DIR = Path.home() / "projects/mk/music-output/midi"

# Token-Format des Modells:
# PIECE_START — Anfang
# TRACK_START — neuer Track
# INST=<n>    — Instrument (GM-Nummer)
# DENSITY=<n> — Notendichte 0-5
# BAR_START / BAR_END — Taktgrenzen
# NOTE_ON=<pitch>  — Note an (0-127)
# NOTE_OFF=<pitch> — Note aus
# TIME_DELTA=<n>   — Zeitschritt in 16th notes

GM_INSTRUMENTS = {
    "piano":      0,
    "bass":       32,
    "e-bass":     33,
    "synth-bass": 38,
    "strings":    48,
    "synth-pad":  88,
    "synth-lead": 80,
    "drums":      -1,  # Channel 9
}

DENSITY_MAP = {
    "sparse": 1,
    "normal": 3,
    "dense":  5,
}


def build_prompt(instrument: str = "piano", density: str = "normal", bars: int = 4) -> str:
    """Baut einen validen Prompt für das Modell."""
    inst_num = GM_INSTRUMENTS.get(instrument, 0)
    dens_num = DENSITY_MAP.get(density, 3)

    if inst_num == -1:
        prompt = f"PIECE_START TRACK_START INST=DRUMS DENSITY={dens_num} BAR_START"
    else:
        prompt = f"PIECE_START TRACK_START INST={inst_num} DENSITY={dens_num} BAR_START"

    return prompt


def generate_tokens(prompt: str, max_new_tokens: int = 512) -> str:
    """Generiert Token-Sequenz mit GPT-2."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    logger.info(f"Lade Modell '{MODEL_ID}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    logger.info("Modell geladen.")

    inputs = tokenizer(prompt, return_tensors="pt")

    logger.info(f"Generiere MIDI-Tokens (max {max_new_tokens} tokens)...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Generiert: {len(generated.split())} Tokens")
    return generated


def tokens_to_midi(token_string: str, output_path: Path, bpm: int = 120) -> None:
    """
    Konvertiert Token-String zu MIDI-Datei.

    FIX v1.1:
    - Vollständiger Multi-Track-Parser: alle TRACK_START…TRACK_END Blöcke
      werden geparst, nicht nur der erste.
    - Jeder Track bekommt Channel track_idx % 9 (Channel 9 für Drums übersprungen,
      außer wenn INST=DRUMS explizit gesetzt).
    - BAR_END snap entfernt (war Quelle von Timing-Drift).
    - Pro-Track active_notes Dict → kein Cross-Track Pitch-Konflikt.
    """
    from midiutil import MIDIFile

    tokens = token_string.split()

    # ── Schritt 1: Tracks vorab zählen für MIDIFile-Initialisierung ──────────
    track_count = max(1, token_string.count("TRACK_START"))
    midi = MIDIFile(track_count)

    duration_16th = 0.25  # 16th note in beats (bei 4/4)

    # ── Schritt 2: Multi-Track-Parse ─────────────────────────────────────────
    current_track   = -1
    current_channel = 0
    current_time    = 0.0
    active_notes: dict[int, float] = {}   # pitch → start_time, pro Track zurückgesetzt

    for i, token in enumerate(tokens):

        if token == "PIECE_START":
            continue

        elif token == "TRACK_START":
            # Vorherigen Track abschließen falls noch offene Noten
            if current_track >= 0 and active_notes:
                for pitch, start in active_notes.items():
                    dur = max(current_time - start, 0.25)
                    midi.addNote(current_track, current_channel, pitch, start, dur, 80)
                active_notes = {}

            current_track += 1
            if current_track >= track_count:
                break   # Mehr Tracks als gezählt — sicher abbrechen

            current_time    = 0.0
            active_notes    = {}
            # Standard-Channel: track_idx, aber Channel 9 reserviert für Drums
            # → einfaches Mapping: track 0→CH0, 1→CH1, … 8→CH8, 9→CH10 (skip 9)
            ch = current_track
            if ch >= 9:
                ch += 1   # CH9 überspringen, GM-Drums-Channel freihalten
            current_channel = ch

            midi.addTempo(current_track, 0, bpm)

        elif token.startswith("INST="):
            inst_str = token.split("=")[1]
            if inst_str == "DRUMS":
                current_channel = 9   # GM Drums immer auf CH9

        elif token == "BAR_START":
            pass   # Nur strukturell, kein Timing-Einfluss

        elif token == "BAR_END":
            pass   # FIX: snap entfernt — kein Timing-Drift mehr

        elif token.startswith("NOTE_ON="):
            try:
                pitch = int(token.split("=")[1])
                if 0 <= pitch <= 127:
                    active_notes[pitch] = current_time
            except ValueError:
                pass

        elif token.startswith("NOTE_OFF="):
            try:
                pitch = int(token.split("=")[1])
                if pitch in active_notes:
                    start        = active_notes.pop(pitch)
                    note_duration = max(current_time - start, 0.25)
                    midi.addNote(current_track, current_channel, pitch,
                                 start, note_duration, 90)
            except ValueError:
                pass

        elif token.startswith("TIME_DELTA="):
            try:
                delta = int(token.split("=")[1])
                current_time += delta * duration_16th
            except ValueError:
                pass

        elif token == "TRACK_END":
            # Offene Noten des aktuellen Tracks schließen
            for pitch, start in active_notes.items():
                dur = max(current_time - start, 0.25)
                midi.addNote(current_track, current_channel, pitch, start, dur, 80)
            active_notes = {}
            # KEIN break — nächster TRACK_START wird weitergeparst

        elif token == "PIECE_END":
            # Alles abschließen
            for pitch, start in active_notes.items():
                dur = max(current_time - start, 0.25)
                midi.addNote(current_track, current_channel, pitch, start, dur, 80)
            active_notes = {}
            break

    # Letzten Track abschließen falls PIECE_END gefehlt hat
    if active_notes and current_track >= 0:
        for pitch, start in active_notes.items():
            dur = max(current_time - start, 0.25)
            midi.addNote(current_track, current_channel, pitch, start, dur, 80)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        midi.writeFile(f)

    logger.info(f"MIDI gespeichert: {output_path} ({track_count} Track(s))")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MAXXKI MIDI Generator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--instrument", type=str, default="piano",
                        choices=list(GM_INSTRUMENTS.keys()),
                        help="Instrument (default: piano)")
    parser.add_argument("--density", type=str, default="normal",
                        choices=["sparse", "normal", "dense"],
                        help="Notendichte (default: normal)")
    parser.add_argument("--bpm", type=int, default=120,
                        help="BPM (default: 120)")
    parser.add_argument("--tokens", type=int, default=512,
                        help="Max generierte Tokens (default: 512)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output-Pfad (default: ~/projects/mk/music-output/midi/)")
    return parser.parse_args()


def main():
    args = parse_args()

    prompt = build_prompt(
        instrument=args.instrument,
        density=args.density,
    )
    logger.info(f"Prompt: {prompt}")

    token_string = generate_tokens(prompt, max_new_tokens=args.tokens)

    if args.output:
        out_path = Path(args.output)
    else:
        safe_name = f"{args.instrument}_{args.density}_{args.bpm}bpm.mid"
        out_path  = OUTPUT_DIR / safe_name

    tokens_to_midi(token_string, out_path, bpm=args.bpm)

    print(f"\n[✓] Fertig! → {out_path}")
    print(f"    In LMMS öffnen: xdg-open '{out_path}'\n")


if __name__ == "__main__":
    main()
