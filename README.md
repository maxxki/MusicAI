# ⬡ MAXXKI — Local AI Music Production Suite

> MusicGen generiert. GPT-2 analysiert. FluidSynth rendert. 100% lokal, kein API-Key.

MAXXKI ist ein modulares, vollständig lokal laufendes Music-Production-System.
Es kombiniert Text-to-Audio-Generierung (MusicGen), GPT-2-basierte Mastering-Analyse,
algorithmische MIDI-Komposition, Audio-Fingerprinting und eine append-only Registry
für Provenance-Tracking. Kein Cloud-Audio. Kein externer API-Key.

---

## Module-Übersicht

| Datei | Funktion |
|---|---|
| `maxxki_music_gen.py` | MusicGen-Wrapper — Text-to-Audio, Singleton, CPU/CUDA/MPS |
| `maxxki_beat_gen.py` | Algorithmischer 4-Track MIDI-Generator (Chords, Bass, Melody, Drums) |
| `maxxki_midi_gen.py` | GPT-2 MIDI-Tokenizer + Multi-Track-Parser |
| `maxxki_master_ear.py` | GPT-2 Mastering-Analyse + scipy Post-Processing Pipeline + Streamlit UI |
| `maxxki_orchestrator.py` | Session-Orchestrator — verbindet alle Module zu vollständiger Produktion |
| `maxxki_kalado_dancehall.py` | Kalado-Style Dancehall Preset mit eigenem KaladoProducer |
| `maxxki_fingerprint.py` | SHA-256 + HMAC-Signatur für Audio-Manifeste |
| `maxxki_registry.py` | Append-only Hash-Chain Registry für alle Produktionen |
| `maxxki_sign_session.py` | Wrapper: nach jeder Produktion automatisch signieren + registrieren |
| `maxxki_verify.py` | Echtheitsprüfung: Datei-Hash, Signatur, Chain-Integrität |
| `maxxki_session.py` | Batch-Generierung — Modell einmal laden, mehrere Tracks ausgeben |
| `test_beat.py` | Schnelltest: Dancehall MIDI generieren + FluidSynth-Render |

---

## Installation

```bash
pip install torch transformers midiutil scipy numpy soundfile

# FluidSynth (optional, für WAV-Rendering aus MIDI)
sudo apt-get install fluidsynth    # Debian/Ubuntu
sudo dnf install fluidsynth        # Fedora
brew install fluid-synth           # macOS
```

**SoundFont:** `VintageDreamsWaves-v2.sf2` muss unter
`/var/home/mk/projects/mk/music/VintageDreamsWaves-v2.sf2` liegen
(oder ein anderer SF2 in `SOUNDFONT_PATHS` in `maxxki_orchestrator.py`).

---

## Quick Start

### Text-to-Audio (MusicGen)

```bash
# Einzelner Track
python maxxki_music_gen.py "dark dancehall riddim, heavy 808 sub bass, 100 BPM"
python maxxki_music_gen.py lofi --duration 30

# Alle Genre-Presets anzeigen
python maxxki_music_gen.py --list-presets

# Mehrere Tracks ohne Modell-Reload
python maxxki_session.py
```

### Algorithmischer MIDI-Beat

```bash
python maxxki_beat_gen.py
# Interaktives CLI: Genre → Key → BPM → Takte → Chord/Bass/Melody-Stil → Dateiname
```

### Vollständige Produktion (Orchestrator)

```bash
python maxxki_orchestrator.py "dark warehouse 808, minimal percussion, 3am vibe"
python maxxki_orchestrator.py "lofi jazz, mellow drums" --bpm 85 --key F --duration 90
python maxxki_orchestrator.py "drill beat" --vibe industrial_drill --no-music-gen
```

### Kalado Dancehall

```bash
python maxxki_kalado_dancehall.py                          # interaktives Menu
python maxxki_kalado_dancehall.py --preset produck_bashment
python maxxki_kalado_dancehall.py --prompt "dark 808, minimal, Kingston night"
python maxxki_kalado_dancehall.py --preset 808_pressure --duration 120 --key G
```

### Master-Ear (Streamlit UI)

```bash
pip install streamlit
streamlit run maxxki_master_ear.py
```

### MIDI → WAV (FluidSynth)

```bash
python test_beat.py
# Erzeugt ~/projects/mk/music-output/midi/test_kalado.mid + test_kalado.wav
```

### Audio signieren & verifizieren

```bash
# Einzelne WAV signieren und in Registry eintragen
python maxxki_sign_session.py output.wav --prompt "dark dancehall" --generator musicgen-small

# Batch-Signierung
python maxxki_sign_session.py *.wav --generator musicgen-small

# Echtheit prüfen
python maxxki_verify.py output.wav

# Registry einsehen
python maxxki_registry.py list
python maxxki_registry.py verify
python maxxki_registry.py find mxk_20260603_134722
```

---

## MIDI-Generator: Genres, Scales & Progressionen

| Genre | Scale | Drums | Progression |
|---|---|---|---|
| `dancehall` | major | dancehall | I–V–IIIm–IV |
| `lofi` | major | lofi | Imaj7–IVm7–IIm7–Vdom7 |
| `dark` | phrygian | trap | Im–VII–VI–IVm |
| `drill` | minor | trap | Im–VI–VII–IVm |
| `techno` | dorian | techno | Im–Im–VIIm–VIm |
| `afrobeats` | pentatonic | afrobeats | I–II–I–VII |
| `jazz` | major | lofi | Imaj7–Vdom7–IIm7–Vdom7 |
| `blues` | blues | basic | I7–I7–V7–I7 |
| `sad` | minor | lofi | Im–VI–IIIm–IV |
| `euphoric` | major | basic | I–IV–V–IIIm |

**Chord-Stile:** `block` (ganzer Takt), `arp` (arpeggiert), `stab` (kurze Stabs)

**Bass-Stile:** `root`, `walking` (Root + Quinte), `syncopated` (off-beat)

---

## Orchestrator: Vibe-Profile

| Vibe | BPM | Sub Boost | Sidechain | Beschreibung |
|---|---|---|---|---|
| `the_produck` | 142 | +6 dB | 50% | Minimalist, druckvoll, düster |
| `dark_minimal` | 130 | +5 dB | 60% | Reduziert aufs Essentielle |
| `industrial_drill` | 140 | +3 dB | 70% | UK Drill meets Industrial |
| `lofi_noir` | 85 | +2 dB | aus | Düstere Lo-Fi Ästhetik |
| `warehouse_techno` | 138 | +4 dB | 45% | Roh, hypnotisch, repetitiv |

---

## Kalado-Style Parameter

| Parameter | Wert | Beschreibung |
|---|---|---|
| BPM | 100 | Lazy but heavy |
| Bass Octave | 1 | Tiefer als Standard (sub-fokus) |
| Bass Style | `syncopated` | Off-beat bounce |
| Chord Style | `stab` | Kurze, aggressive Stabs |
| Melody Density | 0.3 | Sehr sparsam |
| Sub Boost | +8 dB | Extrem (Standard: 4–6 dB) |
| Sidechain Amount | 0.6 | Aggressiv (Standard: 0.4–0.5) |

Presets: `dark_riddim`, `produck_bashment`, `midnight_kingston`, `808_pressure`

---

## Fingerprinting & Provenance

Jede Produktion kann mit einem signierten Manifest versehen werden:

```
output.wav
output.manifest.json    ← SHA-256 + HMAC-Signatur
~/.maxxki/registry.chain ← append-only Hash-Chain aller Einträge
```

Das Manifest enthält: `raw_sha256`, `final_sha256`, `prompt`, `generator`,
`session_id`, `created_at`, `host`, `user` und eine HMAC-SHA256-Signatur
über das kanonische JSON. Manipulation eines Registry-Eintrags bricht
die Hash-Chain — `maxxki_verify.py` meldet es.

Der HMAC-Key wird beim ersten Start lokal generiert (`~/.maxxki/signing.key`,
chmod 600) und verlässt das Gerät nicht.

---

## Orchestrator: Session-Flow

```
SessionConfig
    │
    ├─ MasterEarEngine.analyze()     → MasteringProfile + MusicGen-Prompt
    │
    ├─ _generate_midi_foundation()   → MIDI (Chords + Bass + Drums + Melody)
    │      ├─ beat_gen: Chords, Bass, Drums
    │      └─ midi_gen: GPT-2 Melodie (Fallback: algorithmisch)
    │
    ├─ MusicGenerator.async_generate() → Texture-WAV (MusicGen)
    │
    ├─ MIDIRenderer.render_stems()   → WAV-Stems via FluidSynth
    │
    ├─ StemMixer.mix_stems()         → Stereo-Mix (Gain, Pan, EQ, Reverb, Delay)
    │
    └─ PostProcessor.process()       → Mastered WAV (EQ, Kompression, Sidechain, Limiter)
```

---

## Output-Verzeichnisse

```
~/projects/mk/music-output/
├── *.wav                  # MusicGen-Outputs
├── final/
│   └── mxk_<id>_mastered.wav
├── stems/
│   └── <session_id>/
│       └── *.wav
└── midi/
    └── *.mid              # Beat Generator + MIDI Gen
```

---

## Orchestrator CLI-Flags

```
--vibe     {the_produck, dark_minimal, industrial_drill, lofi_noir, warehouse_techno}
--bpm      BPM (überschreibt Vibe-Default)
--key      Tonart (C, C#, D, … B)
--duration Sekunden (default: 60)
--no-claude      Master-Ear deaktivieren (Rule-based Fallback)
--no-midi-gen    GPT-2 Melodie deaktivieren (algorithmischer Fallback)
--no-music-gen   MusicGen-Textur deaktivieren
```

---

## Umgebungsvariablen

| Variable | Pflicht | Beschreibung |
|---|---|---|
| `HF_TOKEN` | Nein | HuggingFace — für höhere Rate Limits |

---

## Hardware-Empfehlungen

| Setup | Modell | ~Zeit pro 10s |
|---|---|---|
| CPU | musicgen-small | ~10 min |
| CUDA GPU | musicgen-small/medium | ~30 sec |
| Apple Silicon (MPS) | musicgen-small | ~2 min |

---

## Lizenz

MIT — do whatever, credit appreciated.

---

*Built by mk · Kingston vibes, local first*
