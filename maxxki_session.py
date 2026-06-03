#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from maxxki_music_gen import MusicGenerator, MusicGeneratorConfig

# Einmalig laden
print("Loading MusicGen...")
gen = MusicGenerator.get_instance(MusicGeneratorConfig())
gen.load_model()
print("✓ Ready — generating multiple tracks...")

# Mehrere Prompts ohne Neuladen
prompts = [
    "dark dancehall riddim, heavy 808",
    "bashment energy, minimal drums",
    "midnight Kingston, deep sub",
]

for prompt in prompts:
    print(f"\n→ {prompt}")
    path = gen.generate(prompt, duration=10)
    print(f"  ✓ {path}")

print("\nAll done — model still cached until this script exits")
