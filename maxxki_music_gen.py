"""
MAXXKI Music Generator — MusicGen Small
Supports CPU, CUDA, MPS.

Fixes v2:
- top_p=0.0 + do_sample=True war widersprüchlich → top_p=0.95
- guidance_scale ohne negative prompt arbeitet gegen sich selbst
  bei unkonditionierter Generation → 1.0 (aus), mit Prompt → 3.0
- Singleton hat config-Änderungen (--model) gecacht und ignoriert → invalidiert bei Mismatch
- ThreadPoolExecutor wurde pro async_generate-Call neu erstellt → shared executor
- NaN/Inf-Bereinigung bleibt, Normalisierung bleibt
"""

import asyncio
import concurrent.futures
import gc
import logging
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("music_gen.log"),
    ],
)
logger = logging.getLogger("MAXXKI-MusicGen")


@dataclass(frozen=True)
class MusicGeneratorConfig:
    model_id: str = "facebook/musicgen-small"
    output_dir: Path = field(default_factory=lambda: Path.home() / "projects/mk/music-output")
    model_cache_dir: Path = field(
        default_factory=lambda: Path.home() / "projects/mk/models/musicgen-small"
    )
    max_duration: int = 30
    # MusicGen internal token rate — DO NOT CHANGE
    tokens_per_second: int = 50


# ─── Genre Presets ──────────────────────────────────────────────────────────
# Keep prompts short and concrete. MusicGen ignores abstract quality descriptors.
GENRE_PRESETS = {
    "dancehall": (
        "dancehall riddim, heavy 808 sub bass, syncopated kick drum, skank guitar, "
        "energetic, 90 BPM, Jamaica, reggae influence"
    ),
    "electro": (
        "electro banger, aggressive lead synth, four on the floor kick, "
        "hard hitting 808 bass, rave energy, 135 BPM"
    ),
    "lofi": (
        "lo-fi hip hop, jazzy chords, mellow boom bap drums, vinyl warmth, "
        "relaxed, 75 BPM"
    ),
    "techno": (
        "techno, driving kick drum, dark minimal bassline, industrial atmosphere, "
        "hypnotic, 140 BPM"
    ),
    "afrobeats": (
        "afrobeats, percussion groove, bright guitar riff, warm bass, "
        "energetic danceable, 100 BPM"
    ),
    "drill": (
        "UK drill beat, sliding 808 bass, dark melody, hi hat rolls, "
        "aggressive trap drums, 140 BPM"
    ),
}


class MusicGeneratorError(Exception):
    pass


class MusicGenerator:
    """
    Singleton mit config-bewusster Cache-Invalidierung.
    Shared ThreadPoolExecutor für async_generate.
    """

    _instance: Optional["MusicGenerator"] = None
    _executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    @classmethod
    def get_instance(cls, config: Optional[MusicGeneratorConfig] = None) -> "MusicGenerator":
        cfg = config or MusicGeneratorConfig()
        # Invalidiere Singleton wenn model_id sich geändert hat
        if cls._instance is not None and cls._instance.config.model_id != cfg.model_id:
            logger.info(
                f"model_id geändert ({cls._instance.config.model_id} → {cfg.model_id}), "
                "lade neuen Generator."
            )
            cls._instance.unload_model()
            cls._instance = None
        if cls._instance is None:
            cls._instance = cls(cfg)
        return cls._instance

    @classmethod
    def _get_executor(cls) -> concurrent.futures.ThreadPoolExecutor:
        if cls._executor is None:
            cls._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return cls._executor

    @classmethod
    def shutdown(cls) -> None:
        """Executor und Modell sauber freigeben — aufrufen bei Programmende."""
        if cls._executor is not None:
            cls._executor.shutdown(wait=True)
            cls._executor = None
        if cls._instance is not None:
            cls._instance.unload_model()
            cls._instance = None

    def __init__(self, config: Optional[MusicGeneratorConfig] = None):
        self.config = config or MusicGeneratorConfig()
        self._processor: Any = None
        self._model: Any = None
        self._device: Optional[str] = None
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.model_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._detect_device()
        return self._device

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA detected — using GPU.")
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("MPS detected — using Apple Silicon.")
                return "mps"
        except ImportError:
            pass
        logger.info("No GPU found — using CPU (slow but works).")
        return "cpu"

    def load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            import torch

            logger.info(f"Loading model '{self.config.model_id}' on {self.device}...")
            torch_dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

            self._processor = AutoProcessor.from_pretrained(
                self.config.model_id,
                cache_dir=str(self.config.model_cache_dir),
            )
            self._model = MusicgenForConditionalGeneration.from_pretrained(
                self.config.model_id,
                cache_dir=str(self.config.model_cache_dir),
                torch_dtype=torch_dtype,
            ).to(self.device)
            self._model.eval()
            logger.info(f"Model ready ({torch_dtype}).")

        except Exception as e:
            raise MusicGeneratorError(f"Model load failed: {e}") from e

    def unload_model(self) -> None:
        import torch
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    def generate(self, prompt: str, duration: int) -> Path:
        """
        Generiert Audio aus einem Text-Prompt.

        guidance_scale-Logik:
          - Mit Prompt (konditioniert): 3.0 — CFG zieht die Generation
            in Richtung des Prompts weg von unkonditionierter Baseline.
          - Kein Prompt / leerer Prompt: 1.0 — CFG deaktiviert,
            da es keine sinnvolle Richtung gibt von der weggelenkt werden kann.

        top_p=0.95 + do_sample=True + temperature=1.0:
          - top_p=0.0 im Original hat nucleus sampling effektiv deaktiviert
            (nur das wahrscheinlichste Token wird gewählt = greedy).
            Das produziert repetitive, sterile Outputs.
          - 0.95 lässt genug Varianz für musikalische Überraschungen.
        """
        self.load_model()

        import torch
        import scipy.io.wavfile
        import numpy as np

        duration = max(1, min(duration, self.config.max_duration))
        max_tokens = duration * self.config.tokens_per_second

        has_prompt = bool(prompt and prompt.strip())
        guidance = 3.0 if has_prompt else 1.0

        logger.info(f"Generating {duration}s — guidance={guidance} — prompt: '{prompt}'")

        inputs = self._processor(
            text=[prompt] if has_prompt else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        try:
            with torch.no_grad():
                audio_values = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    guidance_scale=guidance,
                    temperature=1.0,
                    top_k=250,
                    top_p=0.95,      # war 0.0 → nucleus sampling war deaktiviert
                )

            sampling_rate = self._model.config.audio_encoder.sampling_rate
            audio_data = audio_values[0, 0].cpu().numpy().astype("float32")

            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

            max_val = np.abs(audio_data).max()
            if max_val > 0:
                audio_data = (audio_data / max_val) * 0.92

            audio_int16 = (audio_data * 32767).astype(np.int16)

            safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in prompt[:40])
            if not safe_name:
                safe_name = "unconditional"
            out_path = self.config.output_dir / f"{safe_name}_{duration}s.wav"

            scipy.io.wavfile.write(str(out_path), rate=sampling_rate, data=audio_int16)
            logger.info(f"Saved: {out_path}")
            return out_path

        except Exception as e:
            raise MusicGeneratorError(f"Generation failed: {e}") from e

    async def async_generate(self, prompt: str, duration: int) -> Path:
        """
        Async wrapper. Nutzt shared executor statt pro-Call neuen ThreadPoolExecutor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._get_executor(),
            self.generate,
            prompt,
            duration,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="MAXXKI Music Generator v2",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Genre presets: " + ", ".join(GENRE_PRESETS.keys()),
    )
    parser.add_argument(
        "prompt", type=str, nargs="?",
        help="Musik-Beschreibung oder Genre-Preset-Name",
    )
    parser.add_argument(
        "--duration", type=int, default=15,
        help="Dauer in Sekunden (default: 15, max: 30)",
    )
    parser.add_argument(
        "--model", type=str, default="facebook/musicgen-small",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="Zeigt alle Genre-Presets an",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug-Logging aktivieren",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_presets:
        print("\n⬡ MAXXKI Genre Presets:\n")
        for name, prompt in GENRE_PRESETS.items():
            print(f"  {name:12s} → {prompt[:60]}...")
        print()
        sys.exit(0)

    prompt = args.prompt
    if not prompt:
        print("\n⬡ MAXXKI Music Generator")
        print("  Genre Presets:", ", ".join(GENRE_PRESETS.keys()))
        prompt = input("\nPrompt oder Preset eingeben: ").strip()

    if prompt and prompt.lower() in GENRE_PRESETS:
        logger.info(f"Preset '{prompt}' erkannt.")
        prompt = GENRE_PRESETS[prompt.lower()]

    config = MusicGeneratorConfig(model_id=args.model)

    try:
        generator = MusicGenerator.get_instance(config)
        output_file = generator.generate(prompt or "", args.duration)
        print(f"\n[✓] Fertig! → {output_file}")
        print(f"    Abspielen: xdg-open '{output_file}'\n")
    except MusicGeneratorError as e:
        logger.critical(f"Fehler: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Abgebrochen.")
        sys.exit(0)


if __name__ == "__main__":
    main()
