"""
MAXXKI Master-Ear Agent v2.0 — GPT-2 Edition
Kein Claude, kein API-Key, 100% lokal.

Architektur:
- GPT-2 als Mastering Engineer (JSON-basierte Frequenz/Dynamik-Analyse)
- MusicGen für Roh-Audio-Generation
- Post-Processing Pipeline: scipy-basiert
- MIDI-Gen für strukturelle Vorgaben

GPT-2 wird fine-tuned auf Mastering-Profile-Generierung aus Prompts.
Falls kein fine-tuned Modell vorhanden: Rule-basierte Fallback-Analyse.
"""

import os
import sys
import logging
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from datetime import datetime
from enum import Enum

import numpy as np
import scipy.signal
import scipy.ndimage
import scipy.io.wavfile

# GPT-2 für lokale Analyse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

from maxxki_music_gen import MusicGenerator, MusicGeneratorConfig, MusicGeneratorError, GENRE_PRESETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MAXXKI-MasterEar-GPT2")


# ─── Vibe-Profile (unverändert) ─────────────────────────────────────────────

class VibeProfile(Enum):
    THE_PRODUCK      = "the_produck"
    DARK_MINIMAL     = "dark_minimal"
    INDUSTRIAL_DRILL = "industrial_drill"
    LOFI_NOIR        = "lofi_noir"
    WAREHOUSE_TECHNO = "warehouse_techno"


@dataclass
class MasteringProfile:
    name: str
    description: str

    # EQ
    low_cut_hz: float    = 30.0
    sub_boost_db: float  = 4.0
    mid_scoop_db: float  = -2.0
    high_shelf_db: float = 2.0

    # Kompression
    compression_ratio: float     = 4.0
    compression_threshold: float = -12.0
    attack_ms: float             = 10.0
    release_ms: float            = 100.0

    # Sidechain
    sidechain_enabled: bool  = True
    sidechain_freq_hz: float = 100.0
    sidechain_amount: float  = 0.4

    # Reverb
    reverb_size: float = 0.3
    reverb_damp: float = 0.8
    reverb_wet: float  = 0.15

    # MIDI-Vorgabe
    midi_density: str = "sparse"
    midi_bpm: int     = 140


VIBE_PROFILES: Dict[VibeProfile, MasteringProfile] = {
    VibeProfile.THE_PRODUCK: MasteringProfile(
        name="The Produck",
        description="Minimalist, druckvoll, düster.",
        sub_boost_db=6.0,
        mid_scoop_db=-3.0,
        sidechain_amount=0.5,
        reverb_wet=0.10,
        midi_density="sparse",
        midi_bpm=142,
    ),
    VibeProfile.DARK_MINIMAL: MasteringProfile(
        name="Dark Minimal",
        description="Reduziert aufs Essentielle.",
        sub_boost_db=5.0,
        compression_ratio=6.0,
        sidechain_amount=0.6,
        reverb_wet=0.08,
        midi_density="sparse",
        midi_bpm=130,
    ),
    VibeProfile.INDUSTRIAL_DRILL: MasteringProfile(
        name="Industrial Drill",
        description="UK Drill meets Industrial.",
        low_cut_hz=40.0,
        sub_boost_db=3.0,
        high_shelf_db=4.0,
        compression_ratio=8.0,
        sidechain_amount=0.7,
        reverb_size=0.5,
        reverb_wet=0.20,
        midi_density="normal",
        midi_bpm=140,
    ),
    VibeProfile.LOFI_NOIR: MasteringProfile(
        name="Lo-Fi Noir",
        description="Düstere Lo-Fi Ästhetik.",
        low_cut_hz=60.0,
        sub_boost_db=2.0,
        mid_scoop_db=-1.0,
        compression_ratio=2.0,
        sidechain_enabled=False,
        reverb_wet=0.25,
        midi_density="sparse",
        midi_bpm=85,
    ),
    VibeProfile.WAREHOUSE_TECHNO: MasteringProfile(
        name="Warehouse Techno",
        description="Roh, hypnotisch, repetitiv.",
        low_cut_hz=35.0,
        sub_boost_db=4.0,
        mid_scoop_db=-4.0,
        compression_ratio=5.0,
        sidechain_amount=0.45,
        reverb_size=0.6,
        reverb_wet=0.12,
        midi_density="sparse",
        midi_bpm=138,
    ),
}


# ─── GPT-2 Master-Ear Engine ───────────────────────────────────────────────

class GPT2MasterEar:
    """
    GPT-2 basierte Mastering-Analyse.
    Nutzt entweder ein fine-tuned Modell oder rule-basierte Fallback-Logik.
    """

    # Prompt-Template für GPT-2 JSON-Generierung
    ANALYSIS_PROMPT = """Analyze this music production request and output a JSON mastering plan.

Request: "{user_prompt}"
Vibe hint: {vibe_hint}

Output strict JSON:
{{
    "vibe_profile": "the_produck",
    "mastering_notes": "brief frequency analysis",
    "frequency_plan": {{"low_cut": 30, "sub_boost_db": 6, "mid_scoop_db": -3, "high_shelf_db": 2}},
    "dynamics_plan": {{"compression_ratio": 4, "threshold_db": -12, "attack_ms": 10, "release_ms": 100, "sidechain_enabled": true, "sidechain_amount": 0.5}},
    "spatial_plan": {{"reverb_size": 0.3, "reverb_damp": 0.8, "reverb_wet": 0.1}},
    "structure_plan": {{"midi_density": "sparse", "bpm": 142}},
    "musicgen_prompt": "concise English prompt, max 2 sentences"
}}

JSON:"""

    def __init__(self, model_id: str = "gpt2"):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._load_model()

        # Keyword-basierte Fallback-Analyse
        self.keyword_map = {
            # Vibe-Profile
            "dark": "the_produck", "minimal": "dark_minimal", "industrial": "industrial_drill",
            "lofi": "lofi_noir", "warehouse": "warehouse_techno", "techno": "warehouse_techno",
            "drill": "industrial_drill", "noir": "lofi_noir",

            # BPM-Hints
            "slow": 85, "lazy": 100, "mid": 120, "fast": 140, "aggressive": 142,
            "hypnotic": 138, "driving": 138,

            # Density
            "sparse": "sparse", "minimal": "sparse", "few": "sparse",
            "full": "normal", "dense": "dense", "busy": "dense",

            # Sub-Bass
            "heavy 808": 8, "deep sub": 7, "sub-bass": 6, "bass": 4,

            # Sidechain
            "pump": 0.6, "duck": 0.6, "aggressive sidechain": 0.6,
            "smooth": 0.3, "subtle": 0.3,
        }

    def _load_model(self):
        """GPT-2 ohne Fine-Tuning generiert kein valides JSON → Rule-based Fallback."""
        logger.info("GPT-2 skipped — using rule-based analysis.")
        self._model = None

    def analyze(self, user_prompt: str, vibe_hint: Optional[str] = None) -> Tuple[MasteringProfile, str, Dict]:
        """
        Analysiert User-Prompt und gibt MasteringProfile + MusicGen-Prompt zurück.
        """
        gpt2_result = self._try_gpt2_analysis(user_prompt, vibe_hint)
        if gpt2_result:
            return gpt2_result

        logger.info("GPT-2 failed, using rule-based analysis.")
        return self._rule_based_analysis(user_prompt, vibe_hint)

    def _try_gpt2_analysis(self, user_prompt: str, vibe_hint: Optional[str]) -> Optional[Tuple[MasteringProfile, str, Dict]]:
        """Versucht GPT-2 JSON zu generieren und zu parsen."""
        if self._model is None:
            return None

        try:
            prompt = self.ANALYSIS_PROMPT.format(
                user_prompt=user_prompt,
                vibe_hint=vibe_hint or "none"
            )

            inputs = self._tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            generated = self._tokenizer.decode(output[0], skip_special_tokens=True)

            json_match = re.search(r'\{.*\}', generated, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                profile = self._build_profile_from_json(analysis)
                musicgen_prompt = analysis.get("musicgen_prompt", user_prompt)
                return profile, musicgen_prompt, analysis

        except Exception as e:
            logger.warning(f"GPT-2 analysis failed: {e}")

        return None

    def _rule_based_analysis(self, user_prompt: str, vibe_hint: Optional[str]) -> Tuple[MasteringProfile, str, Dict]:
        """Rule-basierte Analyse via Keyword-Matching."""
        prompt_lower = user_prompt.lower()
        analysis = {
            "vibe_profile": "the_produck",
            "mastering_notes": "Rule-based analysis",
            "frequency_plan": {},
            "dynamics_plan": {},
            "spatial_plan": {},
            "structure_plan": {},
            "musicgen_prompt": user_prompt,
        }

        VALID_VIBES = {v.value for v in VibeProfile}

        if vibe_hint:
            analysis["vibe_profile"] = vibe_hint
        else:
            for keyword, profile in self.keyword_map.items():
                if isinstance(profile, str) and profile in VALID_VIBES and keyword in prompt_lower:
                    analysis["vibe_profile"] = profile
                    break

        for keyword, bpm in self.keyword_map.items():
            if isinstance(bpm, int) and bpm > 50 and keyword in prompt_lower:
                analysis["structure_plan"]["bpm"] = bpm
                break

        for keyword, density in self.keyword_map.items():
            if density in ["sparse", "normal", "dense"] and keyword in prompt_lower:
                analysis["structure_plan"]["midi_density"] = density
                break

        for keyword, boost in self.keyword_map.items():
            if isinstance(boost, (int, float)) and boost > 3 and keyword in prompt_lower:
                analysis["frequency_plan"]["sub_boost_db"] = boost
                break

        for keyword, amount in self.keyword_map.items():
            if isinstance(amount, float) and keyword in prompt_lower:
                analysis["dynamics_plan"]["sidechain_amount"] = amount
                analysis["dynamics_plan"]["sidechain_enabled"] = True
                break

        analysis["musicgen_prompt"] = self._optimize_musicgen_prompt(user_prompt)

        profile = self._build_profile_from_json(analysis)
        return profile, analysis["musicgen_prompt"], analysis

    def _build_profile_from_json(self, analysis: Dict) -> MasteringProfile:
        """Baut MasteringProfile aus JSON-Analysis."""
        vibe_key = analysis.get("vibe_profile", "the_produck")
        try:
            base = VIBE_PROFILES[VibeProfile(vibe_key)]
        except (ValueError, KeyError):
            base = VIBE_PROFILES[VibeProfile.THE_PRODUCK]

        fp  = analysis.get("frequency_plan", {})
        dp  = analysis.get("dynamics_plan", {})
        sp  = analysis.get("spatial_plan", {})
        st_ = analysis.get("structure_plan", {})

        return MasteringProfile(
            name=base.name,
            description=analysis.get("mastering_notes", base.description),
            low_cut_hz=fp.get("low_cut", base.low_cut_hz),
            sub_boost_db=fp.get("sub_boost_db", base.sub_boost_db),
            mid_scoop_db=fp.get("mid_scoop_db", base.mid_scoop_db),
            high_shelf_db=fp.get("high_shelf_db", base.high_shelf_db),
            compression_ratio=dp.get("compression_ratio", base.compression_ratio),
            compression_threshold=dp.get("threshold_db", base.compression_threshold),
            attack_ms=dp.get("attack_ms", base.attack_ms),
            release_ms=dp.get("release_ms", base.release_ms),
            sidechain_enabled=dp.get("sidechain_enabled", base.sidechain_enabled),
            sidechain_amount=dp.get("sidechain_amount", base.sidechain_amount),
            reverb_size=sp.get("reverb_size", base.reverb_size),
            reverb_damp=sp.get("reverb_damp", base.reverb_damp),
            reverb_wet=sp.get("reverb_wet", base.reverb_wet),
            midi_density=st_.get("midi_density", base.midi_density),
            midi_bpm=st_.get("bpm", base.midi_bpm),
        )

    def _optimize_musicgen_prompt(self, user_prompt: str) -> str:
        """Konvertiert User-Prompt zu MusicGen-tauglichem Prompt."""
        prompt = user_prompt.lower()
        replacements = {
            "vibe": "atmosphere", "feel": "texture", "energy": "drive",
            "heavy": "hard hitting", "dark": "dark minimal", "minimal": "sparse",
        }
        for old, new in replacements.items():
            prompt = prompt.replace(old, new)
        sentences = prompt.split(".")
        return ". ".join(sentences[:2]).strip() + "."


# ─── Kompatibilitäts-Wrapper ──────────────────────────────────────────────

class MasterEarEngine(GPT2MasterEar):
    """API-kompatibler Wrapper für den Orchestrator."""
    pass


# ─── Post-Processor (unverändert) ───────────────────────────────────

class PostProcessor:
    """scipy-basierte Mastering-Chain."""

    def __init__(self, profile: MasteringProfile):
        self.p = profile

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = audio.astype(np.float64)
        audio = self._highpass(audio, sr)
        audio = self._eq(audio, sr)
        audio = self._compress(audio, sr)
        if self.p.sidechain_enabled:
            audio = self._sidechain(audio, sr)
        if self.p.reverb_wet > 0:
            audio = self._reverb(audio, sr)
        audio = self._limiter(audio)
        return audio.astype(np.float32)

    def _highpass(self, audio: np.ndarray, sr: int) -> np.ndarray:
        nyq = sr / 2.0
        cutoff = max(self.p.low_cut_hz, 1.0)
        if cutoff >= nyq:
            return audio
        sos = scipy.signal.butter(4, cutoff / nyq, btype="high", output="sos")
        return scipy.signal.sosfilt(sos, audio)

    def _eq(self, audio: np.ndarray, sr: int) -> np.ndarray:
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
        fft[freqs < 100] *= 10 ** (self.p.sub_boost_db / 20)
        mid = (freqs >= 200) & (freqs <= 500)
        fft[mid] *= 10 ** (self.p.mid_scoop_db / 20)
        fft[freqs > 8000] *= 10 ** (self.p.high_shelf_db / 20)
        return np.fft.irfft(fft, n=len(audio))

    def _compress(self, audio: np.ndarray, sr: int) -> np.ndarray:
        window = max(1, int(sr * 0.01))
        rms_sq = scipy.ndimage.uniform_filter1d(audio ** 2, size=window)
        rms = np.sqrt(np.maximum(rms_sq, 1e-12))
        db = 20 * np.log10(rms)
        over = np.maximum(db - self.p.compression_threshold, 0.0)
        gain_db = -over * (1.0 - 1.0 / self.p.compression_ratio)
        gain = 10 ** (gain_db / 20)
        return audio * gain

    def _sidechain(self, audio: np.ndarray, sr: int) -> np.ndarray:
        block_size = int(sr * 0.01)
        attack_coef = np.exp(-1.0 / max(1, int(sr * self.p.attack_ms / 1000)))
        release_coef = np.exp(-1.0 / max(1, int(sr * self.p.release_ms / 1000)))

        nyq = sr / 2.0
        split_freq = min(self.p.sidechain_freq_hz, nyq * 0.9)
        sos_low = scipy.signal.butter(4, split_freq / nyq, btype="low", output="sos")
        sos_high = scipy.signal.butter(4, split_freq / nyq, btype="high", output="sos")
        low_band = scipy.signal.sosfilt(sos_low, audio)
        high_band = scipy.signal.sosfilt(sos_high, audio)

        envelope = np.abs(audio)
        smoothed = np.zeros_like(envelope)
        smoothed[0] = envelope[0]
        for i in range(1, len(envelope)):
            if envelope[i] > smoothed[i - 1]:
                smoothed[i] = attack_coef * smoothed[i - 1] + (1 - attack_coef) * envelope[i]
            else:
                smoothed[i] = release_coef * smoothed[i - 1] + (1 - release_coef) * envelope[i]

        threshold = np.percentile(smoothed, 85)
        duck_curve = np.where(smoothed > threshold, 1.0 - self.p.sidechain_amount, 1.0)
        return low_band * duck_curve + high_band

    def _reverb(self, audio: np.ndarray, sr: int) -> np.ndarray:
        comb_delays_ms = [29.7, 37.1, 41.1, 43.7]
        comb_feedback = self.p.reverb_damp * 0.8
        wet = np.zeros_like(audio)
        for delay_ms in comb_delays_ms:
            delay_s = delay_ms * self.p.reverb_size / 1000.0
            d = max(1, int(sr * delay_s))
            if d >= len(audio):
                continue
            b = np.zeros(d + 1); b[0] = 1.0
            a = np.zeros(d + 1); a[0] = 1.0; a[d] = -comb_feedback
            wet += scipy.signal.lfilter(b, a, audio) * 0.25

        allpass_delay = max(1, int(sr * 0.005 * self.p.reverb_size))
        if allpass_delay < len(audio):
            b_ap = np.zeros(allpass_delay + 1)
            b_ap[0] = -0.7; b_ap[allpass_delay] = 1.0
            a_ap = np.zeros(allpass_delay + 1)
            a_ap[0] = 1.0; a_ap[allpass_delay] = -0.7
            wet = scipy.signal.lfilter(b_ap, a_ap, wet)

        return audio * (1.0 - self.p.reverb_wet) + wet * self.p.reverb_wet

    def _limiter(self, audio: np.ndarray) -> np.ndarray:
        ceiling = 10 ** (-0.5 / 20)
        peak = np.max(np.abs(audio))
        if peak > ceiling:
            audio = audio * (ceiling / peak)
        return np.clip(audio, -1.0, 1.0)


# ─── Streamlit UI ─────────────────────────────────────────────
# Lazy import — nur wenn als Streamlit-App gestartet, nicht beim CLI-Import
try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore

def init_session_state() -> None:
    defaults = {
        "master_ear": None,
        "music_gen": None,
        "last_profile": None,
        "last_analysis": None,
        "generated_files": [],
        "selected_vibe": VibeProfile.THE_PRODUCK,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_header() -> None:
    st.markdown("""
    <style>
    .mer-title { font-family: 'Courier New', monospace; color: #ff3333; letter-spacing: 4px; font-size: 2em; margin: 0; }
    .mer-sub { color: #555; font-size: 0.8em; letter-spacing: 2px; margin-bottom: 1em; }
    .param-label { color: #777; font-size: 0.72em; text-transform: uppercase; letter-spacing: 1px; }
    .param-val { color: #ff3333; font-family: monospace; font-weight: bold; }
    </style>
    <p class="mer-title">⬡ MASTER-EAR GPT-2</p>
    <p class="mer-sub">100% LOCAL — NO API — NO CLOUD</p>
    """, unsafe_allow_html=True)
    st.divider()


def render_vibe_selector() -> None:
    st.markdown("### VIBE PROFILE")
    cols = st.columns(len(VIBE_PROFILES))
    for idx, (vibe, profile) in enumerate(VIBE_PROFILES.items()):
        with cols[idx]:
            is_active = st.session_state.selected_vibe == vibe
            border = "#ff3333" if is_active else "#2a2a3e"
            st.markdown(f"""
            <div style="border:1px solid {border}; border-radius:4px; padding:12px;
                        background:{'#1a0a0a' if is_active else '#0a0a0f'}; min-height:90px;">
                <strong style="color:#ff3333;">{profile.name}</strong><br>
                <span style="color:#555;font-size:0.78em;">{profile.description[:55]}…</span><br>
                <span class="param-label">BPM </span>
                <span class="param-val">{profile.midi_bpm}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button(profile.name.upper(), key=f"vibe_{vibe.value}",
                        use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.selected_vibe = vibe
                st.rerun()


def render_mastering_params(profile: MasteringProfile) -> None:
    st.markdown("### MASTERING CHAIN")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**FREQUENCY**<br><p class='param-label'>Sub Boost</p><p class='param-val'>+{profile.sub_boost_db:.1f} dB</p>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"**DYNAMICS**<br><p class='param-label'>Sidechain</p><p class='param-val'>{profile.sidechain_amount*100:.0f}%</p>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"**SPATIAL**<br><p class='param-label'>BPM</p><p class='param-val'>{profile.midi_bpm}</p>", unsafe_allow_html=True)


def execute_generation(user_input: str, duration: int, use_gpt2: bool) -> None:
    if st.session_state.master_ear is None:
        st.session_state.master_ear = GPT2MasterEar()

    if st.session_state.music_gen is None:
        st.session_state.music_gen = MusicGenerator.get_instance(MusicGeneratorConfig())

    if use_gpt2:
        with st.spinner("GPT-2 analysiert Prompt…"):
            profile, musicgen_prompt, analysis = st.session_state.master_ear.analyze(
                user_input, vibe_hint=st.session_state.selected_vibe.value
            )
        st.session_state.last_profile = profile
        st.session_state.last_analysis = analysis
    else:
        profile = VIBE_PROFILES[st.session_state.selected_vibe]
        musicgen_prompt = user_input
        st.session_state.last_profile = profile
        st.session_state.last_analysis = {}

    with st.expander("Mastering Chain Parameter", expanded=True):
        render_mastering_params(profile)
        if st.session_state.last_analysis:
            st.json(st.session_state.last_analysis)

    with st.spinner(f"Generiere {duration}s Raw Audio…"):
        try:
            raw_path = st.session_state.music_gen.generate(musicgen_prompt, duration)
            sr, raw_data = scipy.io.wavfile.read(str(raw_path))
            raw_audio = raw_data.astype(np.float32) / 32767.0
            if raw_audio.ndim > 1:
                raw_audio = raw_audio.mean(axis=1)
        except Exception as e:
            st.error(f"Generation fehlgeschlagen: {e}")
            return

    with st.spinner("Mastering Chain läuft…"):
        processor = PostProcessor(profile)
        mastered = processor.process(raw_audio, sr)
        mastered_i16 = (mastered * 32767).astype(np.int16)
        mastered_path = raw_path.parent / f"{raw_path.stem}_MASTERED.wav"
        scipy.io.wavfile.write(str(mastered_path), sr, mastered_i16)

    st.success("Fertig.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**RAW (MusicGen)**")
        st.audio(str(raw_path))
    with c2:
        st.markdown("**MASTERED (GPT-2 Master-Ear)**")
        st.audio(str(mastered_path))


def main() -> None:
    st.set_page_config(page_title="MAXXKI Master-Ear GPT-2", page_icon="⬡", layout="wide")
    init_session_state()
    render_header()
    render_vibe_selector()
    st.divider()

    st.markdown("### GENERATE")
    user_input = st.text_area("Beschreibe deinen Track", 
                              placeholder="z.B. 'dark warehouse 808, minimal percussion, 3am vibe'",
                              height=80)
    col1, col2 = st.columns([1, 3])
    with col1:
        duration = st.slider("Dauer (s)", 5, 30, 15, 5)
    with col2:
        use_gpt2 = st.toggle("GPT-2 Analyse", value=True, 
                            help="GPT-2 analysiert und optimiert die Mastering-Chain")

    if st.button("⬡ GENERATE", type="primary", use_container_width=True, disabled=not user_input.strip()):
        execute_generation(user_input, duration, use_gpt2)


if __name__ == "__main__":
    main()
