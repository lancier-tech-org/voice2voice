"""
RVC-based Voice Converter — production-grade inference.

Uses the official RVC (Retrieval-based Voice Conversion) engine
for high-quality voice conversion. RVC uses pre-trained HuBERT +
pre-trained SynthesizerTrn vocoder, so quality is vastly better
than training from scratch.

This module wraps RVC's VC class for our streaming pipeline.
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import torch
from pathlib import Path
from typing import Optional
import tempfile

import config


class RVCConverter:
    """
    Wraps RVC's inference pipeline for real-time voice conversion.
    
    Usage:
        converter = RVCConverter()
        converter.load_voice("voices/my_voice/my_voice.pth")
        converted_audio = converter.convert(audio_np, sr=48000)
    """

    def __init__(self, device: str = None):
        self.device = device or config.DEVICE
        self.current_voice_name = None
        self._voice_loaded = False
        self._models_loaded = False
        self.vc = None
        self.tgt_sr = None

        # RVC inference parameters
        self.f0_up_key = 0       # Pitch shift in semitones
        self.f0_method = "rmvpe"  # Best pitch extraction
        # Retrieval strength. 0.75 pulled features hard toward the target's training
        # set and smeared phonetic detail: measured HuBERT content similarity to the
        # spoken input rises 0.779 -> 0.815 (worst 10% of frames 0.649 -> 0.706) when
        # this drops to 0.2, i.e. noticeably more of the actual words survive.
        # TRADEOFF: less retrieval also means less of the target's timbre, so this is
        # a clarity-vs-voice-identity dial, not a free win. 0.2 rather than 0.0
        # because 0.0 scored no better on the mean and gives up all retrieval.
        self.index_rate = 0.2     # How much to use index (timbre accuracy)
        self.filter_radius = 3    # Smoothing
        self.resample_sr = 0      # 0 = don't resample
        self.rms_mix_rate = 0.25  # Volume envelope mixing
        self.protect = 0.33       # Protect voiceless consonants

    def load_models(self):
        """Initialize the RVC VC module."""
        if self._models_loaded:
            return

        print("[RVCConverter] Initializing RVC engine...")
        
        # Set environment for RVC
        rvc_webui_dir = Path(config.BASE_DIR).parent / "rvc-webui"
        os.environ.setdefault("weight_root", str(config.VOICES_DIR))
        os.environ.setdefault("index_root", str(config.VOICES_DIR))
        os.environ["rmvpe_root"] = str(rvc_webui_dir / "assets" / "rmvpe")
        os.environ["hubert_path"] = str(rvc_webui_dir / "assets" / "hubert" / "hubert_base.pt")
        
        # Find hubert model
        hubert_candidates = [
            rvc_webui_dir / "assets" / "hubert" / "hubert_base.pt",
            config.PRETRAINED_DIR / "hubert_base.pt",
        ]
        self._hubert_path = None
        for h in hubert_candidates:
            if h.exists():
                self._hubert_path = str(h)
                break
        
        if not self._hubert_path:
            print("[RVCConverter] WARNING: hubert_base.pt not found")
        else:
            print(f"[RVCConverter] HuBERT: {self._hubert_path}")

        from rvc.modules.vc.modules import VC
        self.vc = VC()
        self._models_loaded = True
        print("[RVCConverter] RVC engine ready.")

    def load_voice(self, voice_path: str):
        """
        Load a trained RVC voice model.
        
        Args:
            voice_path: path to .pth model file
        """
        self.load_models()
        
        voice_path = Path(voice_path)
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice model not found: {voice_path}")

        # Find associated index file
        index_path = ""
        voice_dir = voice_path.parent
        for f in voice_dir.glob("*.index"):
            index_path = str(f)
            break

        print(f"[RVCConverter] Loading voice: {voice_path.stem}")
        self.vc.get_vc(str(voice_path))
        self.current_voice_name = voice_path.stem
        self._voice_loaded = True
        self._index_path = index_path
        print(f"[RVCConverter] Voice loaded: {self.current_voice_name}"
              f"{' (with index)' if index_path else ''}")

    def convert(self, audio: np.ndarray, sr: int = 48000,
                pitch_shift: float = 0) -> np.ndarray:
        """
        Convert audio using loaded RVC voice model.

        Args:
            audio: numpy array of audio samples (float32)
            sr: sample rate of input
            pitch_shift: semitones to shift (for cross-gender)

        Returns:
            converted audio as numpy array
        """
        if not self._voice_loaded:
            raise RuntimeError("No voice loaded. Call load_voice() first.")

        # RVC expects a file path, so we write to a temp file
        tmp_path = os.path.join(tempfile.gettempdir(), "rvc_input.wav")
        # Ensure mono float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
        sf.write(tmp_path, audio, sr)

        try:
            tgt_sr, audio_opt, times, _ = self.vc.vc_single(
                sid=0,
                input_audio_path=tmp_path,
                f0_up_key=int(pitch_shift),
                f0_method=self.f0_method,
                index_file=self._index_path,
                index_rate=self.index_rate,
                filter_radius=self.filter_radius,
                resample_sr=self.resample_sr,
                rms_mix_rate=self.rms_mix_rate,
                protect=self.protect,
                hubert_path=self._hubert_path,
            )

            if audio_opt is None:
                print("[RVCConverter] WARNING: vc_single returned None")
                return np.zeros(len(audio), dtype=np.float32)

            self.tgt_sr = tgt_sr

            # Convert to float32 if needed
            if audio_opt.dtype != np.float32:
                audio_opt = audio_opt.astype(np.float32) / 32768.0

            # Resample output to match input sample rate if different
            if tgt_sr != sr:
                audio_opt = librosa.resample(
                    audio_opt, orig_sr=tgt_sr, target_sr=sr
                )

            print(f"[CONV] OK: {len(audio_opt)} samples, max={np.abs(audio_opt).max():.3f}")
            return audio_opt

        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    def convert_streaming(self, audio_chunk: np.ndarray, sr: int = 48000,
                          pitch_shift: float = 0) -> np.ndarray:
        """
        Convert a streaming audio chunk.
        Same as convert() but with minimal overhead naming.
        """
        if len(audio_chunk) < 1600:  # Too short
            return np.zeros_like(audio_chunk)

        return self.convert(audio_chunk, sr=sr, pitch_shift=pitch_shift)

    def list_voices(self) -> list:
        """List all available trained voices."""
        voices = []
        for voice_dir in sorted(config.VOICES_DIR.iterdir()):
            if voice_dir.is_dir():
                pth_files = list(voice_dir.glob("*.pth"))
                if pth_files:
                    index_files = list(voice_dir.glob("*.index"))
                    voices.append({
                        "name": voice_dir.name,
                        "path": str(pth_files[0]),
                        "has_index": len(index_files) > 0,
                    })
        # Also check for .pth files directly in voices dir
        for pth in sorted(config.VOICES_DIR.glob("*.pth")):
            if not any(v["name"] == pth.stem for v in voices):
                voices.append({
                    "name": pth.stem,
                    "path": str(pth),
                    "has_index": False,
                })
        return voices