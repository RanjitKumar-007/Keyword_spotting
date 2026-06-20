import logging
import wave
import numpy as np
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Indian languages supported by IndicConformer
INDIC_LANGUAGES = {"hi", "ta", "kn", "te", "ml", "bn", "gu", "mr", "pa", "ur"}

# Sample audio (in seconds) for language detection
LANGUAGE_DETECTION_SECONDS = 10


def _load_wav_as_tensor(path: str) -> tuple[torch.Tensor, int]:
    """
    Load a PCM WAV file into a mono float32 tensor of shape [1, T], using
    only the stdlib `wave` module + numpy.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    wav = torch.from_numpy(data).unsqueeze(0)  # [1, T]
    return wav, sample_rate


class WhisperASRError(Exception):
    pass


class IndicConformerASR:
    """
    Wraps AI4Bharat's IndicConformer for Indian-language transcription.
    """

    INDIC_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

    def __init__(
        self,
        decoding: str = "rnnt",
        cache_dir: str | None = None,
    ):
        self.decoding = decoding
        self._cache_dir = cache_dir
        if torch.cuda.is_available():
            self.device = "cuda"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(
                f"IndicConformer — GPU: {torch.cuda.get_device_name(0)} "
                f"({vram_gb:.1f} GB VRAM)"
            )
        else:
            self.device = "cpu"
            logger.warning(
                "IndicConformer — No GPU detected. Running on CPU. "
                "Transcription will be significantly slower."
            )
        logger.info(f"Loading IndicConformer model: {self.INDIC_MODEL_ID} ...")
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.INDIC_MODEL_ID,
                trust_remote_code=True,
                cache_dir=self._cache_dir,
                local_files_only=True,
            )
            if hasattr(self.model, "to"):
                self.model = self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            logger.info("IndicConformer model loaded.")
        except Exception as e:
            raise WhisperASRError(
                f"Failed to load IndicConformer ({self.INDIC_MODEL_ID}): {e}. "
                "Ensure the model (with trust_remote_code files) is present "
                "in the local HuggingFace cache."
            ) from e

    
    def transcribe(self, audio_file: str, language: str) -> dict:
        """
        Transcribes Indian-language audio using IndicConformer.
        """
        return self._transcribe_indicconformer(audio_file, language)


    def _transcribe_indicconformer(self, audio_file: str, language: str) -> dict:
        logger.info(
            f"IndicConformer transcribing [{language}]: {audio_file} "
            f"(decoding={self.decoding})"
        )
        try:
            wav, sr = _load_wav_as_tensor(audio_file)
            if sr != 16000:
                import torchaudio
                wav = torchaudio.functional.resample(wav, sr, 16000)
            duration = wav.shape[-1] / 16000.0
            wav = wav.to(self.device)
            with torch.no_grad():
                transcript = self.model(wav, language, self.decoding)
            if isinstance(transcript, (list, tuple)):
                transcript = transcript[0] if transcript else ""
            transcript = (transcript or "").strip()
        except Exception as e:
            raise WhisperASRError(
                f"IndicConformer transcription failed for [{language}]: {e}"
            ) from e
        if not transcript:
            logger.warning("[IndicConformer] Empty transcript returned.")
            return {"transcript": "", "segments": []}
        segment = {
            "start": 0.0,
            "end": round(duration, 3),
            "text": transcript,
            "confidence": None,
            "no_speech_prob": None,
        }
        logger.info(f"[IndicConformer] transcript length: {len(transcript)} chars")
        return {
            "transcript": transcript,
            "segments": [segment],
        }


class WhisperASR:
    """
    Language-aware ASR router.
    Detection pass (10-second sample via Faster-Whisper Large-v3):
        English      → full transcription with Faster-Whisper Large-v3
        Indian lang  → full transcription with IndicConformer
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        confidence_threshold: float = 0.5,
        no_speech_threshold: float = 0.6,
    ):
        self.confidence_threshold = confidence_threshold
        self.no_speech_threshold = no_speech_threshold
        if torch.cuda.is_available():
            self.device = "cuda"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.compute_type = "float16" if vram_gb >= 8 else "int8_float16"
            logger.info(
                f"GPU: {torch.cuda.get_device_name(0)} "
                f"({vram_gb:.1f} GB VRAM) — compute_type={self.compute_type}"
            )
        else:
            self.device = "cpu"
            self.compute_type = "int8"
            logger.warning(
                "No GPU detected — running Faster-Whisper on CPU (int8). "
                "Transcription will be significantly slower."
            )
        logger.info(f"Loading Faster-Whisper '{model_size}' on {self.device}...")
        try:
            self.fw_model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Faster-Whisper model loaded.")
        except Exception as e:
            raise WhisperASRError(f"Failed to load Faster-Whisper model: {e}") from e
        self._indic_asr: IndicConformerASR | None = None

    def _get_indic_asr(self) -> IndicConformerASR:
        """Initialise IndicConformer on Indian-language"""
        if self._indic_asr is None:
            logger.info("Initialising IndicConformer for Indian-language input")
            self._indic_asr = IndicConformerASR()
        return self._indic_asr

    def _detect_language(self, audio_file: str) -> tuple[str, float]:
        """
        Language detection using Faster-Whisper Large-v3.
        """
        try:
            from faster_whisper.audio import decode_audio
            audio = decode_audio(audio_file, sampling_rate=16000)
            clip = audio[: LANGUAGE_DETECTION_SECONDS * 16000]
            _, info = self.fw_model.transcribe(
                clip,
                task="transcribe",
                beam_size=1,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            return info.language, info.language_probability
        except Exception as e:
            raise WhisperASRError(f"Language detection failed: {e}") from e
            

    def _transcribe_english(self, audio_file: str) -> dict:
        """Full transcription for English audio using Faster-Whisper Large-v3."""
        logger.info("English audio — transcribing with Faster-Whisper Large-v3.")
        try:
            raw_segments, info = self.fw_model.transcribe(
                audio_file,
                task="translate",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                word_timestamps=True,
            )
            raw_segments = list(raw_segments)
        except Exception as e:
            raise WhisperASRError(f"English transcription failed: {e}") from e
        return self._filter_segments(raw_segments, info)


    def _filter_segments(self, raw_segments: list, info) -> dict:
        accepted, rejected = [], 0
        full_text_parts = []
        for seg in raw_segments:
            avg_logprob    = getattr(seg, "avg_logprob",    -1.0) or -1.0
            no_speech_prob = getattr(seg, "no_speech_prob",  0.0) or  0.0
            if no_speech_prob > self.no_speech_threshold:
                rejected += 1
                logger.debug(
                    f"Segment rejected (no_speech_prob={no_speech_prob:.2f}): "
                    f"'{seg.text.strip()}'"
                )
                continue
            confidence = max(0.0, min(1.0, pow(2.718281828, avg_logprob)))
            if confidence < self.confidence_threshold:
                rejected += 1
                logger.debug(
                    f"Segment rejected (conf={confidence:.3f}): "
                    f"'{seg.text.strip()}'"
                )
                continue
            accepted.append({
                "start":          round(seg.start, 3),
                "end":            round(seg.end,   3),
                "text":           seg.text.strip(),
                "confidence":     round(confidence,     4),
                "no_speech_prob": round(no_speech_prob, 4),
            })
            full_text_parts.append(seg.text.strip())
        logger.info(f"Segments — accepted: {len(accepted)}, rejected: {rejected}")
        return {
            "language":             info.language,
            "language_probability": round(info.language_probability, 4),
            "transcript":           " ".join(full_text_parts),
            "segments":             accepted,
        }

    
    def transcribe(self, audio_file: str) -> dict:
        """
        Route audio through the correct ASR model based on detected language.
        Detect language (Faster-Whisper Large-v3, 10-sec sample).
        English → Faster-Whisper Large-v3 full transcription.
        Indian language → IndicConformer full transcription.
        """
        logger.info(f"Transcribing: {audio_file}")
        detected_lang, lang_prob = self._detect_language(audio_file)
        logger.info(
            f"Detected language: {detected_lang} (prob={lang_prob:.2f})"
        )
        if detected_lang == "en":
            result = self._transcribe_english(audio_file)
            result["asr_model"] = "faster-whisper-large-v3"
        elif detected_lang in INDIC_LANGUAGES:
            indic_asr = self._get_indic_asr()
            result = indic_asr.transcribe(audio_file, language=detected_lang)
            result["language"] = detected_lang
            result["language_probability"] = round(lang_prob, 4)
            result["asr_model"] = f"indic-conformer-600m ({detected_lang})"
            logger.info(
                f"Indian-language audio [{detected_lang}] — "
                "transcript is in source language; IndicTrans2 will translate."
            )
        else:
            logger.warning(
                f"Language '{detected_lang}' is not in INDIC_LANGUAGES and "
                "is not English. Falling back to Faster-Whisper Large-v3 "
                "transcription (quality may be lower)."
            )
            result = self._transcribe_english(audio_file)
            result["asr_model"] = f"faster-whisper-large-v3 (fallback for {detected_lang})"
        logger.info(
            f"ASR complete — model: {result['asr_model']}, "
            f"segments: {len(result['segments'])}, "
            f"transcript length: {len(result['transcript'])} chars"
        )
        return result
        