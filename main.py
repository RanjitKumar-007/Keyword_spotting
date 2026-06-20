import json
import sys
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import logging
from json_logger import JSONLogger, setup_logging
from audio_preprocessor import AudioPreprocessor, AudioPreprocessorError
from whisper_asr import WhisperASR, WhisperASRError, INDIC_LANGUAGES
from translator import Translator, TranslatorError
from text_normalizer import TextNormalizer
from keyword_detector import KeywordDetector, KeywordDetectorError
from contextual_analyzer import ContextualAnalyzer, ContextualAnalyzerError


logger = logging.getLogger(__name__)

CONFIG_PATH = "keywords.json"
OUTPUT_DIR  = "outputs"
LOG_DIR     = "logs"

# Must match the exact tag from "ollama list"
CONTEXTUAL_MODEL = "qwen2.5:7b"

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def validate_audio_path(audio_file: str):
    if not audio_file:
        raise ValueError("No audio file provided.")
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

def build_pipeline(config: dict) -> tuple:
    """
    Builds all pipeline components once at startup.
    ASR routing (inside WhisperASR.transcribe):
        English      → Faster-Whisper Large-v3
        Indian lang  → Indic Conformer 600M
    Translation (inside Translator.translate_to_english):
        Indian lang  → IndicTrans2
        English      → no-op passthrough
    """
    confidence_threshold = config.get("confidence_threshold", 0.5)
    preprocessor = AudioPreprocessor()
    asr = WhisperASR(
        model_size="large-v3",
        confidence_threshold=confidence_threshold,
    )
    translator = Translator()
    detector = KeywordDetector(config["keywords"])
    analyzer = ContextualAnalyzer(model=CONTEXTUAL_MODEL)
    return preprocessor, asr, translator, detector, analyzer, confidence_threshold

def process_file(
    audio_file: str,
    preprocessor: AudioPreprocessor,
    asr: WhisperASR,
    translator: Translator,
    detector: KeywordDetector,
    analyzer: ContextualAnalyzer,
) -> dict:
    preprocessed_path = None
    try:
        validate_audio_path(audio_file)
        preprocessed_path = preprocessor.preprocess(audio_file)
        asr_result = asr.transcribe(preprocessed_path)
        source_lang = asr_result.get("language", "en")
        segments = asr_result.get("segments", [])
        asr_model = asr_result.get("asr_model", "unknown")
        confidences = [
            s["confidence"] for s in segments
            if s.get("confidence") is not None
        ]
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else None
        )
        conf_str = f"{avg_confidence:.4f}" if avg_confidence is not None else "N/A (Indic path)"
        logger.info(
            f"Language: {source_lang} | ASR model: {asr_model} | "
            f"Avg confidence: {conf_str}"
        )
        raw_transcript = asr_result["transcript"]
        if source_lang == "en":
            english_transcript = raw_transcript
            logger.info("English audio — IndicTrans2 translation step skipped.")
        elif source_lang in INDIC_LANGUAGES:
            english_transcript = translator.translate_to_english(
                raw_transcript, source_lang
            )
            logger.info(
                f"[IndicTrans2 {source_lang}→en] Original  : {raw_transcript[:80]}"
            )
            logger.info(
                f"[IndicTrans2 {source_lang}→en] Translated: {english_transcript[:80]}"
            )
        else:
            logger.warning(
                f"Language '{source_lang}' is not English or a supported "
                "Indic language. Passing transcript as-is to normalization."
            )
            english_transcript = raw_transcript
        normalized_text = TextNormalizer.normalize(english_transcript)
        if not normalized_text:
            return {
                "input_file": audio_file,
                "detected_language": source_lang,
                "asr_model": asr_model,
                "original_transcript": raw_transcript,
                "translated_transcript": "",
                "normalized_transcript": "",
                "total_detected_keywords": 0,
                "detected_keywords": [],
                "contextual_meaning": "",
            }
        detections = detector.detect(normalized_text)
        unique_keywords = list(dict.fromkeys(d["keyword"] for d in detections))
        total_keyword_hits = sum(d["count"] for d in detections)
        try:
            contextual_meaning = analyzer.analyze(normalized_text, unique_keywords)
        except ContextualAnalyzerError as e:
            logger.error(f"Contextual analysis failed: {e}")
            contextual_meaning = ""
        return {
            "input_file": audio_file,
            "detected_language": source_lang,
            "asr_model": asr_model,
            "original_transcript": raw_transcript,
            "translated_transcript": english_transcript,
            "normalized_transcript": normalized_text,
            "total_detected_keywords": total_keyword_hits,
            "detected_keywords": unique_keywords,
            "contextual_meaning": contextual_meaning,
        }
    finally:
        if preprocessed_path:
            AudioPreprocessor.cleanup(preprocessed_path)

def main(audio_file: str):
    setup_logging(log_dir=LOG_DIR)
    try:
        config = load_config(CONFIG_PATH)
        preprocessor, asr, translator, detector, analyzer, _ = (
            build_pipeline(config)
        )
        result = process_file(
            audio_file, preprocessor, asr, translator, detector, analyzer
        )
        output_path = JSONLogger.save(OUTPUT_DIR, result)
        print(json.dumps(result, indent=4, ensure_ascii=False))
        print(f"\n[Saved] {output_path}", file=sys.stderr)
    except (AudioPreprocessorError, WhisperASRError,
            KeywordDetectorError, TranslatorError) as e:
        logger.error(f"Component error: {e}")
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"[ERROR] Unexpected failure: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    INPUT_AUDIO = r"audio_input/audio8.m4a" # Input audio path
    main(INPUT_AUDIO)
