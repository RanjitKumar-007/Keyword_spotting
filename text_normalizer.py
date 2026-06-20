import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

class TextNormalizer:
    """
    Normalizes transcribed text before keyword detection.
    """

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        nfd = unicodedata.normalize("NFD", text)
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    @classmethod
    def normalize(cls, text: str) -> str:
        if not text or not text.strip():
            logger.warning("Empty text passed to normalizer.")
            return ""
        text = text.lower()                      # step 1: lowercase
        text = cls._normalize_unicode(text)      # step 2: strip accents
        text = re.sub(r"[^\w\s\-]", " ", text)   # step 3: punctuation → space (keep hyphens)
        text = re.sub(r"\s+", " ", text).strip() # step 4: collapse whitespace
        logger.debug(f"Normalized: {text[:120]}")
        return text
    