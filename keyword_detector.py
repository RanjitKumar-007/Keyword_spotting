import logging
from flashtext import KeywordProcessor

logger = logging.getLogger(__name__)

class KeywordDetectorError(Exception):
    pass

class KeywordDetector:
    """
    Detects keywords and multi-word synonyms using FlashText.
    """

    def __init__(self, keyword_config: dict):
        if not keyword_config:
            raise KeywordDetectorError("keyword_config is empty.")
        self.keyword_config = keyword_config
        self.processor = KeywordProcessor(case_sensitive=False)
        loaded = 0
        for keyword, data in keyword_config.items():
            synonyms = data.get("synonyms", [])
            self.processor.add_keyword(keyword, keyword)
            for syn in synonyms:
                if syn.strip():
                    self.processor.add_keyword(syn.strip(), keyword)
            loaded += 1
        logger.info(
            f"KeywordDetector ready — {loaded} keywords, "
            f"{sum(len(v.get('synonyms',[])) for v in keyword_config.values())} synonyms indexed."
        )

    def detect(self, text: str) -> list[dict]:
        if not text:
            return []
        matches = self.processor.extract_keywords(text, span_info=True)
        counts: dict[str, int] = {}
        matched_terms: dict[str, list[str]] = {}
        for canonical, start, end in matches:
            counts[canonical] = counts.get(canonical, 0) + 1
            matched_terms.setdefault(canonical, []).append(text[start:end])
        detections = []
        for keyword, count in counts.items():
            weight = self.keyword_config[keyword]["weight"]
            detections.append({
                "keyword": keyword,
                "count": count,
                "weight": weight,
                "matched_terms": list(set(matched_terms[keyword])),
            })
        detections.sort(key=lambda x: x["weight"] * x["count"], reverse=True)
        logger.info(f"Detected {len(detections)} unique keyword(s).")
        return detections
