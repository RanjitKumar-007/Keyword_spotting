import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Default Ollama REST endpoint (generate API, non-streaming).
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Must exactly match the tag shown by "ollama list".
DEFAULT_MODEL = "qwen2.5:7b"

class ContextualAnalyzerError(Exception):
    pass


class ContextualAnalyzer:
    """
    Adds an LLM-based contextual-understanding pass AFTER keyword spotting.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_url: str = OLLAMA_API_URL,
        timeout: int = 120,
        temperature: float = 0.2,
    ):
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.temperature = temperature
        logger.info(
            f"ContextualAnalyzer ready — model: {self.model}, endpoint: {self.api_url}"
        )

    @staticmethod
    def _build_prompt(transcript: str, detected_keywords: list[str]) -> str:
        keyword_list = ", ".join(detected_keywords) if detected_keywords else "none"
        return (
            "You are assisting a security monitoring pipeline. A speech "
            "transcript has already been scanned by a keyword spotter, "
            f"which flagged the following terms: {keyword_list}.\n\n"
            "Transcript:\n"
            f"\"{transcript}\"\n\n"
            "In 3-5 sentences, summarize what the transcript is most "
            "likely describing and explain how the flagged keywords fit "
            "into that context. Base your summary only on what the "
            "transcript actually says — do not invent specifics, names, "
            "locations, or outcomes that are not present in the text. "
            "Respond with plain prose only, no headings, no bullet points, "
            "no preamble like 'Here is a summary'."
        )

    def analyze(self, transcript: str, detected_keywords: list[str]) -> str:
        """
        Returns a short contextual-meaning paragraph.
        """
        if not transcript or not transcript.strip():
            return ""
        prompt = self._build_prompt(transcript, detected_keywords)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        logger.info(f"Requesting contextual meaning from '{self.model}' via Ollama...")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise ContextualAnalyzerError(
                f"Could not reach Ollama at {self.api_url}: {e}. "
                "Is 'ollama serve' running and is the model pulled "
                f"('ollama list' should show '{self.model}')?"
            ) from e
        except TimeoutError as e:
            raise ContextualAnalyzerError(
                f"Ollama request timed out after {self.timeout}s: {e}"
            ) from e
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ContextualAnalyzerError(f"Invalid JSON from Ollama: {e}") from e
        if "error" in body:
            raise ContextualAnalyzerError(f"Ollama returned an error: {body['error']}")
        meaning = (body.get("response") or "").strip()
        if not meaning:
            raise ContextualAnalyzerError("Ollama returned an empty response.")
        logger.info(f"Contextual meaning generated ({len(meaning)} chars).")
        return meaning
    