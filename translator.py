import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)

# Indian languages supported
INDIC_LANGUAGES = {"hi", "ta", "kn", "te", "ml", "bn", "gu", "mr", "pa", "ur"}

INDIC_TRANS2_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"

LANG_TO_FLORES = {
    "hi": "hin_Deva", "ta": "tam_Taml", "kn": "kan_Knda",
    "te": "tel_Telu", "ml": "mal_Mlym", "bn": "ben_Beng",
    "gu": "guj_Gujr", "mr": "mar_Deva", "pa": "pan_Guru",
    "ur": "urd_Arab",
}
FLORES_ENGLISH = "eng_Latn"

class TranslatorError(Exception):
    pass

class Translator:
    """
    Translates Indian-language text to English using IndicTrans2.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir   = cache_dir
        self._device      = "cuda" if torch.cuda.is_available() else "cpu"
        self._it2_model   = None
        self._it2_tok     = None
        self._it2_proc    = None
        self._it2_toolkit = False
        logger.info(
            f"Translator initialised — cache: {self._cache_dir or 'default (HF resolved)'}, "
            f"device: {self._device}"
        )

    def _try_load_indictrans2(self) -> bool:
        """Try to load IndicTrans2 from local cache. Returns True on success."""
        if self._it2_model is not None:
            return True
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            logger.info(f"Trying IndicTrans2 from cache: {self._cache_dir or 'default'} ...")
            tok = AutoTokenizer.from_pretrained(
                INDIC_TRANS2_MODEL,
                trust_remote_code=True,
                cache_dir=self._cache_dir,
                local_files_only=True,
            )
            mdl = AutoModelForSeq2SeqLM.from_pretrained(
                INDIC_TRANS2_MODEL,
                trust_remote_code=True,
                cache_dir=self._cache_dir,
                local_files_only=True,
            ).to(self._device)
            mdl.eval()
            try:
                from IndicTransToolkit import IndicProcessor
                self._it2_proc    = IndicProcessor(inference=True)
                self._it2_toolkit = True
                logger.info("IndicTrans2 loaded with IndicTransToolkit.")
            except ImportError:
                self._it2_proc    = None
                self._it2_toolkit = False
                logger.info(
                    "IndicTrans2 loaded (no IndicTransToolkit — run "
                    "'pip install IndicTransToolkit' in this environment "
                    "for proper preprocessing/normalization and better "
                    "translation quality)."
                )
            self._it2_tok   = tok
            self._it2_model = mdl
            return True
        except Exception as e:
            logger.error(
                f"IndicTrans2 not available in cache: {e}. "
                "No fallback configured — original text will be returned."
            )
            return False
            

    def _translate_indictrans2(self, text: str, source_lang: str) -> str:
        src_flores = LANG_TO_FLORES[source_lang]
        if self._it2_toolkit and self._it2_proc:
            batch  = self._it2_proc.preprocess_batch(
                [text], src_lang=src_flores, tgt_lang=FLORES_ENGLISH
            )
            inputs = self._it2_tok(
                batch, truncation=True, padding="longest",
                return_tensors="pt", return_attention_mask=True,
            ).to(self._device)
            with torch.no_grad():
                outputs = self._it2_model.generate(
                    **inputs, num_beams=5, num_return_sequences=1, max_length=512
                )
            decoded = self._it2_tok.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            return self._it2_proc.postprocess_batch(decoded, lang=FLORES_ENGLISH)[0]
        else:
            tagged  = f"{src_flores} {FLORES_ENGLISH} {text}"
            inputs  = self._it2_tok(
                tagged, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            ).to(self._device)
            with torch.no_grad():
                outputs = self._it2_model.generate(**inputs, num_beams=5, max_length=512)
            return self._it2_tok.decode(outputs[0], skip_special_tokens=True)

    
    def translate_to_english(self, text: str, source_lang: str) -> str:
        """
        Translate Indian-language text to English using IndicTrans2.
        """
        if not text or not text.strip():
            return text
        if source_lang == "en":
            logger.info("Source is English — skipping translation.")
            return text
        if source_lang not in INDIC_LANGUAGES:
            logger.warning(
                f"Language '{source_lang}' not in supported Indic set — "
                "returning original text."
            )
            return text
        if self._try_load_indictrans2():
            try:
                translated = self._translate_indictrans2(text, source_lang)
                logger.info(f"[IndicTrans2 {source_lang}→en] {translated[:120]}")
                return translated
            except Exception as e:
                logger.error(f"IndicTrans2 translation failed ({e}).")
        logger.error(
            f"IndicTrans2 unavailable for [{source_lang}]. "
            "Returning original text — keyword detection will be unreliable."
        )
        return text
        