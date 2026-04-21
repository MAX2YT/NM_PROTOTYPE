"""
tamil/translator.py — Tamil ↔ English translation for Neuro-Vault.

Provides two layers of translation:
  1. ``deep-translator`` (GoogleTranslator) — online, used when available
  2. Offline dictionary fallback for common medical Tamil phrases

In a hospital network without internet access the offline fallback ensures
the system remains functional.  Detection of Tamil script uses Unicode range.

Tamil Unicode range: U+0B80 – U+0BFF
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Offline fallback dictionary
#  (common clinical Tamil ↔ English phrase pairs)
# ------------------------------------------------------------------ #

_OFFLINE_TAM_TO_EN: dict[str, str] = {
    "நீரிழிவு நோய்": "diabetes mellitus",
    "நீரிழிவு": "diabetes",
    "சிகிச்சை": "treatment",
    "சிகிச்சை முறை": "treatment method",
    "இரத்த அழுத்தம்": "blood pressure",
    "உயர் இரத்த அழுத்தம்": "hypertension",
    "இதய நோய்": "heart disease",
    "இதய செயலிழப்பு": "heart failure",
    "காசநோய்": "tuberculosis",
    "டெங்கு காய்ச்சல்": "dengue fever",
    "மலேரியா": "malaria",
    "கொலரா": "cholera",
    "டைஃபாய்டு": "typhoid",
    "மஞ்சள் காமாலை": "jaundice",
    "சுவாச நோய்": "respiratory disease",
    "நுரையீரல்": "lung",
    "சிறுநீரகம்": "kidney",
    "கல்லீரல்": "liver",
    "அறுவை சிகிச்சை": "surgery",
    "மருந்து": "medicine",
    "மருத்துவர்": "doctor",
    "மருத்துவமனை": "hospital",
    "நோயாளி": "patient",
    "அறிகுறிகள்": "symptoms",
    "கண்டறிதல்": "diagnosis",
    "தடுப்பூசி": "vaccine",
    "ரத்த பரிசோதனை": "blood test",
    "உடல் பருமன்": "obesity",
    "புற்றுநோய்": "cancer",
    "தலைவலி": "headache",
    "காய்ச்சல்": "fever",
    "இருமல்": "cough",
    "மூச்சு திணறல்": "breathlessness",
    "வலி": "pain",
    "வாந்தி": "vomiting",
    "வயிற்றுப்போக்கு": "diarrhea",
    "மயக்கம்": "dizziness",
}

_OFFLINE_EN_TO_TAM: dict[str, str] = {v: k for k, v in _OFFLINE_TAM_TO_EN.items()}


class TamilTranslator:
    """Translate between Tamil and English for clinical queries.

    Attempts online translation via ``deep-translator`` (GoogleTranslator)
    and falls back to offline dictionary lookup if the network is
    unavailable or the library raises an error.

    Args:
        use_online: Try online translation engine first (default True).
    """

    def __init__(self, use_online: bool = True) -> None:
        self.use_online = use_online
        self._online_available: Optional[bool] = None   # tested lazily

    # ------------------------------------------------------------------ #
    #  Detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_tamil(text: str) -> bool:
        """Return True if *text* contains Tamil Unicode characters.

        Args:
            text: Any string.

        Returns:
            ``True`` if any character falls in Tamil Unicode block
            (U+0B80–U+0BFF).
        """
        return any(
            "\u0B80" <= ch <= "\u0BFF"
            for ch in text
        )

    # ------------------------------------------------------------------ #
    #  Tamil → English
    # ------------------------------------------------------------------ #

    def tamil_to_english(self, text: str) -> str:
        """Translate Tamil text to English.

        Args:
            text: Tamil-script query or phrase.

        Returns:
            English translation string.  Returns *text* unchanged if
            no translation can be obtained.
        """
        if not text.strip():
            return text

        # Try online first
        if self.use_online and self._check_online():
            translated = self._online_translate(text, src="ta", dest="en")
            if translated:
                logger.debug("Online translation: '%s' → '%s'", text[:40], translated[:40])
                return translated

        # Offline dictionary fallback
        return self._offline_translate(text, _OFFLINE_TAM_TO_EN)

    # ------------------------------------------------------------------ #
    #  English → Tamil
    # ------------------------------------------------------------------ #

    def english_to_tamil(self, text: str) -> str:
        """Translate English text to Tamil.

        Args:
            text: English string.

        Returns:
            Tamil translation string.  Returns *text* unchanged on failure.
        """
        if not text.strip():
            return text

        if self.use_online and self._check_online():
            translated = self._online_translate(text, src="en", dest="ta")
            if translated:
                return translated

        return self._offline_translate(text, _OFFLINE_EN_TO_TAM)

    # ------------------------------------------------------------------ #
    #  Online translation helper
    # ------------------------------------------------------------------ #

    def _online_translate(
        self, text: str, src: str, dest: str
    ) -> Optional[str]:
        """Attempt translation using ``deep-translator`` GoogleTranslator.

        Args:
            text: Source text.
            src:  ISO language code of source (e.g. ``"ta"``).
            dest: ISO language code of destination (e.g. ``"en"``).

        Returns:
            Translated string, or ``None`` on any error.
        """
        try:
            from deep_translator import GoogleTranslator  # type: ignore
            result = GoogleTranslator(source=src, target=dest).translate(text)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Online translation failed: %s", exc)
            self._online_available = False   # Disable for session
            return None

    # ------------------------------------------------------------------ #
    #  Offline dictionary fallback
    # ------------------------------------------------------------------ #

    @staticmethod
    def _offline_translate(text: str, dictionary: dict[str, str]) -> str:
        """Apply dictionary-based phrase substitution.

        Iterates over known phrases longest-first and replaces matches.

        Args:
            text:       Source text.
            dictionary: Mapping of source phrases to target phrases.

        Returns:
            String with substitutions applied; original text if no match.
        """
        result = text
        for phrase in sorted(dictionary.keys(), key=len, reverse=True):
            if phrase.lower() in result.lower():
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                result = pattern.sub(dictionary[phrase], result)
        return result

    # ------------------------------------------------------------------ #
    #  Online availability check
    # ------------------------------------------------------------------ #

    def _check_online(self) -> bool:
        """Test whether the online translation service is reachable.

        Caches the result for the session lifetime.

        Returns:
            ``True`` if a network translation request can be made.
        """
        if self._online_available is not None:
            return self._online_available

        try:
            from deep_translator import GoogleTranslator  # type: ignore
            # Quick probe with a short string
            GoogleTranslator(source="en", target="ta").translate("test")
            self._online_available = True
        except Exception:  # noqa: BLE001
            self._online_available = False

        return self._online_available
