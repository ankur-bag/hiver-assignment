"""
Multilingual Language Service.
Provides:
1. Fast rule-based and vocabulary-based language detection (English, Hindi, Hinglish, Spanish, French, German).
2. Language normalization and prompt formatting directives.
"""

import re
from typing import Tuple


LANGUAGE_DISPLAY_NAMES = {
    "en": "English",
    "hi": "Hindi (Devanagari script)",
    "hi-Latn": "Hinglish (Hindi written in Roman script)",
    "es": "Spanish",
    "de": "German",
    "fr": "French"
}


def detect_language(text: str) -> str:
    """
    Detects the primary language of the customer query.

    Args:
        text: Raw user query string.

    Returns:
        Language code: 'en', 'hi', 'hi-Latn', 'es', 'de', 'fr'.
    """
    if not text or not isinstance(text, str):
        return "en"

    text_lower = text.lower()

    # 1. Devanagari script detection
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    # 2. Hinglish (Romanized Hindi) vocabulary
    hinglish_words = {"mera", "meri", "kya", "nahi", "aaya", "karo", "bhai", "hai", "mujhe", "kab", "milega", "hoga", "apna", "kar", "diya"}
    words = set(re.findall(r"\b\w+\b", text_lower))
    if len(words.intersection(hinglish_words)) >= 2:
        return "hi-Latn"

    # 3. Spanish markers & accent letters
    spanish_words = {"el", "la", "de", "que", "en", "por", "paquete", "pedido", "llegado", "reembolso", "cuenta", "donde", "esta", "mi"}
    if len(words.intersection(spanish_words)) >= 2 or any(c in text for c in "áéíóúñ¿¡"):
        return "es"

    # 4. German markers & umlauts
    german_words = {"und", "der", "die", "das", "nicht", "bestellung", "paket", "konto", "bitte", "lieferung", "wo", "ist", "mein"}
    if len(words.intersection(german_words)) >= 2 or any(c in text for c in "äöüß"):
        return "de"

    # 5. French markers & accents
    french_words = {"le", "la", "les", "des", "pour", "mon", "colis", "livraison", "compte", "remboursement", "où", "est"}
    if len(words.intersection(french_words)) >= 2 or any(c in text for c in "àâéèêëîïôùûç"):
        return "fr"

    return "en"


def get_language_prompt_instruction(lang_code: str) -> str:
    """
    Returns explicit instruction for Gemini on how to format the target language.
    """
    if lang_code == "hi-Latn":
        return "natural, polite Hinglish (Hindi written in the Roman/English alphabet, e.g. 'Namaste, aapka order...')"
    if lang_code == "hi":
        return "fluent Hindi in Devanagari script (हिंदी)"
    if lang_code == "es":
        return "fluent, professional Spanish (Español)"
    if lang_code == "de":
        return "fluent, professional German (Deutsch)"
    if lang_code == "fr":
        return "fluent, professional French (Français)"
    return "clear, professional English"
