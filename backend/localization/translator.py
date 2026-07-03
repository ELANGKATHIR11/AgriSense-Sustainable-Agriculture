import logging

logger = logging.getLogger("Translator")

# Multi-lingual crop search mapping dictionary
CROP_DICTIONARY = {
    "rice": {"en": "Rice", "ta": "அரிசி", "te": "వరి", "ml": "അരി", "hi": "चावल"},
    "wheat": {"en": "Wheat", "ta": "கோதுமை", "te": "గోధుమలు", "ml": "ഗോതമ്പ്", "hi": "गेहूं"},
    "tomato": {
        "en": "Tomato",
        "ta": "தக்காளி",
        "te": "టమోటా",
        "ml": "തക്കാളി",
        "hi": "टमाटर",
    },
    "potato": {
        "en": "Potato",
        "ta": "உருளைக்கிழங்கு",
        "te": "బంగాళాదుంప",
        "ml": "ഉരുളക്കിഴങ്ങ്",
        "hi": "आलू",
    },
    "onion": {"en": "Onion", "ta": "வெங்காயம்", "te": "ఉల్లిపాయ", "ml": "സവാള", "hi": "प्याज"},
    "cotton": {
        "en": "Cotton",
        "ta": "பருத்தி",
        "te": "పత్తి",
        "ml": "പരുത്തി",
        "hi": "कपास",
    },
    "sugarcane": {
        "en": "Sugarcane",
        "ta": "கரும்பு",
        "te": "చెరకు",
        "ml": "കരിമ്പ്",
        "hi": "गन्ना",
    },
    "mustard": {
        "en": "Mustard",
        "ta": "கடுகு",
        "te": "ఆవాలు",
        "ml": "കടുക്",
        "hi": "सरसों",
    },
    "soybean": {
        "en": "Soybean",
        "ta": "சோயாபீன்ஸ்",
        "te": "సోయాబీన్",
        "ml": "സോയാബീൻ",
        "hi": "सोयाबीन",
    },
}

# General UI translation dictionaries for simple server-side translations
PHRASE_DICTIONARY = {
    "Verified": {
        "en": "Verified",
        "ta": "சரிபார்க்கப்பட்டது",
        "te": "ధృవీకరించబడింది",
        "ml": "സ്ഥിരീകരിച്ചു",
        "hi": "सत्यापित",
    },
    "Likely": {
        "en": "Likely",
        "ta": "சாத்தியமான",
        "te": "బహుశా",
        "ml": "സാധ്യതയുള്ള",
        "hi": "संभावित",
    },
    "Estimated": {
        "en": "Estimated",
        "ta": "மதிப்பிடப்பட்டது",
        "te": "అంచనా వేయబడింది",
        "ml": "കണക്കാക്കപ്പെടുന്നു",
        "hi": "अनुमानित",
    },
    "Low confidence": {
        "en": "Low confidence",
        "ta": "குறைந்த நம்பிக்கை",
        "te": "తక్కువ విశ్వాసం",
        "ml": "കുറഞ്ഞ വിശ്വാസ്യത",
        "hi": "कम आत्मविश्वास",
    },
}

# Translation cache to avoid translating identical strings multiple times
_translation_cache = {}


def get_normalized_crop_name(name: str) -> str:
    """
    Search crop name in all supported languages and return the English equivalent.
    """
    if not name:
        return ""
    clean_name = name.strip().lower()
    for english_key, translations in CROP_DICTIONARY.items():
        if clean_name == english_key:
            return translations["en"]
        for val in translations.values():
            if val.lower() == clean_name:
                return translations["en"]
    return name


def translate_crop_name(name: str, target_lang: str) -> str:
    """
    Translate English or alternative language crop name to the target language.
    """
    english_name = get_normalized_crop_name(name)
    clean_eng = english_name.lower()
    if clean_eng in CROP_DICTIONARY:
        return CROP_DICTIONARY[clean_eng].get(target_lang, english_name)
    return name


def translate_text(text: str, target_lang: str) -> str:
    """
    Translate raw phrases, updates, or alerts.
    Implements translation caching to save execution runtime.
    """
    if not text:
        return ""

    cache_key = f"{text}:{target_lang}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    # Match basic terms
    for term, trans in PHRASE_DICTIONARY.items():
        if term.lower() in text.lower():
            translated = trans.get(target_lang, term)
            _translation_cache[cache_key] = translated
            return translated

    # Crop translation replacement inside textual strings
    translated_text = text
    for eng_crop, langs in CROP_DICTIONARY.items():
        lang_val = langs.get(target_lang)
        if lang_val:
            # Replace crop name safely
            pattern = re.compile(rf"\b{re.escape(eng_crop)}\b", re.IGNORECASE)
            translated_text = pattern.sub(lang_val, translated_text)

    _translation_cache[cache_key] = translated_text
    return translated_text


import re
