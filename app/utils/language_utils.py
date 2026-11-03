
import json
from pathlib import Path

# Find translation file
BASE_DIR = Path(__file__).resolve().parents[2]

TRANSLATION_FILE = (
    BASE_DIR / "app" / "translations" / "translations.json"
)

# Load translations
with open(TRANSLATION_FILE, "r", encoding="utf-8") as file:
    TRANSLATIONS = json.load(file)


def get_text(key, language="en"):
    """
    Return translated text for the selected language.
    """

    if language not in TRANSLATIONS:
        language = "en"

    return TRANSLATIONS[language].get(
        key,
        TRANSLATIONS["en"].get(key, key)
    )


def get_available_languages():
    """Return available application languages."""

    return list(TRANSLATIONS.keys())
