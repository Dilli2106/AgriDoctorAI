
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

KNOWLEDGE_FILE = (
    BASE_DIR
    / "app"
    / "knowledge_base"
    / "agricultural_knowledge.json"
)


def load_knowledge_base():
    """Load the agricultural knowledge JSON file."""

    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


KNOWLEDGE_BASE = load_knowledge_base()


def get_disease_information(crop_id, disease_name, language="en"):
    """
    Retrieve bilingual disease information.

    crop_id: plantvillage, rice, or sugarcane
    disease_name: exact predicted class name
    language: en or ta
    """

    if language not in ["en", "ta"]:
        raise ValueError("Language must be 'en' or 'ta'.")

    crops = KNOWLEDGE_BASE.get("crops", {})

    if crop_id not in crops:
        raise ValueError(f"Unknown crop: {crop_id}")

    diseases = crops[crop_id].get("diseases", {})

    if disease_name not in diseases:
        raise ValueError(
            f"Knowledge not found for disease: {disease_name}"
        )

    disease_data = diseases[disease_name]

    sections = [
        "symptoms",
        "causes",
        "prevention",
        "medicine",
        "recovery"
    ]

    result = {
        "crop_id": crop_id,
        "disease": disease_name,
        "language": language
    }

    for section in sections:
        result[section] = disease_data.get(section, {}).get(
            language, []
        )

    return result
