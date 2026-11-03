
import json
from pathlib import Path
import tensorflow as tf

# Project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Configuration file
CONFIG_FILE = (
    BASE_DIR / "app" / "config" / "crop_config.json"
)

# Read model configuration
with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    APP_CONFIG = json.load(file)

# Store loaded models to avoid loading repeatedly
MODEL_CACHE = {}


def load_crop_model(crop_id):
    """
    Load the correct trained model for the selected crop.
    """

    if crop_id not in APP_CONFIG["models"]:
        raise ValueError(
            f"Unknown crop selection: {crop_id}"
        )

    # Return cached model if already loaded
    if crop_id in MODEL_CACHE:
        return MODEL_CACHE[crop_id]

    model_path = (
        BASE_DIR / APP_CONFIG["models"][crop_id]["model_path"]
    ).resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    # Load trained Keras model
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False
    )

    # Verify output classes
    expected_classes = APP_CONFIG["models"][crop_id]["num_classes"]
    actual_classes = model.output_shape[-1]

    if actual_classes != expected_classes:
        raise ValueError(
            f"Class mismatch for {crop_id}: "
            f"expected {expected_classes}, got {actual_classes}"
        )

    MODEL_CACHE[crop_id] = model

    return model


def get_crop_classes(crop_id):
    """Return disease classes for the selected crop."""

    if crop_id not in APP_CONFIG["models"]:
        raise ValueError(f"Unknown crop: {crop_id}")

    return APP_CONFIG["models"][crop_id]["class_names"]


def get_crop_information(crop_id):
    """Return crop configuration information."""

    if crop_id not in APP_CONFIG["models"]:
        raise ValueError(f"Unknown crop: {crop_id}")

    return APP_CONFIG["models"][crop_id]
