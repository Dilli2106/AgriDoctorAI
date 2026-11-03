
import numpy as np
import tensorflow as tf
from PIL import Image


def preprocess_leaf_image(image):
    """
    Prepare uploaded image for MobileNetV2.
    """

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = (
        tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def predict_disease(image, model, class_names):
    """
    Predict disease from an uploaded leaf image.
    """

    processed_image = preprocess_leaf_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_disease = class_names[predicted_index]

    model_score = float(probabilities[predicted_index])

    return {
        "disease": predicted_disease,
        "model_score": model_score,
        "all_scores": probabilities.tolist()
    }
