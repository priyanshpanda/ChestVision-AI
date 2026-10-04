from pathlib import Path
import tensorflow as tf


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Trained model
MODEL_PATH = PROJECT_ROOT / "models" / "best_mobilenetv2.keras"

# Classification threshold selected using validation data
FINAL_THRESHOLD = 0.55


def load_model():
    """Load the trained ChestVision AI model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


def predict_image(image_path, model):
    """
    Predict NORMAL or PNEUMONIA for a chest X-ray.

    Parameters
    ----------
    image_path : str or Path
        Path to the chest X-ray image.

    model : tf.keras.Model
        Loaded ChestVision AI model.

    Returns
    -------
    dict
        Prediction result.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Read image
    image_bytes = tf.io.read_file(str(image_path))

    # Decode JPEG as RGB
    image = tf.image.decode_jpeg(
        image_bytes,
        channels=3
    )

    # Resize to model input size
    image = tf.image.resize(
        image,
        [224, 224]
    )

    # Add batch dimension
    image = tf.expand_dims(
        image,
        axis=0
    )

    # Model prediction
    probability = float(
        model(image, training=False)[0, 0]
    )

    # Apply locked validation-selected threshold
    if probability >= FINAL_THRESHOLD:
        predicted_class = "PNEUMONIA"
    else:
        predicted_class = "NORMAL"

    return {
        "class": predicted_class,
        "pneumonia_probability": probability,
        "threshold": FINAL_THRESHOLD
    }