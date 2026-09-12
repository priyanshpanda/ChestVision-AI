from pathlib import Path
import tensorflow as tf


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Trained model path
MODEL_PATH = PROJECT_ROOT / "models" / "best_mobilenetv2.keras"


def load_model():
    """
    Load the trained ChestVision AI MobileNetV2 model.

    Returns
    -------
    tf.keras.Model
        Trained classification model.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    return model