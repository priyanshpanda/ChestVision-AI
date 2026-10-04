from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# MobileNetV2's final convolutional feature layer.
TARGET_LAYER_NAME = "Conv_1"


def generate_gradcam(model, image_tensor):
    """
    Generate a Grad-CAM heatmap for a single chest X-ray.

    Parameters
    ----------
    model : tf.keras.Model
        Trained ChestVision AI model.

    image_tensor : tf.Tensor
        Preprocessed image with shape (1, 224, 224, 3).

    Returns
    -------
    heatmap : np.ndarray
        Normalized Grad-CAM heatmap with values between 0 and 1.

    probability : float
        Pneumonia probability produced by the model.
    """

    # Get the nested MobileNetV2 model.
    backbone = model.get_layer("mobilenetv2_1.00_224")

    # Get MobileNetV2's final convolutional layer.
    conv_layer = backbone.get_layer(TARGET_LAYER_NAME)

    # We use a temporary hook so that we capture the actual
    # convolutional activation from the original model forward pass.
    captured = {}

    original_conv_call = conv_layer.call

    def hooked_conv_call(*args, **kwargs):
        output = original_conv_call(*args, **kwargs)
        captured["activation"] = output
        return output

    conv_layer.call = hooked_conv_call

    try:
        # Record the operations needed for Grad-CAM.
        with tf.GradientTape() as tape:

            # Run the ORIGINAL model.
            prediction = model(image_tensor, training=False)

            # Retrieve the activation captured from Conv_1.
            conv_activation = captured["activation"]

            # We want the pneumonia probability.
            pneumonia_score = prediction[:, 0]

        # Calculate gradients of the pneumonia score
        # with respect to the convolutional activation.
        gradients = tape.gradient(
            pneumonia_score,
            conv_activation
        )

    finally:
        # Always restore the original layer function.
        conv_layer.call = original_conv_call

    # Convert tensors to NumPy arrays.
    activation = conv_activation[0]
    gradients = gradients[0]

    # Global-average-pool the gradients.
    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1)
    )

    # Weighted combination of feature maps.
    cam = tf.reduce_sum(
        activation * weights,
        axis=-1
    )

    # Keep only positive activation.
    cam = tf.maximum(cam, 0)

    # Normalize to 0–1.
    cam_max = tf.reduce_max(cam)

    if float(cam_max) > 0:
        cam = cam / cam_max

    heatmap = cam.numpy()

    probability = float(prediction[0, 0])

    return heatmap, probability


def create_overlay(image_tensor, heatmap, alpha=0.4):
    """
    Create a Grad-CAM overlay on the original X-ray.

    Parameters
    ----------
    image_tensor : tf.Tensor
        Image tensor with shape (1, 224, 224, 3).

    heatmap : np.ndarray
        Grad-CAM heatmap.

    alpha : float
        Heatmap transparency.

    Returns
    -------
    overlay : np.ndarray
        RGB image containing the Grad-CAM overlay.
    """

    # Convert TensorFlow image to NumPy.
    image = image_tensor[0].numpy()

    # Convert floating-point image to uint8.
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Resize Grad-CAM from 7x7 to 224x224.
    heatmap_resized = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )

    # Convert heatmap to 0–255.
    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    # Apply OpenCV color map.
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # OpenCV uses BGR; convert to RGB.
    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Blend original image and heatmap.
    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay