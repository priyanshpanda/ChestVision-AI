import tempfile

import streamlit as st
import tensorflow as tf

from src.prediction import load_model, predict_image
from src.gradcam import generate_gradcam, create_overlay


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ChestVision AI",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():
    return load_model()


model = get_model()


# ============================================================
# HEADER
# ============================================================

st.title("ChestVision AI")
st.subheader("AI-Powered Chest X-ray Disease Detection")

st.write(
    """
    Upload a chest X-ray image to analyze it using a trained
    MobileNetV2 deep learning model.
    """
)

st.info(
    "Research prototype — this system is not intended to provide "
    "a clinical diagnosis."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # Read uploaded image bytes.
    image_bytes = uploaded_file.getvalue()

    # Display original image.
    st.subheader("Uploaded X-ray")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image_bytes,
            caption="Original Chest X-ray",
            use_container_width=True
        )

    # Save temporary copy because prediction.py
    # currently accepts an image file path.
    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as temp_file:

        temp_file.write(image_bytes)
        temp_path = temp_file.name

    # ========================================================
    # PREDICTION
    # ========================================================

    with st.spinner("Analyzing X-ray..."):

        result = predict_image(
            temp_path,
            model
        )

    predicted_class = result["class"]
    probability = result["pneumonia_probability"]
    threshold = result["threshold"]

    # ========================================================
    # PREPARE IMAGE FOR GRAD-CAM
    # ========================================================

    image = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [224, 224]
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    with col2:

        st.subheader("Prediction")

        if predicted_class == "PNEUMONIA":
            st.error(
                f"Prediction: {predicted_class}"
            )
        else:
            st.success(
                f"Prediction: {predicted_class}"
            )

        st.metric(
            "Pneumonia Probability",
            f"{probability * 100:.2f}%"
        )

        st.write(
            f"Classification threshold: {threshold:.2f}"
        )

        st.progress(
            min(probability, 1.0)
        )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.divider()

    st.subheader("Grad-CAM Explainability")

    with st.spinner("Generating Grad-CAM heatmap..."):

        heatmap, gradcam_probability = generate_gradcam(
            model,
            image
        )

        overlay = create_overlay(
            image,
            heatmap
        )

    grad_col1, grad_col2 = st.columns(2)

    with grad_col1:

        st.image(
            image[0].numpy().astype("uint8"),
            caption="Preprocessed X-ray",
            use_container_width=True
        )

    with grad_col2:

        st.image(
            overlay,
            caption="Grad-CAM Heatmap Overlay",
            use_container_width=True
        )

    st.caption(
        "Grad-CAM highlights image regions that contributed to "
        "the model's prediction. The visualization should not be "
        "interpreted as a clinically validated lesion map."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ChestVision AI | MobileNetV2 Transfer Learning | "
    "Research/Academic Prototype"
)