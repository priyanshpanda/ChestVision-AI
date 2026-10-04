import sys
from pathlib import Path

# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import tempfile

import streamlit as st
import tensorflow as tf

from src.prediction import load_model, predict_image
from src.gradcam import generate_gradcam, create_overlay


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="ChestVision AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

        .main-title {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.15rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }

        .info-box {
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid rgba(128, 128, 128, 0.2);
            margin-top: 1rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 650;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }

        .footer {
            text-align: center;
            color: #808080;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(128, 128, 128, 0.2);
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">ChestVision AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered chest X-ray analysis using MobileNetV2 and Grad-CAM"
    "</div>",
    unsafe_allow_html=True,
)

st.write(
    "Upload a chest X-ray image to obtain a model prediction and "
    "visual explanation of the regions that influenced the prediction."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("About the Model")

    st.write(
        """
        **ChestVision AI** is an academic research prototype for
        binary classification of chest X-ray images into:

        - **NORMAL**
        - **PNEUMONIA**
        """
    )

    st.divider()

    st.subheader("Model Details")

    st.write("**Architecture:** MobileNetV2")
    st.write("**Approach:** Transfer Learning")
    st.write("**Input Size:** 224 × 224")
    st.write("**Output:** Pneumonia probability")
    st.write("**Decision Threshold:** 0.55")
    st.write("**Explainability:** Grad-CAM")

    st.divider()

    st.subheader("Important Note")

    st.caption(
        "This application is an academic/research prototype. "
        "It is not a medical diagnostic tool and should not be used "
        "for clinical decision-making."
    )


# ---------------------------------------------------------
# File uploader
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Upload Chest X-ray</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG.",
)


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()


# ---------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # Display uploaded image
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">Input Image</div>',
            unsafe_allow_html=True,
        )

        image_col, info_col = st.columns([2, 1])

        with image_col:

            st.image(
                uploaded_file,
                caption="Uploaded chest X-ray",
                use_container_width=True,
            )

        with info_col:

            st.markdown(
                """
                <div class="info-box">
                    <b>File Information</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(f"**Filename:** {uploaded_file.name}")

            file_size_kb = uploaded_file.size / 1024
            st.write(f"**Size:** {file_size_kb:.1f} KB")

            st.write(f"**Format:** {uploaded_file.type}")

        # -------------------------------------------------
        # Save temporary image
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        # -------------------------------------------------
        # Load model
        # -------------------------------------------------

        with st.spinner("Loading model..."):

            model = get_model()

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        with st.spinner("Analyzing X-ray..."):

            result = predict_image(
                temp_path,
                model,
            )

        predicted_class = result["class"]
        probability = result["pneumonia_probability"]
        threshold = result["threshold"]

        # -------------------------------------------------
        # Prediction result
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True,
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if predicted_class == "PNEUMONIA":

                st.error(
                    f"Prediction: {predicted_class}",
                    icon="⚠️",
                )

            else:

                st.success(
                    f"Prediction: {predicted_class}",
                )

        with result_col2:

            st.metric(
                "Pneumonia Probability",
                f"{probability * 100:.2f}%",
            )

        # -------------------------------------------------
        # Probability bar
        # -------------------------------------------------

        st.progress(
            probability,
            text=f"Pneumonia probability: {probability * 100:.2f}%",
        )

        st.caption(
            f"Decision threshold: {threshold:.2f}. "
            "Probabilities at or above the threshold are classified "
            "as PNEUMONIA."
        )

        # -------------------------------------------------
        # Grad-CAM section
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">Model Explainability</div>',
            unsafe_allow_html=True,
        )

        st.write(
            "Grad-CAM highlights image regions that contributed to "
            "the model's prediction."
        )

        # -------------------------------------------------
        # Decode uploaded image
        # -------------------------------------------------

        image_bytes = uploaded_file.getvalue()

        image_tensor = tf.image.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )

        image_tensor = tf.image.resize(
            image_tensor,
            [224, 224],
        )

        image_tensor = tf.cast(
            image_tensor,
            tf.float32,
        )

        image_tensor = tf.expand_dims(
            image_tensor,
            axis=0,
        )

        # -------------------------------------------------
        # Generate Grad-CAM
        # -------------------------------------------------

        with st.spinner("Generating Grad-CAM explanation..."):

            heatmap, cam_probability = generate_gradcam(
                model,
                image_tensor,
            )

            overlay = create_overlay(
                image_tensor,
                heatmap,
                alpha=0.4,
            )

        # -------------------------------------------------
        # Display Grad-CAM
        # -------------------------------------------------

        cam_col1, cam_col2 = st.columns(2)

        with cam_col1:

            st.image(
                image_tensor[0].numpy().astype("uint8"),
                caption="Preprocessed X-ray",
                use_container_width=True,
            )

        with cam_col2:

            st.image(
                overlay,
                caption="Grad-CAM Heatmap Overlay",
                use_container_width=True,
            )

        st.caption(
            "Grad-CAM provides a visual explanation of regions "
            "associated with the model's prediction. It is not a "
            "clinically validated lesion or disease map."
        )

    except Exception as e:

        st.error(
            "Unable to process this image. "
            "Please check that the uploaded file is a valid chest X-ray."
        )

        with st.expander("Technical details"):

            st.exception(e)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        ChestVision AI • MobileNetV2 Transfer Learning • Grad-CAM
        <br>
        Academic / Research Prototype
    </div>
    """,
    unsafe_allow_html=True,
)