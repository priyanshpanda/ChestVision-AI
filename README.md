# ChestVision AI

## AI-Powered Chest X-ray Disease Detection using Deep Learning and Image Processing

ChestVision AI is a deep learning-based project for detecting **pneumonia from chest X-ray images**.

The project uses **MobileNetV2 Transfer Learning** for binary image classification and **Grad-CAM** for visual model explainability.

A **Streamlit web application** is also included, allowing users to upload a chest X-ray and view the model prediction, pneumonia probability, and Grad-CAM visualization.

---

## Features

- Chest X-ray classification
- NORMAL vs PNEUMONIA detection
- MobileNetV2 transfer learning
- Image preprocessing and data augmentation
- Validation-based classification threshold selection
- Model evaluation and error analysis
- ROC-AUC and Precision-Recall analysis
- Grad-CAM explainability
- Interactive Streamlit web application
- Support for JPG, JPEG and PNG images

---

## Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.

**Dataset:**  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains two classes:

- NORMAL
- PNEUMONIA

### Dataset Split

A new validation split was created from the original training data, while the original test set was kept untouched for final evaluation.

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1,139 | 3,293 | 4,432 |
| Validation | 202 | 582 | 784 |
| Test | 234 | 390 | 624 |

---

## Methodology

The overall workflow of ChestVision AI is:

```text
Chest X-ray
     |
     v
Image Preprocessing
     |
     v
Resize to 224 x 224
     |
     v
Data Augmentation
     |
     v
MobileNetV2
     |
     v
Global Average Pooling
     |
     v
Dropout
     |
     v
Dense + Sigmoid
     |
     v
Pneumonia Probability
     |
     v
Classification Threshold = 0.55
     |
     +----------------------+
     |                      |
     v                      v
  NORMAL                PNEUMONIA
     |
     v
Grad-CAM Explainability
Image Preprocessing
The X-ray images have different original dimensions. The images are therefore resized to 224 × 224 pixels before being passed to the model.
Training augmentation includes:
- Random rotation
- Random zoom
- Random translation
- Random contrast
Horizontal flipping was not used because left-right orientation can have clinical significance in chest radiography.
Model
The project uses MobileNetV2 with ImageNet pretrained weights.
Architecture
Input Image (224 × 224 × 3)
            |
            v
     Data Augmentation
            |
            v
        MobileNetV2
            |
            v
 Global Average Pooling
            |
            v
       Dropout (0.3)
            |
            v
      Dense + Sigmoid
            |
            v
  Pneumonia Probability

The model was trained using transfer learning.
A classification threshold of 0.55 was selected using the validation set.
Probability >= 0.55  →  PNEUMONIA
Probability <  0.55  →  NORMAL

Results
The final evaluation was performed on the untouched test set using the validation-selected threshold of 0.55.
Metric	Score
Accuracy	78.53%
Precision	75.10%
Sensitivity / Recall	98.21%
Specificity	45.73%
F1-Score	85.11%
ROC-AUC	94.25%
Average Precision	96.60%


Confusion Matrix
                 Predicted
                 NORMAL  PNEUMONIA

Actual NORMAL       107      127
Actual PNEUMONIA      7      383

The model achieved high sensitivity for pneumonia detection, with 383 true-positive predictions and only 7 false-negative predictions on the test set.
However, the model produced 127 false-positive predictions, resulting in relatively low specificity.
Model Explainability — Grad-CAM
ChestVision AI uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize image regions that contributed to the model's prediction.
Grad-CAM helps provide an interpretable visual representation of the model's decision process.
Example outputs include:
- Preprocessed chest X-ray
- Grad-CAM heatmap overlay
Grad-CAM is an interpretability technique and should not be considered a clinically validated lesion or disease map.

Web Application
ChestVision AI includes an interactive Streamlit application.
The application allows the user to:
1. Upload a chest X-ray image
2. Preprocess the image
3. Generate a model prediction
4. View the pneumonia probability
5. View the classification threshold
6. Generate a Grad-CAM explanation
Run the Application
Activate the project environment and run:
streamlit run app/app.py

The application will be available at:
http://localhost:8501

Project Structure
ChestVision-AI/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── demo_images/
│   ├── IM-0001-0001.jpeg
│   ├── NORMAL2-IM-0256-0001.jpeg
│   └── person100_bacteria_475.jpeg
│
├── models/
│   └── best_mobilenetv2.keras
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── prediction.py
│   └── gradcam.py
│
├── .gitignore
├── README.md
└── requirements.txt

Technologies Used
- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Git / GitHub
Limitations
- The model produces a relatively high number of false-positive predictions.
- The dataset contains class imbalance between NORMAL and PNEUMONIA images.
- The dataset may contain variations in image quality, positioning, and acquisition conditions.
- Grad-CAM provides an approximate explanation of model behavior and does not prove the presence of a clinical lesion.
- The model has not been clinically validated.
- The system should not be used for real-world medical diagnosis.
Future Work
Possible improvements include:
- Fine-tuning the MobileNetV2 backbone
- Comparing other CNN architectures
- Improving specificity while maintaining high sensitivity
- Hyperparameter optimization
- External dataset validation
- Probability calibration
- Cross-dataset generalization testing
- Clinical expert evaluation
Disclaimer
ChestVision AI is an academic and research prototype.
The predictions generated by this application are intended for educational and research purposes only.
This system is not a medical diagnostic device and should not be used as a substitute for professional medical evaluation or clinical decision-making.
Author
Priyansh Panda
MCA — VIT Bhopal University
