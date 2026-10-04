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