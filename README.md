# GourmetVision-XAI
A high-performance PyTorch classifier for the Food-101 dataset using EfficientNet-B3, achieving 87% accuracy. Features robust data augmentation and Explainable AI (Grad-CAM) to visualize model attention and interpretability.


# 🍔 GourmetVision: Explainable Food Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)
![Status](https://img.shields.io/badge/Status-Maintained-green)
![Accuracy](https://img.shields.io/badge/Top--1_Accuracy-87.28%25-brightgreen)
![Accuracy](https://img.shields.io/badge/Top--5_Accuracy-97.69%25-brightgreen)

**GourmetVision** is a state-of-the-art deep learning project designed to classify images into 101 distinct food categories (from *Apple Pie* to *Waffles*). 

Built on **EfficientNet-B3**, this project goes beyond simple classification by integrating **Explainable AI (XAI)** to visualize model focus, and a **Research-Grade Evaluation Suite** including calibration curves and rare-class analysis.

## 🌟 Key Features

* **⚡ EfficientNet-B3 Backbone:** Utilizes transfer learning on ImageNet weights for high efficiency and accuracy.
* **🧠 Explainable AI (Grad-CAM):** Generates heatmaps to visualize exactly *where* the model is looking when making a prediction.
* **📊 Advanced Metrics:** Automatically calculates Top-1/Top-5 Accuracy, Macro F1-Score, and AUROC.
* **📉 Reliability Analysis:** Includes Calibration Curves (Reliability Diagrams) to ensure model confidence matches reality.
* **🛡️ Robust Training:** Implements heavy data augmentation (RandomResizedCrop, ColorJitter, GaussianBlur, RandomRotation) and Early Stopping.

## 🏆 Performance Results

Evaluated on the official **Food-101 Test Set** (25,250 images):

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | **87.28%** | Correctly predicts the exact food item. |
| **Top-5 Accuracy** | **97.69%** | The correct food is in the top 5 guesses. |
| **Macro AUROC** | **0.9974** | Exceptional ability to distinguish between classes. |
| **Macro F1-Score** | **0.8721** | Balanced precision and recall across classes. |

## 🧠 Technical Details

### Model Configuration
| Component | Specification | Purpose |
| :--- | :--- | :--- |
| **Backbone** | EfficientNet-B3 | Pre-trained on ImageNet for robust feature extraction. |
| **Classifier Head** | Linear Layer (101 classes) | Modified final layer with **Dropout (p=0.4)** to reduce overfitting. |
| **Loss Function** | CrossEntropyLoss | Implements **Label Smoothing (0.05)** to prevent the model from becoming over-confident. |
| **Optimizer** | Adam | Learning rate set to `1e-4` for stable convergence. |

### Data Augmentation Pipeline
To generalize well across diverse food images, we apply a rigorous augmentation pipeline using `torchvision.transforms`:

* **Geometric:** `RandomResizedCrop` (scale 0.8–1.0), `RandomHorizontalFlip`, `RandomRotation` (15°), `RandomAffine`.
* **Photometric:** `ColorJitter` (Brightness, Contrast, Saturation, Hue).
* **Filtering:** `GaussianBlur` to simulate varying camera focus quality.
* **Normalization:** Standard ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### Computational Complexity
* **Time Complexity:** $O(E \cdot N \cdot C)$, where $E$ is epochs, $N$ is the dataset size, and $C$ is the forward-pass cost (approx. 1.8 GFLOPs for EfficientNet-B3).
* **Space Complexity:** The model contains approximately **12 Million parameters**. Training requires ~4-6GB VRAM depending on batch size.

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/haroonwaheed19/GourmetVision-XAI.git
    cd GourmetVision
    ```

2.  **Install Dependencies**
    ```bash
    pip install torch torchvision matplotlib pandas scikit-learn tqdm seaborn opencv-python
    ```

3.  **Dataset**
    The code automatically downloads the **Food-101** dataset (~5GB) to the `./data` folder on the first run.

## 🚀 Usage

### 1. Training the Model
Run the Jupyter Notebook `AIFood101Explainer.ipynb`. The training loop includes:
* Automatic download of dataset (when you change the download=True in this line
* (full_train_aug = datasets.Food101(root=DATA_ROOT, split='train', transform=train_transform, download=False)).
* Data splitting (Train/Validation/Testing).
* Training with **Early Stopping** to prevent overfitting.
* Saving the best model to `AI_food_best_model.pth`.

### 2. Evaluation & Reports
The final cells of the notebook run a comprehensive evaluation pipeline that generates:
* `EffB3_Food101_classification_report.txt`: Detailed precision/recall for every dish.
* `EffB3_Food101_confusion_matrix.csv`: Raw confusion data.
* `AIFood_calibration_plot.png`: Visualizes model confidence reliability.
* `confusion_matrix_heatmap.png`: A massive heatmap of all 101 classes.

### 3. Running Inference & XAI
To classify a single image and see the Grad-CAM heatmap:

```python
# Example Code Snippet
from model_utils import predict_and_explain

image_path = "path/to/hamburger.jpg"
prediction, heatmap = predict_and_explain(model, image_path)

# This will display the image with the heatmap overlay

