# Interpretable Neural Network for Visual Pattern Attribution


This project explores **interpretable deep learning for visual pattern recognition** using a combination of:
- A Convolutional Neural Network (ResNet-18)
- Handcrafted image descriptors (HOG + GLCM)
- Gradient-based explanation techniques

The goal is to improve **model interpretability** while maintaining strong classification performance on the **Fashion-MNIST** dataset.

---

## 📌 Project Overview

Traditional CNNs achieve strong accuracy but often lack interpretability.  
This project addresses this by:

1. Training a **baseline CNN** for image classification
2. Extracting **handcrafted texture and gradient features**
3. Building a **hybrid model** combining CNN embeddings with handcrafted features
4. Visualizing decision explanations using:
   - Grad-CAM
   - Saliency Maps
   - Integrated Gradients

---
## 🚀 How to Run

### 1️⃣ Create a virtual environment (recommended)

```bash
python -m venv ivp_env
ivp_env\Scripts\activate
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train baseline CNN
```bash
cd src
python baseline_cnn.py
```

### 4️⃣ Prepare data for hybrid model
```bash
python save_fashionmnist_tensor.py
```

### 5️⃣ Train hybrid model
```bash
python hybrid_model.py
```

### 6️⃣ Generate interpretability visualizations
```bash
python gradcam.py
python explain_gradients.py
```

### 7️⃣ Evaluate performance
```bash
python evaluate_metrics.py
```

## 📊 Results Summary

Baseline CNN Test Accuracy: ~90% <br />
#### Hybrid Feature Dimensionality: 
CNN Embedding: 512 <br />
Handcrafted Features: 147 <br />
Total Hybrid Vector: 659 <br />
Visual explanations show that the hybrid approach focuses more consistently on semantic garment regions rather than background noise.

## 🔍 Interpretability Methods
Method	Purpose
Grad-CAM	Highlights spatial regions influencing CNN predictions
Saliency Maps	Shows pixel-level sensitivity
Integrated Gradients	Measures feature importance along prediction paths

All visual outputs are saved in outputs/visualizations/.

## ⚠️ Notes

Due to platform-specific dependency constraints on Windows, hybrid test accuracy may not be reproducible in all environments.
However, hybrid training and qualitative interpretability analyses are fully reproducible and documented.

## 📚 Dataset

Fashion-MNIST
Zalando Research
https://github.com/zalandoresearch/fashion-mnist

