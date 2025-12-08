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

