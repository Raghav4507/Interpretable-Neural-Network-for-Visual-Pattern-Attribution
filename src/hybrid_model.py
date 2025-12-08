import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from handcrafted_features import extract_handcrafted
from torchvision import models
import joblib
import os


# ============================================================
# 1. Build ResNet18 Embedder (same architecture as baseline)
# ============================================================
def build_resnet18_embedder():
    model = models.resnet18(weights=None)

    # Change first conv → accept 1 channel
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Remove classifier → return 512-dim embedding
    model.fc = nn.Identity()

    return model


# ============================================================
# 2. Hybrid Model Training
# ============================================================
def train_hybrid():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline_path = "../outputs/models/baseline_cnn.pth"
    tensor_path = "../data/fashionmnist_train_tensor.pt"

    # ------------------------------------------------------------
    # Check required files
    # ------------------------------------------------------------
    if not os.path.exists(baseline_path):
        raise FileNotFoundError("❌ baseline_cnn.pth NOT FOUND! Run baseline_cnn.py first.")

    if not os.path.exists(tensor_path):
        raise FileNotFoundError("❌ fashionmnist_train_tensor.pt NOT FOUND! "
                                "Run save_fmnist_tensor.py first.")

    print("🔥 Loading baseline ResNet18 model...")

    # ------------------------------------------------------------
    # Load embedder with same architecture
    # ------------------------------------------------------------
    embedder = build_resnet18_embedder().to(device)

    # Load baseline CNN weights
    baseline_sd = torch.load(baseline_path, map_location=device)

    # Remove FC because embedder has no FC
    baseline_sd = {k: v for k, v in baseline_sd.items() if not k.startswith("fc.")}

    embedder.load_state_dict(baseline_sd, strict=False)
    embedder.eval()

    print("✔ Baseline weights loaded successfully.")

    # ------------------------------------------------------------
    # Load saved Fashion-MNIST dataset
    # ------------------------------------------------------------
    print("📥 Loading Fashion-MNIST training tensor...")
    data = torch.load(tensor_path)

    X_cnn, X_hand, Y = [], [], []

    print("🔎 Extracting embeddings + handcrafted features...")

    # ------------------------------------------------------------
    # Loop through all saved images
    # ------------------------------------------------------------
    for img, label in data:
        # img shape = (28,28)

        # Add channel + batch → (1,1,28,28)
        img4d = img.unsqueeze(0).unsqueeze(0)

        # Resize to (1,1,224,224)
        img_resized = torch.nn.functional.interpolate(
            img4d,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        ).to(device)

        # CNN embedding
        with torch.no_grad():
            emb = embedder(img_resized).cpu().numpy().flatten()

        # Handcrafted feature vector
        handcrafted = extract_handcrafted(img.numpy())

        X_cnn.append(emb)
        X_hand.append(handcrafted)
        Y.append(label)

    # Convert lists → numpy
    X_cnn = np.array(X_cnn)
    X_hand = np.array(X_hand)
    Y = np.array(Y)

    print(f"📐 CNN embedding size: {X_cnn.shape}")
    print(f"📐 Handcrafted feature size: {X_hand.shape}")

    # Final hybrid feature = concatenate
    X_final = np.concatenate([X_cnn, X_hand], axis=1)

    print(f"📐 Final hybrid feature vector size: {X_final.shape}")

    # ------------------------------------------------------------
    # Train Random Forest
    # ------------------------------------------------------------
    print("🌲 Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=300)
    clf.fit(X_final, Y)

    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------
    os.makedirs("../outputs/models", exist_ok=True)
    joblib.dump(clf, "../outputs/models/hybrid_model.pkl")

    print("🎉 HYBRID model trained and saved successfully!")

    return clf


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    train_hybrid()
