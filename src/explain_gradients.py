import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from train_utils import get_dataloaders, ensure_dirs

ensure_dirs()

# ----------------------------------------------------
# Build same model as baseline
# ----------------------------------------------------
def build_resnet18_full():
    model = models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(512, 10)
    return model


# ----------------------------------------------------
# Compute Saliency Map
# ----------------------------------------------------
def compute_saliency(model, img, label_idx):
    img.requires_grad_()
    output = model(img)
    loss = output[0, label_idx]
    model.zero_grad()
    loss.backward()

    saliency = img.grad.data.abs().squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)
    return saliency


# ----------------------------------------------------
# Compute Integrated Gradients
# ----------------------------------------------------
def integrated_gradients(model, img, label_idx, steps=50):
    baseline = torch.zeros_like(img)
    grads = []

    for i in range(steps):
        x = baseline + (i / steps) * (img - baseline)
        x.requires_grad_()
        output = model(x)
        loss = output[0, label_idx]
        model.zero_grad()
        loss.backward()
        grads.append(x.grad.data.clone())

    avg_grad = torch.mean(torch.stack(grads), dim=0)
    ig = (img - baseline) * avg_grad
    ig = ig.abs().squeeze().cpu().numpy()
    ig = (ig - ig.min()) / (ig.max() + 1e-8)
    return ig


# ----------------------------------------------------
# Run explanations
# ----------------------------------------------------
def run_explanations(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # load model
    model = build_resnet18_full().to(device)
    state = torch.load("../outputs/models/baseline_cnn.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    loader = get_dataloaders(batch_size=1, resize_for_model=224, train=False)
    os.makedirs("../outputs/visualizations/shap", exist_ok=True)

    count = 0
    for img, label in loader:
        img = img.to(device)
        pred = int(model(img).argmax(dim=1).item())

        # Saliency
        sal = compute_saliency(model, img.clone(), pred)

        # Integrated Gradients
        ig = integrated_gradients(model, img.clone(), pred)

        # Overlay
        base_img = img.squeeze().cpu().numpy()

        plt.figure(figsize=(10,4))

        plt.subplot(1,3,1)
        plt.imshow(base_img, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(base_img, cmap="gray")
        plt.imshow(sal, cmap="hot", alpha=0.5)
        plt.title("Saliency Map")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(base_img, cmap="gray")
        plt.imshow(ig, cmap="jet", alpha=0.5)
        plt.title("Integrated Gradients")
        plt.axis("off")

        out_path = f"../outputs/visualizations/shap/explain_{count}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        
        count += 1
        if count >= 10:
            break

    print("Saved 10 gradient-based explainability images to ../outputs/visualizations/shap/")


if __name__ == "__main__":
    run_explanations()
