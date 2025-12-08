import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models


def build_resnet18_full():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 10)
    return model


def eval_baseline(device):
    model = build_resnet18_full().to(device)
    state = torch.load("../outputs/models/baseline_cnn.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    test_ds = datasets.FashionMNIST(root="../data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    baseline_acc = eval_baseline(device)
    print(f"Baseline test accuracy: {baseline_acc:.4f}")

    os.makedirs("../outputs/metrics", exist_ok=True)
    results = {
        "baseline": {"test_accuracy": baseline_acc},
        "hybrid": {"test_accuracy": None}
    }

    with open("../outputs/metrics/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Metrics saved to ../outputs/metrics/results.json")


if __name__ == "__main__":
    main()
