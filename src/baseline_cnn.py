import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    train = datasets.FashionMNIST(root="../data", train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root="../data", train=False, download=True, transform=transform)

    return (torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False))

def build_resnet18():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 10)
    return model

def train_baseline():
    train_loader, test_loader = get_dataloaders()

    model = build_resnet18()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/5 Loss={loss.item()}")

    # 🔥 FIX: Create directory before saving
    os.makedirs("../outputs/models", exist_ok=True)

    torch.save(model.state_dict(), "../outputs/models/baseline_cnn.pth")
    print("Model saved successfully!")
    return model

if __name__ == "__main__":
    train_baseline()
