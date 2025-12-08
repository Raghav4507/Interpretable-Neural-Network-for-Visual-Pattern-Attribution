import os
import torch
from torchvision import datasets, transforms

def ensure_dirs():
    os.makedirs("../outputs/models", exist_ok=True)
    os.makedirs("../outputs/visualizations/gradcam", exist_ok=True)
    os.makedirs("../outputs/visualizations/shap", exist_ok=True)
    os.makedirs("../outputs/metrics", exist_ok=True)

def get_dataloaders(batch_size=64, resize_for_model=224, train=True):
    """
    Returns a torchvision dataset or dataloader depending on `train`.
    resize_for_model: size to resize for model inference (224 for your baseline ResNet)
    """
    transform = transforms.Compose([
        transforms.Resize((resize_for_model, resize_for_model)),
        transforms.ToTensor()
    ])
    ds = datasets.FashionMNIST(root="../data", train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train)
    return loader

def load_saved_tensor(tensor_path="../data/fashionmnist_train_tensor.pt"):
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"{tensor_path} not found. Run save_fmnist_tensor.py to create it.")
    return torch.load(tensor_path)
