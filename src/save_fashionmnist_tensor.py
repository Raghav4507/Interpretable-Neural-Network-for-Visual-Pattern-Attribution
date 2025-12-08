import torch
from torchvision import datasets, transforms

def save_fashionmnist():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = datasets.FashionMNIST(root="../data", train=True, download=True, transform=transform)

    data_list = []
    
    for img, label in train:
        data_list.append((img.squeeze(0), label))  # store tensor + label

    torch.save(data_list, "../data/fashionmnist_train_tensor.pt")
    print("✅ Saved: ../data/fashionmnist_train_tensor.pt")

if __name__ == "__main__":
    save_fashionmnist()
