import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops

def extract_hog(img):
    img = img.squeeze()
    return hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9)

def extract_glcm(img):
    img = (img.squeeze() * 255).astype(np.uint8)
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return np.array([contrast, energy, homogeneity])

def extract_handcrafted(img):
    hog_feat = extract_hog(img)
    glcm_feat = extract_glcm(img)
    return np.concatenate([hog_feat, glcm_feat])


# ---------------------------------------------------------
# SELF TEST (Runs only when you execute: python handcrafted_features.py)
# ---------------------------------------------------------
if __name__ == "__main__":
    from torchvision import datasets, transforms

    print("Loading Fashion-MNIST sample image...")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform
    )

    img, label = dataset[0]
    img_np = img.numpy()

    print(f"Sample label: {label}")

    print("Extracting handcrafted features (HOG + GLCM)...")
    features = extract_handcrafted(img_np)

    print("\n✅ Feature extraction successful!")
    print("Total features:", len(features))
    print("First 20 feature values:", features[:20])
