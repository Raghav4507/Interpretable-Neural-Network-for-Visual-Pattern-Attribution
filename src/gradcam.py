"""
Grad-CAM for your ResNet18 baseline.
Saves heatmaps in ../outputs/visualizations/gradcam/
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms, utils
from train_utils import ensure_dirs, get_dataloaders
from PIL import Image

ensure_dirs()

def build_resnet18():
    model = models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(512, 10)
    return model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.features = None
        self.gradients = None
        # register hooks
        def forward_hook(module, inp, out):
            self.features = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, C, H, W) tensor on same device as model
        returns resized cam (H, W) numpy 0..1
        """
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(out.argmax(dim=1).item())
        score = out[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients  # (N, C, h, w)
        fmap = self.features    # (N, C, h, w)

        weights = grads.mean(dim=(2,3), keepdim=True)  # (N, C,1,1)
        cam = (weights * fmap).sum(dim=1).squeeze(0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        # upsample to input size
        cam = np.uint8(255 * cam)
        cam = Image.fromarray(cam).resize((input_tensor.shape[3], input_tensor.shape[2]), resample=Image.BILINEAR)
        cam = np.array(cam).astype(np.float32) / 255.0
        return cam

def overlay_cam_on_image(img_tensor, cam):
    # img_tensor: (1,C,H,W) CPU, values [0,1] - grayscale
    img = img_tensor.squeeze(0).squeeze(0).cpu().numpy()
    h,w = img.shape
    cmap = plt.get_cmap("jet")
    cam_color = cmap(cam)[:,:,:3]
    cam_color = cam_color[..., ::-1]  # RGB->BGR not necessary, keep RGB
    # blend
    blended = 0.5 * np.dstack([img,img,img]) + 0.5 * cam_color
    blended = np.clip(blended, 0, 1)
    return blended

def run_and_save(n_images=20, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # load model
    model = build_resnet18().to(device)
    state = torch.load("../outputs/models/baseline_cnn.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    # choose target layer: last conv block — model.layer4[-1].conv2
    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)

    loader = get_dataloaders(batch_size=1, resize_for_model=224, train=False)
    os.makedirs("../outputs/visualizations/gradcam", exist_ok=True)

    saved = 0
    for i, (img, label) in enumerate(loader):
        if saved >= n_images:
            break
        img = img.to(device)  # (1,1,224,224)

        cam = gradcam(img, None)  # (H,W) float 0..1
        blended = overlay_cam_on_image(img.cpu(), cam)

        plt.figure(figsize=(4,4))
        plt.imshow(blended)
        plt.axis("off")
        fname = f"../outputs/visualizations/gradcam/gradcam_{i}_label{int(label.item())}.png"
        plt.savefig(fname, bbox_inches="tight", pad_inches=0)
        plt.close()
        saved += 1

    print(f"Saved {saved} gradcam images to ../outputs/visualizations/gradcam/")

if __name__ == "__main__":
    run_and_save(n_images=30)
