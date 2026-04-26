import torch
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

from datasets.koniq_dataset import konIQDataset
from models.model import get_model
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("failure_analysis", exist_ok=True)

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.layer4[-1]

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image):
        self.model.zero_grad()
        output = self.model(input_image)
        output.backward()

        gradients = self.gradients
        activation = self.activation

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = konIQDataset(CSV_PATH, IMAGE_DIR, transform, split="test")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = get_model().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

gradcam = GradCAM(model)

errors = []

with torch.no_grad():
    for i, (img, label) in enumerate(loader):
        img = img.to(device)

        pred = model(img).item()
        true = label.item()

        error = abs(pred - true)
        errors.append((error, i, pred, true))

errors.sort(reverse=True)
worst = errors[:10]

print("\nAnalyzing worst samples...")

for rank, (error, idx, pred, true) in enumerate(worst):
    img_path = dataset.df.iloc[idx]["image_name"]
    full_path = os.path.join(IMAGE_DIR, img_path)

    image = Image.open(full_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    cam = gradcam.generate(input_tensor)

    image_np = np.array(image.resize((224, 224)))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    text = f"Pred: {pred:.2f} | True: {true:.2f}"
    cv2.putText(overlay, text, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    save_path = f"failure_analysis/sample_{rank}.jpg"
    cv2.imwrite(save_path, overlay)

print("Saved failure analysis images in folder: failure_analysis")