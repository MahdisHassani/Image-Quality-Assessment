import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models.model import get_model
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None

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


def apply_gradcam(image_path):
    model = get_model().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    gradcam = GradCAM(model)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])        
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    cam = gradcam.generate(input_tensor)

    image_np = np.array(image.resize((224, 224)))

    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    overlay = np.uint8(overlay)

    cv2.imwrite("gradcam_result.jpg", overlay)

    print("Grad-CAM saved as gradcam_result.jpg")
    
apply_gradcam("data/512x384/499095228.jpg")