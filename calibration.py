import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.koniq_dataset import konIQDataset
from models.model import get_model
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

dataset = konIQDataset(CSV_PATH, IMAGE_DIR, transform, split="test")
loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = get_model().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

preds, targets = [], []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images).cpu().numpy().flatten()

        preds.extend(outputs)
        targets.extend(labels.numpy())

preds = np.array(preds)
targets = np.array(targets)

# binning
bins = np.linspace(0, 1, 10)
digitized = np.digitize(preds, bins)

bin_means_pred = []
bin_means_true = []

for i in range(1, len(bins)):
    mask = digitized == i
    if np.sum(mask) > 0:
        bin_means_pred.append(preds[mask].mean())
        bin_means_true.append(targets[mask].mean())

plt.figure()
plt.plot(bin_means_pred, bin_means_true, marker='o', label="Model")
plt.plot([0,1], [0,1], linestyle='--', label="Perfect")

plt.xlabel("Predicted Score")
plt.ylabel("True Score")
plt.title("Calibration Plot")
plt.legend()

plt.savefig("calibration_plot.png")
plt.show()