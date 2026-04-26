import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.koniq_dataset import konIQDataset
from models.model import get_model
from config import *
from utils import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

test_dataset = konIQDataset(CSV_PATH, IMAGE_DIR, transform, split="test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = get_model().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

preds, targets = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        
        outputs = model(images).cpu().numpy().flatten()
        
        preds.extend(outputs)
        targets.extend(labels.numpy())
        
plcc, srcc = compute_metrics(preds, targets)

print("\nTest Results:")
print(f"PLCC: {plcc:.4f}")
print(f"SRCC: {srcc:.4f}")