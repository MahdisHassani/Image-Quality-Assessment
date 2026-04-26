import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

from datasets.koniq_dataset import konIQDataset
from models.model import get_model
from config import *
from utils import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


train_dataset = konIQDataset(CSV_PATH, IMAGE_DIR, train_transform, "training")
val_dataset = konIQDataset(CSV_PATH, IMAGE_DIR, val_transform, "validation")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = get_model().to(device)

mse = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()

def loss_fn(outputs, labels):
    return mse(outputs, labels) + 0.5 * l1(outputs, labels)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

best_srcc = -1

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    model.train()
    train_loss = 0
    
    loop = tqdm(train_loader)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)
        
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    avg_train_loss = train_loss / len(train_loader)
    
    model.eval()
    val_loss = 0
    preds, targets = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.unsqueeze(1).to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            val_loss += loss.item()
            
            preds.extend(outputs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())
            
    avg_val_loss = val_loss / len(val_loader)
    
    plcc, srcc = compute_metrics(preds, targets)
    
    scheduler.step(srcc)
    
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"PLCC: {plcc:.4f}, SRCC: {srcc:.4f}")
    
    if srcc > best_srcc:
        best_srcc = srcc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("Best model saved!")