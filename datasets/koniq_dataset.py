import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class konIQDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, split="training"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["set"] == split].reset_index(drop=True)
        
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = os.path.join(self.image_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")
        
        label = row["MOS"] / 100.0
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)