import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import SRCNN  # Make sure this is defined correctly

# === Multi-Image Dataset Class ===
class MultiImageDataset(Dataset):
    def __init__(self, image_paths):
        self.lr_images = []
        self.hr_images = []

        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"❌ Skipping missing file: {img_path}")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {img_path}")
                continue

            img = img.astype(np.float32) / 255.0
            hr = img

            lr = cv2.resize(hr, (hr.shape[1] // 4, hr.shape[0] // 4), interpolation=cv2.INTER_AREA)
            lr_up = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_LINEAR)

            self.lr_images.append(lr_up)
            self.hr_images.append(hr)

        print(f"📦 Loaded {len(self.lr_images)} images for training.")

    def __len__(self): return len(self.lr_images)

    def __getitem__(self, idx):
        return torch.tensor(self.lr_images[idx]).unsqueeze(0), torch.tensor(self.hr_images[idx]).unsqueeze(0)

# === Set Multiple Image Paths ===
image_paths = [
    "TRAIN1.png",
    "TRAIN2.png",
    "TRAIN3.png",
    "TRAIN4.png"
]

model_output_path = "S:/oceanproject/srcnn.pth"

# === Load Dataset ===
dataset = MultiImageDataset(image_paths)
loader = DataLoader(dataset, batch_size=1)

# === Create Model ===
model = SRCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# === Train Model ===
print("🚀 Starting SRCNN training on multiple images...")
for epoch in range(50):
    total_loss = 0
    for lr_img, hr_img in loader:
        output = model(lr_img.float())
        loss = loss_fn(output, hr_img.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"✅ Epoch {epoch+1:4d}: Loss = {total_loss / len(loader):.6f}")

# === Save Model ===
torch.save(model.state_dict(), model_output_path)
print(f"\n💾 Model saved successfully at: {model_output_path}")