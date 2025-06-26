import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SRCNN
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# === Fixed file paths ===
image_path = "S:/oceanproject/test image.png"      # ✅ Use forward slashes or raw string
model_path = "S:/oceanproject/srcnn.pth"           # ✅ Load actual trained model (.pth), not .py

# === Load original image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"❌ Could not load image at: {image_path}")
img = img.astype(np.float32) / 255.0
original_shape = img.shape

# === Simulate low-res version
lr = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4), interpolation=cv2.INTER_AREA)
lr_up = cv2.resize(lr, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

# === Load trained model
model = SRCNN()
model.load_state_dict(torch.load(model_path))
model.eval()

# === CNN Super-resolution
input_tensor = torch.tensor(lr_up).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    sr = model(input_tensor.float()).squeeze().numpy()

# === Metrics
print("\n📊 PSNR & SSIM:")
print(f"  Bilinear → PSNR: {psnr(img, lr_up):.2f}, SSIM: {ssim(img, lr_up, data_range=1.0):.4f}")
print(f"  CNN      → PSNR: {psnr(img, sr):.2f}, SSIM: {ssim(img, sr, data_range=1.0):.4f}")

# === Pixel Counts
print("\n🧾 Pixel Count:")
print(f"  Original       : {original_shape[1]} x {original_shape[0]} = {original_shape[0] * original_shape[1]}")
print(f"  Low-Res        : {lr.shape[1]} x {lr.shape[0]} = {lr.shape[0] * lr.shape[1]}")
print(f"  Bilinear/CNN   : {lr_up.shape[1]} x {lr_up.shape[0]} = {lr_up.shape[0] * lr_up.shape[1]}")

# === Grid Size Estimation
region_width_deg = 10
grid_highres = region_width_deg / original_shape[1]
grid_lowres = region_width_deg / lr.shape[1]

print("\n🌐 Grid Size:")
print(f"  High-Res Grid: {grid_highres:.4f}°/pixel")
print(f"  Low-Res Grid : {grid_lowres:.4f}°/pixel")

# === Visualization
plt.figure(figsize=(12, 4))
for i, (im, title) in enumerate([(img, "Original"), (lr_up, "Bilinear"), (sr, "CNN Output")]):
    plt.subplot(1, 3, i+1)
    plt.imshow(im, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()
