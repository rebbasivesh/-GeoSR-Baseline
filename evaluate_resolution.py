import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Load image (grayscale)
img = cv2.imread(r"S:\oceanproject\test image.png",
                 cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

# Original image shape (high-res)
original_shape = img.shape  # (height, width)
original_pixels = original_shape[0] * original_shape[1]

# Downscale (simulate low-res)
lr = cv2.resize(
    img, (original_shape[1]//4, original_shape[0]//4), interpolation=cv2.INTER_AREA)
lr_shape = lr.shape
lr_pixels = lr_shape[0] * lr_shape[1]

# Upscale back using bilinear interpolation
up = cv2.resize(
    lr, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)

# PSNR and SSIM
print("🔍 PSNR:", psnr(img, up))
print("🔍 SSIM:", ssim(img, up, data_range=1.0))

# Pixel count
print(f"\n🧾 Pixel Count:")
print(
    f"Original image  : {original_shape[1]} × {original_shape[0]} = {original_pixels} pixels")
print(f"Low-res image   : {lr_shape[1]} × {lr_shape[0]} = {lr_pixels} pixels")
print(
    f"Upscaled image  : {original_shape[1]} × {original_shape[0]} = {original_pixels} pixels")

# Optional: Grid size calculation
region_width_deg = 10  # Assume image covers 10 degrees horizontally
grid_size_highres = region_width_deg / original_shape[1]
grid_size_lowres = region_width_deg / lr_shape[1]

print(f"\n🌐 Grid Size:")
print(f"High-res grid size: {grid_size_highres:.4f}° per pixel")
print(f"Low-res grid size : {grid_size_lowres:.4f}° per pixel")

# Plot images
plt.figure(figsize=(12, 4))
for i, (im, title) in enumerate([(img, "Original"), (lr, "Low-Res"), (up, "Bilinear Upscaled")]):
    plt.subplot(1, 3, i+1)
    # Resize LR image to original size for consistent display
    im_show = cv2.resize(im, original_shape[::-1]) if i == 1 else im
    plt.imshow(im_show, cmap='gray')
    plt.title(f"{title}")
    plt.axis('off')
plt.tight_layout()
plt.show()
