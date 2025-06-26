import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from models.srcnn import SRCNN
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

# Parameters
test_file = "data/test_sst.nc"
model_path = "models/srcnn_sst.pth"
output_file = "data/predicted_sst.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
day_index = 0  # Change if testing a different day

# Load model
model = SRCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load test data
ds = xr.open_dataset(test_file)
sst_hr = ds['sst'].isel(time=day_index).values  # Shape: [lat, lon]
sst_hr = np.nan_to_num(sst_hr, nan=0.0)

# Normalize
sst_min, sst_max = np.min(sst_hr), np.max(sst_hr)
sst_norm = (sst_hr - sst_min) / (sst_max - sst_min)

# Simulate low-resolution input (downsample + upsample)
scale = 4
lr = F.interpolate(torch.tensor(sst_norm).unsqueeze(0).unsqueeze(0), 
                   scale_factor=1/scale, mode='bicubic', align_corners=False)
lr_up = F.interpolate(lr, size=sst_norm.shape, mode='bicubic', align_corners=False)

# Run CNN
with torch.no_grad():
    output = model(lr_up.to(device)).cpu().squeeze().numpy()

# Denormalize prediction
sst_pred = output * (sst_max - sst_min) + sst_min

# Save output
np.save(output_file, sst_pred)
print(f"Prediction saved to {output_file}")

# -------------------------------
# Evaluate and print improvements
# -------------------------------
hr_eval = sst_hr
pred_eval = sst_pred

rmse = np.sqrt(np.mean((hr_eval - pred_eval)**2))
psnr = peak_signal_noise_ratio(hr_eval, pred_eval, data_range=hr_eval.max() - hr_eval.min())
ssim = structural_similarity(hr_eval, pred_eval, data_range=hr_eval.max() - hr_eval.min())

print("\n=== Super-Resolution Evaluation ===")
print(f"RMSE: {rmse:.4f} °C")
print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")
print("===================================")
