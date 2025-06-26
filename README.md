# -GeoSR-Baseline
Geospatial Super-Resolution Baseline with Bilinear Interpolation
GeoSR-Baseline/
GeoSR-Baseline/
│
├── TRAIN1.png - TRAIN4.png      # Training images (high-res)

├── test image.png               # Ground truth test image (high-res)

├── test.png                     # Low-resolution version of test image

├── output.png                   # Output from SRCNN model (super-resolved)

├── srcnn.pth                    # Saved SRCNN model weights

│
├── model.py                     # SRCNN model architecture

├── train.py                     # Trains the model on 4 images

├── test_model.py                # Applies trained SRCNN on test image

├── test.py                      # Bicubic/bilinear upscaling baseline

├── evaluate_resolution.py       # Computes PSNR, SSIM, RMSE, etc.

Install required libraries:

command - pip install numpy matplotlib scikit-image opencv-python torch torchvision

How to Run the Project:

**Step 1**: Train the SRCNN model
Use the 4 training images:
command - python train.py
Trains on TRAIN1.png to TRAIN4.png

Saves model weights to srcnn.pth

** Step 2**: Test the model
Use the test image:
 command - python test_model.py
Loads srcnn.pth

Upscales the low-res image (test.png)

Saves result as output.png
** Step 3:** Evaluate Results
Compare original, low-res, and SRCNN output:

python evaluate_resolution.py
Calculates:

PSNR

RMSE

MSE

MAE

SSIM
