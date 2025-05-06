#!/usr/bin/env python3
# sample_vis.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk
from preprocessing import HaNSegDataset

# 1) Configuration
DATA_DIR = "./HaN-Seg"
OUTPUT_PATH = "./sample_slice.png"

# 2) Load first case
ds = HaNSegDataset(DATA_DIR)
img_tensor, mask_tensor = ds[0]  # image_tensor: [2, D, H, W], mask_tensor: [C_masks, D, H, W]

# 3) Convert to NumPy and pick a slice
img = img_tensor.numpy()
mask = mask_tensor.numpy().max(axis=0)  # collapse channels into one mask
z = img.shape[1] // 2  # Middle slice, adjust if needed

ct_slice = img[0, z]
mr_slice = img[1, z]
mask_slice = mask[z]

# Debug: Print statistics
print("CT slice stats - min:", ct_slice.min(), "max:", ct_slice.max(), "mean:", ct_slice.mean())
print("MR slice stats - min:", mr_slice.min(), "max:", mr_slice.max(), "mean:", mr_slice.mean())
print("Mask slice stats - min:", mask_slice.min(), "max:", mask_slice.max(), "mean:", mask_slice.mean())

# Normalize MR slice for display (min-max to [0, 1])
if mr_slice.max() > mr_slice.min():
    mr_slice_normalized = (mr_slice - mr_slice.min()) / (mr_slice.max() - mr_slice.min() + 1e-8)
else:
    mr_slice_normalized = mr_slice
    print("Warning: MR slice has no valid range.")

# 4) Plot side-by-side and save
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].imshow(ct_slice, cmap='gray', vmin=0, vmax=1); axs[0].set_title("CT"); axs[0].axis('off')
axs[1].imshow(mr_slice_normalized, cmap='gray'); axs[1].set_title("MR"); axs[1].axis('off')
axs[2].imshow(mask_slice, cmap='viridis'); axs[2].set_title("Mask"); axs[2].axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=150)
print(f"Sample slice saved to {OUTPUT_PATH}")
