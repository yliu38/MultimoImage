#!/usr/bin/env python3
# Preprocessing.py

import numpy as np
import torch
from pathlib import Path
import SimpleITK as sitk

class HaNSegDataset(torch.utils.data.Dataset):
    """
    Head-and-Neck OAR Segmentation (HaN-Seg Set 1).
    Loads CT & T1-MR volumes plus all OAR masks per case.
    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir) / "set_1"
        self.patient_dirs = sorted(self.root_dir.glob("case_*"))
        if not self.patient_dirs:
            raise RuntimeError(f"No 'case_*' folders found in {self.root_dir}")

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]

        # 1) Find CT +/logger.info MR
        ct_path = next(patient_dir.glob("*_IMG_CT.nrrd"), None)
        mr_path = next(patient_dir.glob("*_IMG_MR*.nrrd"), None)
        if ct_path is None or mr_path is None:
            raise FileNotFoundError(f"Missing CT or MR in {patient_dir}")
        print(f"Case {idx}: CT path: {ct_path}")
        print(f"Case {idx}: MR path: {mr_path}")

        # 2) Find masks
        mask_paths = sorted(patient_dir.glob("*.seg.nrrd"))
        if not mask_paths:
            raise FileNotFoundError(f"No .seg.nrrd masks in {patient_dir}")

        # 3) Read + resample MR → CT grid
        ct_img = sitk.ReadImage(str(ct_path))
        mr_img = sitk.ReadImage(str(mr_path))
        
        # Debug pixel types and sizes
        print(f"Case {idx}: CT pixel type: {ct_img.GetPixelIDTypeAsString()}, size: {ct_img.GetSize()}")
        print(f"Case {idx}: MR pixel type: {mr_img.GetPixelIDTypeAsString()}, size: {mr_img.GetSize()}")
        print(f"Case {idx}: CT spacing: {ct_img.GetSpacing()}, origin: {ct_img.GetOrigin()}, direction: {ct_img.GetDirection()}")
        print(f"Case {idx}: MR spacing: {mr_img.GetSpacing()}, origin: {mr_img.GetOrigin()}, direction: {mr_img.GetDirection()}")
        print(f"Case {idx}: MR pre-resample stats: min={sitk.GetArrayFromImage(mr_img).min()}, max={sitk.GetArrayFromImage(mr_img).max()}")

        # Convert images to float32 to ensure compatibility
        ct_img = sitk.Cast(ct_img, sitk.sitkFloat32)
        mr_img = sitk.Cast(mr_img, sitk.sitkFloat32)

        # Resample with identity transform (fallback)
        default_value = float(sitk.GetArrayFromImage(mr_img).min())  # Convert to float
        try:
            # Try centered transform
            transform = sitk.CenteredTransformInitializer(
                ct_img, mr_img, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        except RuntimeError as e:
            print(f"Case {idx}: Warning: CenteredTransformInitializer failed ({e}), using identity transform")
            transform = sitk.Transform()  # Fallback to identity transform
        
        mr_on_ct = sitk.Resample(
            mr_img, ct_img, transform,
            sitk.sitkLinear, default_value, sitk.sitkFloat32
        )
        mr_vol = sitk.GetArrayFromImage(mr_on_ct)
        print(f"Case {idx}: MR post-resample stats: min={mr_vol.min()}, max={mr_vol.max()}")

        # 4) To NumPy (D,H,W)
        ct_vol = sitk.GetArrayFromImage(ct_img)
        image = np.stack([ct_vol, mr_vol], axis=0).astype(np.float32)

        # 5) Load + stack masks (C_masks, D,H,W)
        masks = [sitk.GetArrayFromImage(sitk.ReadImage(str(m))) for m in mask_paths]
        mask = np.stack(masks, axis=0).astype(np.int64)

        # 6) Normalize intensities
        image[0] = np.clip(image[0], -1000, 1000)
        image[0] = (image[0] + 1000) / 2000.0
        mr_mean, mr_std = image[1].mean(), image[1].std() + 1e-8
        print(f"Case {idx}: MR pre-normalization: mean={mr_mean}, std={mr_std}")
        if mr_std < 1e-4:
            print(f"Case {idx}: Warning: MR has low variance, using min-max normalization")
            image[1] = (image[1] - image[1].min()) / (image[1].max() - image[1].min() + 1e-8) if image[1].max() > image[1].min() else image[1]
        else:
            image[1] = (image[1] - mr_mean) / mr_std
        print(f"Case {idx}: MR post-normalization: min={image[1].min()}, max={image[1].max()}")

        return torch.from_numpy(image), torch.from_numpy(mask)

if __name__ == "__main__":
    # ---- User configuration ----
    DATA_DIR = "./HaN-Seg"

    # Instantiate dataset
    ds = HaNSegDataset(DATA_DIR)
    print(f"Found {len(ds)} cases under {DATA_DIR}/set_1\n")

    # Test MR file directly
    mr_path = next(ds.patient_dirs[0].glob("*_IMG_MR*.nrrd"))
    print(f"Testing MR file: {mr_path}")
    mr_img = sitk.ReadImage(str(mr_path))
    mr_array = sitk.GetArrayFromImage(mr_img)
    print(f"MR array shape: {mr_array.shape}, min: {mr_array.min()}, max: {mr_array.max()}, mean: {mr_array.mean()}")

    # Quick metadata overview for the first 2 cases
    for idx in range(min(2, len(ds))):
        pd = ds.patient_dirs[idx]
        ct = next(pd.glob("*_IMG_CT.nrrd")).name
        mr = next(pd.glob("*_IMG_MR*.nrrd")).name
        masks = [m.name for m in sorted(pd.glob("*.seg.nrrd"))]
        print(f"Case {idx:02d}:")
        print(f"  CT : {ct}")
        print(f"  MR : {mr}")
        print(f"  Masks ({len(masks)}): {masks}\n")

    # Load and display shapes for the first case only
    img, mask = ds[0]
    print("Loaded first sample:")
    print("  Image tensor shape:", img.shape, " dtype:", img.dtype)
    print("  Mask tensor shape:", mask.shape, " dtype:", mask.dtype)
