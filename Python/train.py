#!/usr/bin/env python3
# train.py - Memory optimized with streamlined logging and efficient data loading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformer import TransformerSegmentation  # Assuming this is defined elsewhere
import gc
import psutil
import os
import time
import logging
import sys

# Set up logging with console handler to ensure immediate output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output to console immediately
        logging.FileHandler('training.log')  # Also save to file
    ]
)

def get_memory_usage():
    """Return current process memory usage in GB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024**3  # Convert bytes to GB

class MemoryEfficientHaNSeg2DSliceDataset(Dataset):
    """
    Memory-efficient dataset with slice-by-slice loading and minimal logging.
    """
    def __init__(self, root_dir, img_size=224):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.slices = []
        self.cache = {}  # In-memory cache for slices
        self.cache_size_limit = 100  # Limit cache to ~100 slices
        self.slice_counts = {}  # Cache CT slice counts
        
        start_time = time.time()
        logging.info("Initializing dataset")
        
        # Get patient directories
        self.patient_dirs = sorted([p for p in self.root_dir.glob("set_*/case_*") if p.is_dir()])
        if not self.patient_dirs:
            raise FileNotFoundError(f"No patient directories found in {self.root_dir}")
        logging.info(f"Found {len(self.patient_dirs)} case directories")
        
        # Count OARs
        oar_counts = {}
        self.reference_oars = None
        for case_idx, patient_dir in enumerate(self.patient_dirs):
            mask_paths = sorted(patient_dir.glob("*_OAR_*.seg.nrrd"))
            if not mask_paths:
                raise FileNotFoundError(f"No mask files found in {patient_dir}")
            oar_counts[case_idx] = len(mask_paths)
            if case_idx == 0:
                self.reference_oars = [p.name for p in mask_paths]
        
        # Determine expected OAR count
        self.expected_oars = oar_counts[0]
        logging.info(f"Expected number of OARs: {self.expected_oars}")
        
        # Log inconsistent cases only
        inconsistent_cases = [idx for idx, count in oar_counts.items() if count != self.expected_oars]
        if inconsistent_cases:
            logging.info(f"Found {len(inconsistent_cases)} cases with inconsistent OAR counts")
        
        # Build slice index (more efficient)
        logging.info("Building slice index")
        for case_idx, patient_dir in enumerate(self.patient_dirs):
            # Get CT slice count without loading full image
            ct_path = next(patient_dir.glob("*_IMG_CT.nrrd"), None)
            if ct_path is None:
                raise FileNotFoundError(f"Missing CT in {patient_dir}")
            
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(ct_path))
            reader.ReadImageInformation()
            D = reader.GetSize()[2]
            
            self.slice_counts[case_idx] = D
            for z in range(D):
                self.slices.append((case_idx, z))
            
            # Log progress every 5 cases
            if case_idx % 5 == 0 or case_idx == len(self.patient_dirs) - 1:
                logging.info(f"Processed {case_idx+1}/{len(self.patient_dirs)} cases")
        
        logging.info(f"Total slices: {len(self.slices)}")
        logging.info(f"Dataset initialization took {time.time() - start_time:.2f} seconds")
        logging.info(f"Memory usage: {get_memory_usage():.2f} GB")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        case_idx, z = self.slices[idx]
        cache_key = (case_idx, z)
        
        # Check in-memory cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Log loading every 50 slices to reduce output noise
        if idx % 50 == 0:
            logging.info(f"Loading slice {z} for case {case_idx}, idx={idx}")
        
        patient_dir = self.patient_dirs[case_idx]
        
        # Load CT slice
        ct_path = next(patient_dir.glob("*_IMG_CT.nrrd"))
        ct_img = sitk.ReadImage(str(ct_path), imageIO="NrrdImageIO", outputPixelType=sitk.sitkFloat32)
        ct_array = sitk.GetArrayViewFromImage(ct_img)[z:z+1, :, :]  # [1,H,W]
        ct_array = ct_array.astype(np.float32)
        
        # Load MR slice and resample
        mr_path = next(patient_dir.glob("*_IMG_MR*.nrrd"))
        mr_img = sitk.ReadImage(str(mr_path), imageIO="NrrdImageIO", outputPixelType=sitk.sitkFloat32)
        mr_img = sitk.Resample(mr_img, ct_img, interpolator=sitk.sitkLinear, defaultPixelValue=0.0)
        mr_array = sitk.GetArrayViewFromImage(mr_img)[z:z+1, :, :]  # [1,H,W]
        mr_array = mr_array.astype(np.float32)
        
        # Load masks
        mask_paths = sorted(patient_dir.glob("*_OAR_*.seg.nrrd"))
        masks = []
        for mask_path in mask_paths:
            mask_img = sitk.ReadImage(str(mask_path), imageIO="NrrdImageIO", outputPixelType=sitk.sitkUInt8)
            mask_img = sitk.Resample(mask_img, ct_img, interpolator=sitk.sitkNearestNeighbor, defaultPixelValue=0)
            mask_array = sitk.GetArrayViewFromImage(mask_img)[z:z+1, :, :]  # [1,H,W]
            masks.append(mask_array.astype(np.uint8))
            del mask_img, mask_array
        
        # Pad masks if needed
        while len(masks) < self.expected_oars:
            masks.append(np.zeros_like(masks[0], dtype=np.uint8))
        
        # Stack images and masks
        img_array = np.concatenate([ct_array, mr_array], axis=0)  # [2,H,W]
        mask_array = np.concatenate(masks, axis=0)  # [C,H,W]
        
        # Normalize
        img_array[0] = (img_array[0] + 1000) / 2000.0  # CT: [-1000, 1000] to [0, 1]
        mr_mean, mr_std = img_array[1].mean(), img_array[1].std() + 1e-8
        img_array[1] = np.clip((img_array[1] - mr_mean) / mr_std, -10, 10)  # MR: z-score
        
        # Convert to tensors and resize
        img_tensor = torch.from_numpy(img_array).float()  # [2,H,W]
        mask_tensor = torch.from_numpy(mask_array).float()  # [C,H,W]
        
        img_tensor = img_tensor.unsqueeze(0)  # [1,2,H,W]
        mask_tensor = mask_tensor.unsqueeze(0)  # [1,C,H,W]
        img_resized = F.interpolate(img_tensor, size=(self.img_size, self.img_size),
                                   mode='bilinear', align_corners=False).squeeze(0)  # [2,224,224]
        mask_resized = F.interpolate(mask_tensor, size=(self.img_size, self.img_size),
                                    mode='nearest').squeeze(0)  # [C,224,224]
        
        # Cache slice in memory
        self.cache[cache_key] = (img_resized, mask_resized)
        if len(self.cache) > self.cache_size_limit:
            self.cache.pop(list(self.cache.keys())[0])
        
        # Free memory
        del ct_img, mr_img, ct_array, mr_array, masks, img_array, mask_array, img_tensor, mask_tensor
        gc.collect()
        
        return img_resized, mask_resized

def dice_coefficient(pred, target, eps=1e-6):
    """Calculate Dice coefficient for binary segmentation"""
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    return (2 * intersection + eps) / (union + eps)

def train():
    # --- Check and print script start ---
    print("==== Script started ====")
    
    # --- Config ---
    ROOT_DIR = "./HaN-Seg"
    BATCH_SIZE = 2
    IMG_SIZE = 224
    LR = 1e-4
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0  # Avoid multiprocessing issues
    
    # Print Python version and major libraries
    logging.info(f"Python version: {sys.version}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Memory available: {get_memory_usage():.2f} GB")
    logging.info(f"Current working directory: {os.getcwd()}")
    
    # Check if data directory exists
    if os.path.exists(ROOT_DIR):
        logging.info(f"Data directory found: {ROOT_DIR}")
    else:
        logging.error(f"Data directory NOT found: {ROOT_DIR}")
        return
    
    start_time = time.time()
    logging.info("Starting training")
    
    try:
        # --- Create Dataset ---
        logging.info("Creating dataset")
        full_dataset = MemoryEfficientHaNSeg2DSliceDataset(ROOT_DIR, img_size=IMG_SIZE)
        
        # Case-based train/val/test split
        logging.info("Splitting cases")
        case_indices = []
        case_start_idx = 0
        for case_idx in range(len(full_dataset.patient_dirs)):
            D = full_dataset.slice_counts[case_idx]
            case_indices.append(list(range(case_start_idx, case_start_idx + D)))
            case_start_idx += D
        
        train_cases, test_cases = train_test_split(list(range(len(case_indices))), test_size=0.2, random_state=42)
        train_cases, val_cases = train_test_split(train_cases, test_size=0.25, random_state=42)
        
        train_indices = [idx for case_idx in train_cases for idx in case_indices[case_idx]]
        val_indices = [idx for case_idx in val_cases for idx in case_indices[case_idx]]
        test_indices = [idx for case_idx in test_cases for idx in case_indices[case_idx]]
        
        logging.info(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        # Create subset datasets
        logging.info("Creating data loaders")
        train_ds = Subset(full_dataset, train_indices)
        val_ds = Subset(full_dataset, val_indices)
        test_ds = Subset(full_dataset, test_indices)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                                num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=NUM_WORKERS, pin_memory=True)
        
        # --- Get sample to determine number of OARs ---
        logging.info("Fetching sample to determine OAR count")
        # Print progress to verify code is running
        print("Fetching sample...")
        sample_img, sample_mask = full_dataset[0]
        num_oars = sample_mask.shape[0]
        logging.info(f"Number of OARs: {num_oars}")
        
        # --- Model, Loss, Optimizer ---
        logging.info("Initializing model")
        model = TransformerSegmentation(
            img_size=IMG_SIZE,
            patch_size=16,
            in_channels=2,
            num_classes=num_oars,
            embed_dim=512,
            depth=8,
            num_heads=8,
        ).to(DEVICE)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0).to(DEVICE))
        optimizer = Adam(model.parameters(), lr=LR)
        
        # --- Training Loop with Validation ---
        logging.info("=== Starting training loop ===")
        best_val_dice = 0.0
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            logging.info(f"Starting epoch {epoch}/{EPOCHS}")
            # Print progress to verify code is running
            print(f"Starting epoch {epoch}/{EPOCHS}")
            
            model.train()
            train_loss = 0.0
            batch_count = 0
            
            for batch_idx, (imgs, masks) in enumerate(train_loader):
                batch_start = time.time()
                imgs = imgs.to(DEVICE)  # [B,2,224,224]
                masks = masks.to(DEVICE)  # [B,C,224,224]
                
                if DEVICE.type == "cuda" and batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
                    
                preds = model(imgs)
                loss = criterion(preds, masks)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * imgs.size(0)
                batch_count += 1
                
                # Log less frequently to reduce output volume
                if batch_idx == 0 or (batch_idx + 1) % 25 == 0 or batch_idx + 1 == len(train_loader):
                    logging.info(f"Epoch {epoch}/{EPOCHS}, Batch {batch_idx + 1}/{len(train_loader)}, "
                                f"Loss: {loss.item():.4f}, Batch time: {time.time() - batch_start:.2f}s")
                    # Print progress to verify code is running
                    print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
                del imgs, masks, preds, loss
                gc.collect()
            
            train_loss = train_loss / len(train_loader.dataset)
            
            # Validation
            logging.info(f"Validating epoch {epoch}")
            model.eval()
            val_loss = 0.0
            dice_scores = []
            
            with torch.no_grad():
                for batch_idx, (imgs, masks) in enumerate(val_loader):
                    imgs = imgs.to(DEVICE)
                    masks = masks.to(DEVICE)
                    
                    preds = model(imgs)
                    loss = criterion(preds, masks)
                    val_loss += loss.item() * imgs.size(0)
                    
                    preds_sigmoid = torch.sigmoid(preds)
                    preds_binary = (preds_sigmoid > 0.5).float()
                    
                    batch_dice = 0
                    for c in range(masks.size(1)):
                        dice = dice_coefficient(preds_binary[:, c], masks[:, c])
                        batch_dice += dice.item()
                    batch_dice /= masks.size(1)
                    dice_scores.append(batch_dice)
                    
                    # Log validation progress
                    if (batch_idx + 1) % 25 == 0 or batch_idx + 1 == len(val_loader):
                        print(f"Val batch {batch_idx + 1}/{len(val_loader)}")
                    
                    del imgs, masks, preds, preds_sigmoid, preds_binary
                    gc.collect()
            
            val_loss = val_loss / len(val_loader.dataset)
            val_dice = np.mean(dice_scores)
            
            epoch_time = time.time() - epoch_start
            logging.info(f"Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"Val Dice: {val_dice:.4f}, Time: {epoch_time:.1f}s")
            # Print progress to verify code is running
            print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                }, "best_model.pth")
                logging.info(f"Best model saved with dice {val_dice:.4f}")
            
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        
        # --- Testing Phase ---
        logging.info("Testing best model")
        checkpoint = torch.load("best_model.pth", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        test_dice_scores = []
        class_dice_scores = [[] for _ in range(num_oars)]
        
        with torch.no_grad():
            for batch_idx, (imgs, masks) in enumerate(test_loader):
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                
                preds = torch.sigmoid(model(imgs))
                preds_binary = (preds > 0.5).float()
                
                for c in range(masks.size(1)):
                    dice = dice_coefficient(preds_binary[:, c], masks[:, c])
                    class_dice_scores[c].append(dice.item())
                
                batch_dice = 0
                for c in range(masks.size(1)):
                    batch_dice += dice_coefficient(preds_binary[:, c], masks[:, c]).item()
                batch_dice /= masks.size(1)
                test_dice_scores.append(batch_dice)
                
                # Log test progress
                if (batch_idx + 1) % 25 == 0 or batch_idx + 1 == len(test_loader):
                    print(f"Test batch {batch_idx + 1}/{len(test_loader)}")
                
                del imgs, masks, preds, preds_binary
                gc.collect()
        
        final_test_dice = np.mean(test_dice_scores)
        logging.info(f"Final Test Dice Coefficient: {final_test_dice:.4f}")
        
        logging.info("Per-class Dice scores:")
        for c in range(num_oars):
            class_dice = np.mean(class_dice_scores[c])
            logging.info(f"Class {c}: {class_dice:.4f}")
        
        # --- Visualize prediction ---
        def visualize_prediction(model, dataset, idx, device, output_path="prediction.png"):
            model.eval()
            img, mask = dataset[idx]
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(img)
                pred = torch.sigmoid(pred).squeeze(0)
            
            img = img.squeeze(0).cpu()
            mask = mask.cpu()
            pred = pred.cpu()
            
            fig, axs = plt.subplots(2, 3, figsize=(12, 8))
            
            axs[0,0].imshow(img[0], cmap='gray')
            axs[0,0].set_title("CT")
            axs[0,0].axis('off')
            
            axs[0,1].imshow(img[1], cmap='gray')
            axs[0,1].set_title("MR")
            axs[0,1].axis('off')
            
            axs[0,2].imshow(img[0], cmap='gray')
            axs[0,2].imshow(img[1], cmap='hot', alpha=0.5)
            axs[0,2].set_title("CT+MR Overlay")
            axs[0,2].axis('off')
            
            axs[1,0].imshow(img[0], cmap='gray')
            axs[1,0].imshow(mask.sum(0) > 0, cmap='viridis', alpha=0.5)
            axs[1,0].set_title("Ground Truth")
            axs[1,0].axis('off')
            
            axs[1,1].imshow(img[0], cmap='gray')
            axs[1,1].imshow(pred.sum(0) > 0.5, cmap='plasma', alpha=0.5)
            axs[1,1].set_title("Prediction")
            axs[1,1].axis('off')
            
            gt = mask.sum(0) > 0
            pr = pred.sum(0) > 0.5
            diff = torch.zeros_like(gt, dtype=torch.float)
            diff[gt & pr] = 0.7
            diff[gt & (~pr)] = 1.0
            diff[(~gt) & pr] = 0.3
            
            axs[1,2].imshow(diff, cmap='gray')
            axs[1,2].set_title("Difference Map")
            axs[1,2].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            logging.info(f"Prediction visualization saved to {output_path}")
        
        logging.info("Generating visualizations")
        for i in range(min(3, len(test_ds))):
            visualize_prediction(model, test_ds, i, DEVICE, f"prediction_{i}.png")
        
        logging.info(f"Training completed, total time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        import traceback
        logging.error(traceback.format_exc())
        print(f"ERROR: {e}")

if __name__ == "__main__":
    print("Script starting...")
    train()
    print("Script finished")
