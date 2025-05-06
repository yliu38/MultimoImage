# Multimodal Image Analysis
# TransformerHaNSeg: Multimodal Transformer for Head & Neck OAR Segmentation

![Segmentation Example](sample_slice.png)

## Project Overview

This repository contains a deep learning solution for automatic segmentation of Organs at Risk (OARs) in Head and Neck (HaN) CT and MR images using a Vision Transformer architecture. The model leverages both CT and MR modalities to improve segmentation accuracy across 30 different anatomical structures.

### Key Features

- **Multimodal Image Fusion**: Combines CT and MR images for comprehensive anatomical representation
- **Vision Transformer Architecture**: Leverages self-attention mechanisms for capturing global relationships
- **Memory-Efficient Implementation**: Optimized data loading and processing pipeline for large 3D medical images
- **Case-Based Data Splitting**: Ensures generalization to unseen patient data

## Results

The model achieved excellent segmentation performance with an average Dice coefficient of **0.8779** across all OARs on the test set. Performance varies across different structures:

| Performance Category | Structures | Dice Coefficient |
|---------------------|------------|------------------|
| Excellent (>0.95)   | 10 structures | 0.95-0.99 |
| Very Good (0.90-0.95) | 9 structures | 0.90-0.95 |
| Good (0.80-0.90) | 6 structures | 0.80-0.90 |
| Moderate (0.60-0.80) | 3 structures | 0.60-0.80 |
| Challenging (<0.60) | 2 structures | 0.44-0.60 |

Top performing structures (Dice > 0.97):
- Structure 7: 0.9780
- Structure 8: 0.9780 
- Structure 23: 0.9871
- Structure 2: 0.9728
- Structure 28: 0.9715

Most challenging structures:
- Structure 0: 0.4437
- Structure 1: 0.4903
- Structure 29: 0.6003

## Technical Details

### Dataset

The HaN-Seg dataset consists of paired CT and T1-weighted MR images with expert segmentations of 30 organs at risk in the head and neck region. The dataset includes:

- Multiple patient cases from set_1
- CT images (base reference for spatial coordinates)
- T1-weighted MR images (resampled to CT space)
- Manual segmentations for 30 OARs

### Model Architecture

The segmentation model is based on a Vision Transformer architecture with the following components:

1. **Patch Embedding**: Converts 2D image slices into sequence of patch tokens
2. **Transformer Encoder**: Applies self-attention across all patches
   - Depth: 8 layers
   - Number of heads: 8
   - Embedding dimension: 512
3. **Segmentation Decoder**: Reconstructs full-resolution segmentation masks

```
TransformerSegmentation(
    img_size=224,
    patch_size=16,
    in_channels=2,  # CT + MR
    num_classes=30,
    embed_dim=512,
    depth=8,
    num_heads=8
)
```

### Training Setup

- **Optimizer**: Adam with learning rate 1e-4
- **Loss Function**: Binary Cross-Entropy with Logits (weighted for class imbalance)
- **Training Strategy**: 
  - Case-based train/validation/test split (60/20/20)
  - Batch size: 2
  - Epochs: 5
  - Image size: 224×224

### Memory Optimization

The implementation includes several memory optimization techniques:

- Slice-by-slice loading of volumetric data
- Limited in-memory caching with LRU strategy
- Gradient clipping to stabilize training
- Strategic garbage collection to free unused memory

## Code Structure

- `transformer.py`: Contains the Vision Transformer implementation
- `train.py`: Main training script with memory optimizations
- `preprocessing.py`: Data loading and preprocessing pipeline
- `display.py`: Visualization utilities

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- SimpleITK
- NumPy
- Matplotlib
- scikit-learn

### Installation

```bash
pip install -r requirements.txt
```

### Preprocessing images
```bash
python preprocessing.py
```

### Visualize sample images

```bash
python display.py
```

### Training

```bash
python train.py
```


## Future Work

- [ ] Implement 3D transformer to better capture volumetric context
- [ ] Integrate attention visualization for model interpretability
- [ ] Add transfer learning to improve performance on challenging structures
- [ ] Explore additional modalities (e.g., PET) for further improvement

## Acknowledgements

This project utilizes the HaN-Seg dataset for model training and evaluation. Special thanks to the medical imaging team for providing the annotated data.

