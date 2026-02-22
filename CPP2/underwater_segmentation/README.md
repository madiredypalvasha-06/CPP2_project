# Underwater Semantic Segmentation

## Overview

This project implements a comprehensive underwater semantic segmentation framework using:
- **U-Net**: Encoder-decoder architecture with skip connections
- **DeepLabV3+**: ASPP module for multi-scale context extraction  
- **Hybrid Model**: Fusion of both architectures for improved accuracy

## Dataset

Uses SUIM (Semantic Underwater Image Dataset) with 6 classes:
- Background, Fish, Plants, Rocks, Coral, Wrecks

## Project Structure

```
underwater_segmentation/
├── train.py              # Main training script
├── README.md            # This file
├── checkpoints/         # Saved model weights
│   ├── unet_model.keras
│   └── deeplab_model.keras
└── results/            # Output visualizations
    ├── segmentation_results.png
    ├── training_history.png
    ├── per_class_metrics.png
    ├── class_legend.png
    └── metrics.txt
```

## Usage

### Training

```bash
cd underwater_segmentation
python3 train.py
```

### Configuration

Edit the `Config` class in `train.py` to modify:
- `EPOCHS`: Number of training epochs (default: 30)
- `BATCH_SIZE`: Batch size (default: 2)
- `LEARNING_RATE`: Learning rate (default: 1e-4)

## Output Files

| File | Description |
|------|-------------|
| `segmentation_results.png` | Visual comparison of all models |
| `training_history.png` | Loss and accuracy curves |
| `per_class_metrics.png` | IoU comparison per class |
| `class_legend.png` | Color legend for classes |
| `metrics.txt` | Detailed metrics in text format |

## Evaluation Metrics

- **Mean IoU**: Intersection over Union across all classes
- **Dice Score**: F1 score for segmentation quality
- **Pixel Accuracy**: Overall pixel-wise accuracy

## Requirements

- Python 3.11+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
