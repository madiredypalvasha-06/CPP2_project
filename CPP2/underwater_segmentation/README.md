# Underwater Semantic Segmentation Project

## Project Summary
Deep learning-based semantic segmentation for underwater marine object detection using SUIM dataset.

## Models Available
1. **U-Net Enhanced** - Working (used in website)
2. **DeepLabV3+** - Requires 256x256 input (compatible)
3. **U-Net** - Requires older TensorFlow version
4. **Attention U-Net** - Requires older TensorFlow version

## Dataset
- **SUIM Dataset** - 1525 training images, 8 classes:
  - Background, Fish, Plants, Rocks, Coral, Wrecks, Water, Other

## How to Run

### 1. Website
```bash
cd /Users/palvashamadireddy/Downloads/CPP2_project/CPP2/underwater_segmentation
source venv311/bin/activate
python app.py
```
Then open http://127.0.0.1:5001

### 2. Generate Outputs
```bash
source venv311/bin/activate
python generate_final.py
```

### 3. Train New Model
```bash
source venv311/bin/activate
python train_quick200.py
```

## Output Images (results/)
- `segmentation_output.png` - Main segmentation results
- `model_comparison.png` - Model comparison
- `training_curves.png` - Training metrics
- `class_distribution.png` - Class distribution

## Files
- `app.py` - Flask website
- `train_quick200.py` - Training script
- `generate_final.py` - Generate outputs
- `checkpoints/segmentation_final.keras` - Trained model
- `checkpoints/unet_enhanced.keras` - U-Net Enhanced model
- `checkpoints/deeplabv3plus.keras` - DeepLabV3+ model

## Note
Some models (unet.keras, attention_unet.keras) require older TensorFlow version and cannot be loaded with current version.
