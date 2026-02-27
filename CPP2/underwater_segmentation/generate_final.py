#!/usr/bin/env python3
"""
Generate proper colored segmentation outputs
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

print("="*60)
print("GENERATING SEGMENTATION OUTPUTS")
print("="*60)

IMG_SIZE = 128
NUM_CLASSES = 8

CLASS_NAMES = ['Background', 'Fish', 'Plants', 'Rocks', 'Coral', 'Wrecks', 'Water', 'Other']
CLASS_COLORS = np.array([
    [255, 255, 255],
    [255, 0, 0],
    [0, 255, 0],
    [128, 128, 128],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 128, 0]
], dtype=np.uint8)

DATA_PATH = os.path.expanduser("~/.cache/kagglehub/datasets/ashish2001/semantic-segmentation-of-underwater-imagery-suim/versions/2")

print("\nLoading model...")
model = tf.keras.models.load_model("checkpoints/segmentation_final.keras", compile=False)
print("Model loaded!")

print("\nLoading test images...")
images_dir = os.path.join(DATA_PATH, "train_val", "images")
masks_dir = os.path.join(DATA_PATH, "train_val", "masks")

files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])[:8]

images = []
masks = []
filenames = []

for f in files:
    img = cv2.imread(os.path.join(images_dir, f))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    images.append(img)
    
    base = os.path.splitext(f)[0]
    mp = os.path.join(masks_dir, base + ".jpg")
    if not os.path.exists(mp): mp = os.path.join(masks_dir, base + ".png")
    
    if os.path.exists(mp):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    else:
        m = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    masks.append(m)
    filenames.append(f)

images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.uint8)

print("\nGenerating predictions...")
preds = model.predict(images, verbose=0)
pred_masks = np.argmax(preds, axis=-1)

print("\nClass distribution in predictions:")
for i in range(8):
    unique, counts = np.unique(pred_masks[i], return_counts=True)
    print(f"Image {i}: {dict(zip(unique, counts))}")

print("\nSaving outputs...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i in range(4):
    axes[0, i].imshow(images[i])
    axes[0, i].set_title(f'Input {i+1}')
    axes[0, i].axis('off')
    
    gt_colored = CLASS_COLORS[masks[i] % 8]
    axes[1, i].imshow(gt_colored)
    axes[1, i].set_title(f'Ground Truth {i+1}')
    axes[1, i].axis('off')
    
    pred_colored = CLASS_COLORS[pred_masks[i]]
    axes[2, i].imshow(pred_colored)
    axes[2, i].set_title(f'Prediction {i+1}')
    axes[2, i].axis('off')

plt.suptitle('Underwater Semantic Segmentation Results', fontsize=16, fontweight='bold')
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/segmentation_output.png", dpi=150, bbox_inches='tight')
print("Saved: results/segmentation_output.png")

print("\n" + "="*60)
print("DONE! Check results/segmentation_output.png")
print("="*60)
