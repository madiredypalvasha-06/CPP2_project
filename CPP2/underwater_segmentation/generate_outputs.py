#!/usr/bin/env python3
"""
Generate segmentation outputs and improve accuracy
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

print("=" * 60)
print("SEGMENTATION OUTPUT GENERATOR")
print("=" * 60)

IMG_SIZE = 256
NUM_CLASSES = 8
CLASS_NAMES = ['Background', 'Fish', 'Plants', 'Rocks', 'Coral', 'Wrecks', 'Water', 'Other']
CLASS_COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [128, 128, 128],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 128, 0]
], dtype=np.uint8)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

print("\n📦 Loading model...")
model = tf.keras.models.load_model("checkpoints/segmentation_final.keras", compile=False)
print("✅ Model loaded!")

def preprocess(img_path):
    img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img.astype(np.float32) / 255.0

def predict(img):
    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch, verbose=0)
    mask = np.argmax(pred[0], axis=-1)
    return mask

def get_class_distribution(mask):
    unique, counts = np.unique(mask, return_counts=True)
    total = counts.sum()
    distribution = {}
    for cls, cnt in zip(unique, counts):
        percentage = (cnt / total) * 100
        distribution[CLASS_NAMES[cls]] = percentage
    return distribution

os.makedirs("results", exist_ok=True)

print("\n📊 Processing test images...")

test_dir = "SUIM-master/data/test/images"
if os.path.exists(test_dir):
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))])[:8]
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    
    for idx, f in enumerate(test_files):
        img_path = os.path.join(test_dir, f)
        img = preprocess(img_path)
        mask = predict(img)
        
        ax = axes[0, idx]
        ax.imshow(img)
        ax.set_title('Input')
        ax.axis('off')
        
        ax = axes[1, idx]
        ax.imshow(mask, cmap='tab10')
        ax.set_title('Segmentation')
        ax.axis('off')
        
        ax = axes[2, idx]
        colored = CLASS_COLORS[mask]
        ax.imshow(colored)
        ax.set_title('Color Mask')
        ax.axis('off')
        
        overlay = (img * 0.5 + colored.astype(np.float32) / 255 * 0.5)
        ax = axes[3, idx]
        ax.imshow(overlay)
        ax.set_title('Overlay')
        ax.axis('off')
        
        dist = get_class_distribution(mask)
        print(f"\n{f}:")
        for cls, pct in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            if pct > 0.1:
                print(f"  {cls}: {pct:.1f}%")
    
    plt.tight_layout()
    plt.savefig("results/segmentation_output.png", dpi=150, bbox_inches='tight')
    print("\n✅ Saved: results/segmentation_output.png")

print("\n📁 Testing with custom image...")
test_imgs = [
    "SUIM-master/data/test/images/0000.jpg",
    "SUIM-master/data/test/images/0001.jpg",
]

for img_path in test_imgs:
    if os.path.exists(img_path):
        img = preprocess(img_path)
        mask = predict(img)
        
        colored = CLASS_COLORS[mask]
        cv2.imwrite("results/result_" + os.path.basename(img_path).replace('.jpg', '_mask.png'), 
                    cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved: results/result_{os.path.basename(img_path).replace('.jpg', '_mask.png')}")

print("\n" + "=" * 60)
print("OUTPUT IMAGES SAVED IN: results/")
print("=" * 60)
print("\nTo view images, open:")
print("  - results/segmentation_output.png")
print("  - results/result_*.png")
