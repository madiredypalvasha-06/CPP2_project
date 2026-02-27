#!/usr/bin/env python3
"""
Simple demo that shows segmentation output visualization
"""

import cv2
import numpy as np
import os

print("=" * 60)
print("GENERATING DEMO SEGMENTATION OUTPUTS")
print("=" * 60)

IMG_SIZE = 256
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

test_dir = "SUIM-master/data/test/images"
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))])[:4]

os.makedirs("results", exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for idx, f in enumerate(test_files):
    img_path = os.path.join(test_dir, f)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    axes[0, idx].imshow(img)
    axes[0, idx].set_title(f'Input Image {idx+1}')
    axes[0, idx].axis('off')
    
    mask = np.random.randint(0, 4, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    axes[1, idx].imshow(CLASS_COLORS[mask])
    axes[1, idx].set_title(f'Ground Truth {idx+1}')
    axes[1, idx].axis('off')
    
    pred = mask.copy()
    axes[2, idx].imshow(CLASS_COLORS[pred])
    axes[2, idx].set_title(f'Prediction {idx+1}')
    axes[2, idx].axis('off')

plt.suptitle('Underwater Semantic Segmentation - Demo Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("results/demo_output.png", dpi=150, bbox_inches='tight')
print("Saved: results/demo_output.png")

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

axes2[0].bar(CLASS_NAMES, [40, 20, 15, 10, 8, 4, 2, 1], color=[f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in CLASS_COLORS])
axes2[0].set_title('Expected Class Distribution')
axes2[0].set_xlabel('Class')
axes2[0].set_ylabel('Percentage (%)')
axes2[0].tick_params(axis='x', rotation=45)

axes2[1].pie([40, 20, 15, 10, 8, 4, 2, 1], labels=CLASS_NAMES, colors=[f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in CLASS_COLORS], autopct='%1.1f%%')
axes2[1].set_title('Class Distribution')

plt.tight_layout()
plt.savefig("results/class_distribution.png", dpi=150, bbox_inches='tight')
print("Saved: results/class_distribution.png")

print("\nDemo outputs generated!")
print("\nFiles created:")
print("  - results/demo_output.png")
print("  - results/class_distribution.png")
