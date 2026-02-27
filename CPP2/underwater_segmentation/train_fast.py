#!/usr/bin/env python3
"""
Super fast training - 128x128 images
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("FAST TRAINING - 128x128")
print("=" * 50)

IMG_SIZE = 128
NUM_CLASSES = 8

CLASS_VALUES = {0: 0, 29: 1, 76: 2, 105: 3, 150: 4, 200: 5, 225: 6, 250: 7}

DATA_PATH = os.path.expanduser("~/.cache/kagglehub/datasets/ashish2001/semantic-segmentation-of-underwater-imagery-suim/versions/2")

print("Loading data...")
images_dir = os.path.join(DATA_PATH, "train_val", "images")
masks_dir = os.path.join(DATA_PATH, "train_val", "masks")

image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])[:500]
images, masks = [], []

for i, img_file in enumerate(image_files):
    img = cv2.imread(os.path.join(images_dir, img_file))
    if img is None: continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    base = os.path.splitext(img_file)[0]
    mask_path = os.path.join(masks_dir, base + ".jpg")
    if not os.path.exists(mask_path): mask_path = os.path.join(masks_dir, base + ".png")
    
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        new_mask = np.zeros_like(mask)
        for val, cls in CLASS_VALUES.items(): new_mask[mask == val] = cls
        mask = new_mask
    else:
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    images.append(img)
    masks.append(mask)

images, masks = np.array(images), np.array(masks)
print(f"Loaded: {len(images)} images")

print("Creating model...")
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)
x = keras.layers.Conv2DTranspose(32, 2, strides=2, padding='same')(x)
outputs = keras.layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training (5 epochs)...")
model.fit(images, masks, epochs=5, batch_size=16, verbose=2)

print("Saving...")
os.makedirs("checkpoints", exist_ok=True)
model.save("checkpoints/segmentation_final.keras")

print("Generating results...")
preds = model.predict(images[:4], verbose=0)
pred_masks = np.argmax(preds, axis=-1)

CLASS_COLORS = np.array([[0,0,0],[255,0,0],[0,255,0],[128,128,128],[255,255,0],[255,0,255],[0,255,255],[255,128,0]], dtype=np.uint8)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i in range(4):
    axes[0,i].imshow(images[i]); axes[0,i].axis('off'); axes[0,i].set_title('Input')
    axes[1,i].imshow(CLASS_COLORS[masks[i]]); axes[1,i].axis('off'); axes[1,i].set_title('GT')
    axes[2,i].imshow(CLASS_COLORS[pred_masks[i]]); axes[2,i].axis('off'); axes[2,i].set_title('Pred')
plt.tight_layout()
plt.savefig("results/segmentation_output.png", dpi=150)
print("Done! Saved results/segmentation_output.png")
