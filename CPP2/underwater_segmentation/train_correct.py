#!/usr/bin/env python3
"""
Train with CORRECT SUIM RGB mask mapping
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("="*60)
print("TRAINING WITH CORRECT SUIM MAPPING")
print("="*60)

IMG_SIZE = 256
NUM_CLASSES = 8

# Correct SUIM mapping: RGB -> class
# BW=(0,0,0), HD=(0,0,1), PF=(0,1,0), WR=(0,1,1), RO=(1,0,0), RI=(1,0,1), FV=(1,1,0), SR=(1,1,1)
CLASS_NAMES = ['Background', 'Human Divers', 'Plants', 'Wrecks', 'Robots', 'Reefs', 'Fish', 'Sea-floor']

def rgb_to_class(mask):
    h, w = mask.shape[:2]
    classes = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            pixel = mask[r, c]
            b, g, pixel_r = pixel[0], pixel[1], pixel[2]
            class_idx = (pixel_r > 127) * 4 + (g > 127) * 2 + (b > 127)
            classes[r, c] = class_idx
    return classes

DATA_PATH = os.path.expanduser("~/.cache/kagglehub/datasets/ashish2001/semantic-segmentation-of-underwater-imagery-suim/versions/2")

print("\n[1/4] Loading data...")
images_dir = os.path.join(DATA_PATH, "train_val", "images")
masks_dir = os.path.join(DATA_PATH, "train_val", "masks")

files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])[:500]
images, masks = [], []

for f in files:
    img = cv2.imread(os.path.join(images_dir, f))
    if img is None: continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    base = os.path.splitext(f)[0]
    mp = os.path.join(masks_dir, base + ".jpg")
    if not os.path.exists(mp): mp = os.path.join(masks_dir, base + ".bmp")
    
    if os.path.exists(mp):
        m = cv2.imread(mp)
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        m = rgb_to_class(m)
    else:
        m = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    images.append(img)
    masks.append(m)

images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.uint8)
print(f"Loaded: {len(images)} images")
print(f"Unique classes: {np.unique(masks)}")

print("\n[2/4] Creating model...")
inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inp)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Dropout(0.2)(x)

x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Dropout(0.2)(x)

x = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Dropout(0.3)(x)

x = keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
x = keras.layers.BatchNormalization()(x)

x = keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Dropout(0.3)(x)

x = keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Dropout(0.2)(x)

x = keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)

out = keras.layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)

model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n[3/4] Training (20 epochs)...")
history = model.fit(images, masks, epochs=20, batch_size=8, validation_split=0.1, verbose=2)

print("\n[4/4] Saving and generating outputs...")
os.makedirs("checkpoints", exist_ok=True)
model.save("checkpoints/segmentation_final.keras")

preds = model.predict(images[:8], verbose=0)
pred_masks = np.argmax(preds, axis=-1)
print(f"Prediction unique: {np.unique(pred_masks)}")

CLASS_COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [135, 206, 235],
    [128, 128, 128],
    [255, 192, 203],
    [255, 255, 0],
    [255, 255, 255]
], dtype=np.uint8)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i in range(4):
    axes[0,i].imshow(images[i])
    axes[0,i].set_title(f'Input {i+1}')
    axes[0,i].axis('off')
    axes[1,i].imshow(CLASS_COLORS[masks[i]])
    axes[1,i].set_title(f'Ground Truth {i+1}')
    axes[1,i].axis('off')
    axes[2,i].imshow(CLASS_COLORS[pred_masks[i]])
    axes[2,i].set_title(f'Prediction {i+1}')
    axes[2,i].axis('off')

plt.suptitle('U-Net - Correctly Trained', fontsize=16)
plt.tight_layout()
plt.savefig("results/segmentation_results.png", dpi=150)
print("Saved: results/segmentation_results.png")

acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"\nTraining: {acc*100:.2f}%")
print(f"Validation: {val_acc*100:.2f}%")
print("\nDONE!")
