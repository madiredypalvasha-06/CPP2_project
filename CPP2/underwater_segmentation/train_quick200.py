#!/usr/bin/env python3
"""
Quick Training - 200 images, 256x256, 25 epochs
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("="*60)
print("QUICK TRAINING - 200 IMAGES")
print("="*60)

IMG_SIZE = 256
NUM_CLASSES = 8

CLASS_VALUES = {0: 0, 29: 1, 76: 2, 105: 3, 150: 4, 200: 5, 225: 6, 250: 7}

DATA_PATH = os.path.expanduser("~/.cache/kagglehub/datasets/ashish2001/semantic-segmentation-of-underwater-imagery-suim/versions/2")

print("\n[1/4] Loading 200 images...")
images_dir = os.path.join(DATA_PATH, "train_val", "images")
masks_dir = os.path.join(DATA_PATH, "train_val", "masks")

files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])[:200]
images, masks = [], []

for f in files:
    img = cv2.imread(os.path.join(images_dir, f))
    if img is None: continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    base = os.path.splitext(f)[0]
    mp = os.path.join(masks_dir, base + ".jpg")
    if not os.path.exists(mp): mp = os.path.join(masks_dir, base + ".png")
    
    if os.path.exists(mp):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        nm = np.zeros_like(m)
        for v, c in CLASS_VALUES.items(): nm[m == v] = c
        m = nm
    else:
        m = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    images.append(img)
    masks.append(m)

images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.uint8)
print(f"Loaded: {len(images)} images")

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

print("\n[3/4] Training (25 epochs)...")
history = model.fit(images, masks, epochs=25, batch_size=8, validation_split=0.1, verbose=2)

print("\n[4/4] Saving model and generating outputs...")
os.makedirs("checkpoints", exist_ok=True)
model.save("checkpoints/segmentation_final.keras")

preds = model.predict(images[:8], verbose=0)
pred_masks = np.argmax(preds, axis=-1)

print("Pred unique:", np.unique(pred_masks))

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

plt.suptitle('Underwater Semantic Segmentation Results', fontsize=16)
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/segmentation_output.png", dpi=150)
print("Saved: results/segmentation_output.png")

acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"\nTraining Accuracy: {acc*100:.2f}%")
print(f"Validation Accuracy: {val_acc*100:.2f}%")
print("\nDONE!")
