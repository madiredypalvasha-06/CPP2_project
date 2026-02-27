#!/usr/bin/env python3
"""
Quick train with full SUIM dataset - 10 epochs
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("=" * 60)
print("QUICK TRAINING WITH FULL DATASET - 10 EPOCHS")
print("=" * 60)

IMG_SIZE = 256
NUM_CLASSES = 8

CLASS_VALUES = {0: 0, 29: 1, 76: 2, 105: 3, 150: 4, 200: 5, 225: 6, 250: 7}

DATA_PATH = os.path.expanduser("~/.cache/kagglehub/datasets/ashish2001/semantic-segmentation-of-underwater-imagery-suim/versions/2")

def map_mask(mask):
    new_mask = np.zeros_like(mask)
    for val, cls in CLASS_VALUES.items():
        new_mask[mask == val] = cls
    return new_mask

print("\n1. Loading data...")
images_dir = os.path.join(DATA_PATH, "train_val", "images")
masks_dir = os.path.join(DATA_PATH, "train_val", "masks")

image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
images = []
masks = []

for i, img_file in enumerate(image_files):
    if i % 200 == 0:
        print(f"Loading {i}/{len(image_files)}...")
    
    img = cv2.imread(os.path.join(images_dir, img_file))
    if img is None:
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    base = os.path.splitext(img_file)[0]
    mask_path = os.path.join(masks_dir, base + ".jpg")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(masks_dir, base + ".png")
    
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = map_mask(mask)
    else:
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    images.append(img)
    masks.append(mask)

images = np.array(images)
masks = np.array(masks)
print(f"Loaded: {len(images)} images")

print("\n2. Creating model...")
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
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

outputs = keras.layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n3. Training (10 epochs)...")
model.fit(images, masks, epochs=10, batch_size=16, validation_split=0.1, verbose=2)

print("\n4. Saving model...")
os.makedirs("checkpoints", exist_ok=True)
model.save("checkpoints/segmentation_final.keras")
print("Model saved!")

print("\n5. Generating results...")
preds = model.predict(images[:8], verbose=0)
pred_masks = np.argmax(preds, axis=-1)

CLASS_COLORS = np.array([[0,0,0],[255,0,0],[0,255,0],[128,128,128],[255,255,0],[255,0,255],[0,255,255],[255,128,0]], dtype=np.uint8)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
for i in range(4):
    axes[0,i].imshow(images[i])
    axes[0,i].set_title('Input')
    axes[0,i].axis('off')
    axes[1,i].imshow(CLASS_COLORS[masks[i]])
    axes[1,i].set_title('Ground Truth')
    axes[1,i].axis('off')
    axes[2,i].imshow(CLASS_COLORS[pred_masks[i]])
    axes[2,i].set_title('Prediction')
    axes[2,i].axis('off')

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/segmentation_output.png", dpi=150)
print("Saved: results/segmentation_output.png")

print("\nDONE!")
