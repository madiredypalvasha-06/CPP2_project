#!/usr/bin/env python3
"""
Enhanced Training with synthetic underwater images
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

print("=" * 60)
print("ENHANCED TRAINING WITH SYNTHETIC UNDERWATER IMAGES")
print("=" * 60)

IMG_SIZE = 256
NUM_CLASSES = 8

CLASS_VALUES = {0: 0, 29: 1, 76: 2, 105: 3, 150: 4, 200: 5, 225: 6, 250: 7}

def map_mask(mask):
    new_mask = np.zeros_like(mask)
    for val, cls in CLASS_VALUES.items():
        new_mask[mask == val] = cls
    return new_mask

def load_data(data_dir="SUIM-master/data/test"):
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Found {len(image_files)} images")
    
    images = []
    masks = []
    
    for img_file in image_files:
        img = cv2.imread(os.path.join(images_dir, img_file))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        images.append(img)
        
        base = os.path.splitext(img_file)[0]
        mask_path = os.path.join(masks_dir, base + ".bmp")
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            mask = map_mask(mask)
        else:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        
        masks.append(mask)
    
    return np.array(images), np.array(masks)

def synthesize_underwater(img, mask):
    synth_imgs = []
    synth_masks = []
    
    for _ in range(8):
        new_img = img.copy()
        new_mask = mask.copy()
        
        if random.random() > 0.5:
            new_img = np.fliplr(new_img)
            new_mask = np.fliplr(new_mask)
        
        if random.random() > 0.5:
            angle = random.choice([1, 2, 3])
            new_img = np.rot90(new_img, angle)
            new_mask = np.rot90(new_mask, angle)
        
        brightness = random.uniform(0.6, 1.4)
        new_img = np.clip(new_img * brightness, 0, 1)
        
        if random.random() > 0.5:
            blue_shift = random.uniform(0, 0.2)
            new_img[:, :, 0] = np.clip(new_img[:, :, 0] + blue_shift, 0, 1)
        
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.02, new_img.shape).astype(np.float32)
            new_img = np.clip(new_img + noise, 0, 1)
        
        if random.random() > 0.7:
            blur_k = random.choice([3, 5])
            new_img = cv2.GaussianBlur(new_img, (blur_k, blur_k), 0)
        
        synth_imgs.append(new_img)
        synth_masks.append(new_mask)
    
    return synth_imgs, synth_masks

def create_improved_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.35)(x)
    
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.35)(x)
    
    x = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

def main():
    print("\n1. Loading original data...")
    images, masks = load_data()
    print(f"Original: {len(images)} images")
    
    print("\n2. Generating synthetic underwater images...")
    all_images = list(images)
    all_masks = list(masks)
    
    for img, mask in zip(images, masks):
        synth_imgs, synth_masks = synthesize_underwater(img, mask)
        all_images.extend(synth_imgs)
        all_masks.extend(synth_masks)
    
    all_images = np.array(all_images)
    all_masks = np.array(all_masks)
    print(f"Total after synthesis: {len(all_images)} images")
    
    print("\n3. Creating improved model...")
    model = create_improved_model()
    model.summary()
    
    print("\n4. Compiling...")
    model.compile(
        optimizer=keras.optimizers.Adam(5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n5. Training (80 epochs)...")
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    history = model.fit(
        all_images, all_masks,
        epochs=80,
        batch_size=8,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=2
    )
    
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/segmentation_final.keras")
    print("\nModel saved to checkpoints/segmentation_final.keras")
    
    print("\n6. Evaluating...")
    preds = model.predict(all_images[:8], verbose=0)
    pred_masks = np.argmax(preds, axis=-1)
    
    ious = []
    for i in range(min(8, len(all_images))):
        for cls in range(NUM_CLASSES):
            true = (all_masks[i] == cls).astype(float)
            pred = (pred_masks[i] == cls).astype(float)
            inter = np.logical_and(true, pred).sum()
            union = np.logical_or(true, pred).sum()
            if union > 0:
                ious.append(inter / union)
    
    miou = np.mean(ious) if ious else 0
    print(f"Mean IoU: {miou:.4f}")
    
    CLASS_NAMES = ['Background', 'Fish', 'Plants', 'Rocks', 'Coral', 'Wrecks', 'Water', 'Other']
    CLASS_COLORS = np.array([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [128, 128, 128],
        [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 128, 0]
    ], dtype=np.uint8)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for i in range(4):
        axes[0, i].imshow(all_images[i])
        axes[0, i].set_title('Input')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(CLASS_COLORS[all_masks[i]])
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(CLASS_COLORS[pred_masks[i]])
        axes[2, i].set_title('Prediction')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/segmentation_output.png", dpi=150, bbox_inches='tight')
    print("Saved: results/segmentation_output.png")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
