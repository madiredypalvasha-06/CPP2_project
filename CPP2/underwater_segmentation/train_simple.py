#!/usr/bin/env python3
"""
Simplified Training - Fixed for proper mask handling
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 60)
print("SIMPLIFIED TRAINING - FIXED")
print("=" * 60)

IMG_SIZE = 256
NUM_CLASSES = 8


def load_and_prepare_data(data_dir="SUIM-master/data/test"):
    """Load data and prepare for training"""
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Found {len(image_files)} images")
    
    images = []
    masks = []
    
    for img_file in image_files:
        # Load image
        img = cv2.imread(os.path.join(images_dir, img_file))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        images.append(img)
        
        # Load mask
        base = os.path.splitext(img_file)[0]
        mask = None
        
        # Try different mask locations
        for ext in ['.bmp', '.png']:
            for subdir in ['', 'FV', 'HD', 'RI', 'RO', 'WR', 'PF', 'SR']:
                if subdir:
                    path = os.path.join(masks_dir, subdir, base + ext)
                else:
                    path = os.path.join(masks_dir, base + ext)
                if os.path.exists(path):
                    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    break
            if mask is not None:
                break
        
        if mask is None:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        else:
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask is 0-7
        mask = mask % NUM_CLASSES
        masks.append(mask)
    
    images = np.array(images)
    masks = np.array(masks)
    
    # Verify
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Mask unique values: {np.unique(masks)}")
    
    return images, masks


def augment(images, masks):
    """Augment data"""
    aug_img = []
    aug_mask = []
    
    for img, mask in zip(images, masks):
        # Original
        aug_img.append(img)
        aug_mask.append(mask)
        
        # Horizontal flip
        aug_img.append(np.fliplr(img))
        aug_mask.append(np.fliplr(mask))
        
        # Vertical flip
        aug_img.append(np.flipud(img))
        aug_mask.append(np.flipud(mask))
        
        # Rotation 90
        aug_img.append(np.rot90(img))
        aug_mask.append(np.rot90(mask))
        
        # Rotate 180
        aug_img.append(np.rot90(img, 2))
        aug_mask.append(np.rot90(mask, 2))
        
        # Rotate 270
        aug_img.append(np.rot90(img, 3))
        aug_mask.append(np.rot90(mask, 3))
        
        # Brightness variations
        for factor in [0.9, 1.1]:
            aug_img.append(np.clip(img * factor, 0, 1))
            aug_mask.append(mask)
    
    return np.array(aug_img), np.array(aug_mask)


def create_model():
    """Create segmentation model"""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Encoder
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder
    x = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)


def main():
    print("\n1. Loading data...")
    images, masks = load_and_prepare_data()
    
    print("\n2. Augmenting...")
    images, masks = augment(images, masks)
    print(f"Total samples: {len(images)}")
    
    print("\n3. Creating model...")
    model = create_model()
    model.summary()
    
    print("\n4. Compiling...")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n5. Training...")
    history = model.fit(
        images, masks,
        epochs=25,
        batch_size=4,
        validation_split=0.15,
        verbose=2
    )
    
    # Save
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/segmentation_final.keras")
    print("\nModel saved to checkpoints/segmentation_final.keras")
    
    # Test
    print("\n6. Evaluating...")
    preds = model.predict(images[:8], verbose=0)
    pred_masks = np.argmax(preds, axis=-1)
    
    # Calculate IoU
    ious = []
    for i in range(min(8, len(images))):
        for cls in range(NUM_CLASSES):
            true = (masks[i] == cls).astype(float)
            pred = (pred_masks[i] == cls).astype(float)
            inter = np.logical_and(true, pred).sum()
            union = np.logical_or(true, pred).sum()
            if union > 0:
                ious.append(inter / union)
    
    miou = np.mean(ious) if ious else 0
    print(f"Mean IoU: {miou:.4f}")
    
    # Save visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for i in range(4):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title('Input')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(masks[i])
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(pred_masks[i])
        axes[2, i].set_title('Prediction')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/segmentation_output.png", dpi=100)
    print("Saved visualization to results/segmentation_output.png")


if __name__ == "__main__":
    main()
