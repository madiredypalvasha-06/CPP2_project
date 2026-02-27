#!/usr/bin/env python3
"""
Improved Training with proper mask handling
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 60)
print("IMPROVED TRAINING WITH PROPER MASK LOADING")
print("=" * 60)

IMG_SIZE = 256
NUM_CLASSES = 8

CLASS_VALUES = {
    0: 0,    # Background - BW
    29: 1,   # Fish - HF
    76: 2,   # Plants - PF
    105: 3,  # Rocks - RI
    150: 4,  # Coral - RO
    200: 5,  # Wrecks - WR
    225: 6,  # Water - WB
    250: 7,  # Other - Other
}

def map_mask(mask):
    new_mask = np.zeros_like(mask)
    for val, cls in CLASS_VALUES.items():
        new_mask[mask == val] = cls
    unique = np.unique(new_mask)
    print(f"  Unique classes in mask: {unique}")
    return new_mask

def load_and_prepare_data(data_dir="SUIM-master/data/test"):
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
    
    images = np.array(images)
    masks = np.array(masks)
    
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Mask unique values: {np.unique(masks)}")
    
    return images, masks


def augment(images, masks):
    aug_img = []
    aug_mask = []
    
    for img, mask in zip(images, masks):
        aug_img.append(img)
        aug_mask.append(mask)
        
        aug_img.append(np.fliplr(img))
        aug_mask.append(np.fliplr(mask))
        
        aug_img.append(np.flipud(img))
        aug_mask.append(np.flipud(mask))
        
        for angle in [1, 2, 3]:
            aug_img.append(np.rot90(img, angle))
            aug_mask.append(np.rot90(mask, angle))
        
        for factor in [0.8, 1.2]:
            aug_img.append(np.clip(img * factor, 0, 1))
            aug_mask.append(mask)
    
    return np.array(aug_img), np.array(aug_mask)


def create_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
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
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n5. Training (50 epochs)...")
    history = model.fit(
        images, masks,
        epochs=50,
        batch_size=8,
        validation_split=0.15,
        verbose=2
    )
    
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/segmentation_final.keras")
    print("\n✅ Model saved to checkpoints/segmentation_final.keras")
    
    print("\n6. Evaluating...")
    preds = model.predict(images[:8], verbose=0)
    pred_masks = np.argmax(preds, axis=-1)
    
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
        axes[0, i].imshow(images[i])
        axes[0, i].set_title('Input')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(CLASS_COLORS[masks[i]])
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(CLASS_COLORS[pred_masks[i]])
        axes[2, i].set_title('Prediction')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/segmentation_output.png", dpi=150, bbox_inches='tight')
    print("✅ Saved: results/segmentation_output.png")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
