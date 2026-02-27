#!/usr/bin/env python3
"""
Train with full SUIM dataset (1525 images)
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 60)
print("TRAINING WITH FULL SUIM DATASET (1525 IMAGES)")
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

def load_data():
    images_dir = os.path.join(DATA_PATH, "train_val", "images")
    masks_dir = os.path.join(DATA_PATH, "train_val", "masks")
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Found {len(image_files)} images")
    
    images = []
    masks = []
    
    for i, img_file in enumerate(image_files):
        if i % 100 == 0:
            print(f"Loading {i}/{len(image_files)}...")
            
        img = cv2.imread(os.path.join(images_dir, img_file))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        base = os.path.splitext(img_file)[0]
        for ext in ['.jpg', '.png', '.bmp']:
            mask_path = os.path.join(masks_dir, base + ext)
            if os.path.exists(mask_path):
                break
        
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
    print(f"Unique mask values: {np.unique(masks)}")
    
    return images, masks

def create_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
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
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

def main():
    print("\n1. Loading full dataset...")
    images, masks = load_data()
    
    print("\n2. Creating model...")
    model = create_model()
    model.summary()
    
    print("\n3. Compiling...")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n4. Training (30 epochs)...")
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    history = model.fit(
        images, masks,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2
    )
    
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/segmentation_final.keras")
    print("\nModel saved to checkpoints/segmentation_final.keras")
    
    print("\n5. Evaluating...")
    preds = model.predict(images[:16], verbose=0)
    pred_masks = np.argmax(preds, axis=-1)
    
    ious = []
    for i in range(min(16, len(images))):
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
    print("Saved: results/segmentation_output.png")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
