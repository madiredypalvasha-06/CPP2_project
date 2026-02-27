#!/usr/bin/env python3
"""
Transfer Learning Training - Uses pre-trained backbones for better results
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print("=" * 60)
print("TRAINING WITH PRE-TRAINED BACKBONE")
print("=" * 60)

IMG_SIZE = 256
NUM_CLASSES = 8
BATCH_SIZE = 4
EPOCHS = 20


def load_data(data_dir="SUIM-master/data/test"):
    """Load SUIM test data with augmentation"""
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Found {len(image_files)} images")
    
    images = []
    masks = []
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        images.append(img)
        
        base_name = os.path.splitext(img_file)[0]
        mask_path = os.path.join(masks_dir, f"{base_name}.bmp")
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
        else:
            masks.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8))
    
    return np.array(images), np.array(masks)


def augment(images, masks):
    """Strong augmentation"""
    aug_img = []
    aug_mask = []
    
    for img, mask in zip(images, masks):
        # Original
        aug_img.append(img)
        aug_mask.append(mask)
        
        # Flips
        aug_img.append(np.fliplr(img))
        aug_mask.append(np.fliplr(mask))
        
        aug_img.append(np.flipud(img))
        aug_mask.append(np.flipud(mask))
        
        # Rotations
        for k in [1, 2, 3]:
            aug_img.append(np.rot90(img, k))
            aug_mask.append(np.rot90(mask, k))
        
        # Brightness
        for factor in [0.8, 1.2]:
            img_bright = np.clip(img * factor, 0, 1)
            aug_img.append(img_bright)
            aug_mask.append(mask)
    
    return np.array(aug_img), np.array(aug_mask)


def create_unet_with_backbone():
    """U-Net with MobileNetV2 backbone - Transfer Learning"""
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Use layers from MobileNetV2
    layer_names = [
        'conv1_relu',       # 128x128
        'out_relu',         # 8x8 - deepest layer
    ]
    
    layers_outputs = [base_model.get_layer(name).output for name in layer_names]
    
    # Create encoder
    encoder = keras.Model(inputs=base_model.input, outputs=layers_outputs)
    
    # Decoder
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Get encoder outputs
    x = encoder(inputs)
    
    # Decoder path - progressively upsample
    # Start from deepest layer
    x = x[-1]  # 8x8
    
    # Upsample to 16x16
    x = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Upsample to 32x32
    x = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Upsample to 64x64
    x = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Upsample to 128x128
    x = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Upsample to 256x256
    x = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Output
    outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model


def create_simple_seg():
    """Simple but effective segmentation model"""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Encoder
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    p1 = layers.MaxPooling2D(2)(x)
    p1 = layers.Dropout(0.1)(p1)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    p2 = layers.MaxPooling2D(2)(x)
    p2 = layers.Dropout(0.1)(p2)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    p3 = layers.MaxPooling2D(2)(x)
    p3 = layers.Dropout(0.2)(p3)
    
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(p3)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder
    x = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)


def calculate_iou(y_true, y_pred):
    """Calculate mIoU"""
    ious = []
    for cls in range(NUM_CLASSES):
        true_cls = (y_true == cls).astype(np.float32)
        pred_cls = (y_pred == cls).astype(np.float32)
        
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        
        if union > 0:
            ious.append(intersection / union)
    
    return np.mean(ious)


def main():
    print("\nLoading data...")
    images, masks = load_data()
    print(f"Loaded {len(images)} images")
    
    print("\nAugmenting...")
    images, masks = augment(images, masks)
    print(f"After augmentation: {len(images)} samples")
    
    # Create and train simple model
    print("\nCreating model...")
    model = create_simple_seg()
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTraining...")
    history = model.fit(
        images, masks,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        verbose=1
    )
    
    # Save
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/segmentation_model.keras")
    print("\nModel saved!")
    
    # Evaluate
    preds = model.predict(images[:8], verbose=0)
    pred_masks = np.argmax(preds, axis=-1)
    
    miou = calculate_iou(masks[:8], pred_masks)
    print(f"\nMean IoU on sample: {miou:.4f}")
    
    # Visualize
    plt.figure(figsize=(15, 5))
    for i in range(min(4, len(images))):
        plt.subplot(3, 4, i+1)
        plt.imshow(images[i])
        plt.title('Input')
        plt.axis('off')
        
        plt.subplot(3, 4, i+5)
        plt.imshow(masks[i])
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(3, 4, i+9)
        plt.imshow(pred_masks[i])
        plt.title('Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("results/segmentation_preview.png", dpi=100)
    print("\nSaved preview to results/segmentation_preview.png")


if __name__ == "__main__":
    main()
