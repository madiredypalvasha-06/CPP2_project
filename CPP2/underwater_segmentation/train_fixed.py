#!/usr/bin/env python3
"""
Fixed Training Script - Simplified and Working Models
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 60)
print("UNDERWATER SEMANTIC SEGMENTATION - FIXED VERSION")
print("=" * 60)

IMG_SIZE = 256
NUM_CLASSES = 8
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 1e-4

CLASS_NAMES = ['Background', 'Fish', 'Plants', 'Rocks', 'Coral', 'Wrecks', 'Water', 'Other']

# SUIM color mapping
COLOR_MAP = {
    (0, 0, 0): 0,       # Background - Black
    (255, 0, 0): 1,     # Fish - Red
    (0, 255, 0): 2,    # Plants - Green  
    (128, 128, 128): 3, # Rocks - Gray
    (255, 255, 0): 4,   # Coral - Yellow
    (255, 0, 255): 5,   # Wrecks - Magenta
    (0, 255, 255): 6,   # Water - Cyan
    (255, 128, 0): 7    # Other - Orange
}

def load_suim_images(data_dir="SUIM-master/data/test"):
    """Load SUIM dataset"""
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    
    if not os.path.exists(images_dir):
        print(f"Warning: {images_dir} not found!")
        return None, None
    
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
        
        # Load corresponding mask
        base_name = os.path.splitext(img_file)[0]
        mask_path = os.path.join(masks_dir, f"{base_name}.bmp")
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
        else:
            # Try to find in subdirectories
            found = False
            for class_name in ['FV', 'HD', 'RI', 'RO', 'WR', 'PF', 'SR']:
                class_mask_path = os.path.join(masks_dir, class_name, f"{base_name}.bmp")
                if os.path.exists(class_mask_path):
                    mask = cv2.imread(class_mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                    masks.append(mask)
                    found = True
                    break
            if not found:
                masks.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8))
    
    return np.array(images), np.array(masks)


def create_simple_unet():
    """Simple working U-Net"""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Encoder
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(p3)
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(c4)
    c4 = layers.BatchNormalization()(c4)
    
    # Decoder
    u5 = layers.Conv2DTranspose(256, 2, strides=2)(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, 3, padding='same', activation='relu')(u5)
    c5 = layers.Conv2D(256, 3, padding='same', activation='relu')(c5)
    
    u6 = layers.Conv2DTranspose(128, 2, strides=2)(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, 3, padding='same', activation='relu')(u6)
    c6 = layers.Conv2D(128, 3, padding='same', activation='relu')(c6)
    
    u7 = layers.Conv2DTranspose(64, 2, strides=2)(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, 3, padding='same', activation='relu')(u7)
    c7 = layers.Conv2D(64, 3, padding='same', activation='relu')(c7)
    
    outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(c7)
    
    model = keras.Model(inputs, outputs)
    return model


def create_attention_unet():
    """Attention U-Net"""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Encoder
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(p3)
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(c4)
    c4 = layers.BatchNormalization()(c4)
    
    # Attention gates
    def attention_gate(g, x, channels):
        theta = layers.Conv2D(channels, 1)(x)
        phi = layers.Conv2D(channels, 1)(g)
        add = layers.Add()([theta, phi])
        act = layers.Activation('relu')(add)
        psi = layers.Conv2D(1, 1, activation='sigmoid')(act)
        return layers.Multiply()([x, psi])
    
    # Decoder with attention
    u5 = layers.Conv2DTranspose(256, 2, strides=2)(c4)
    att5 = attention_gate(u5, c3, 256)
    u5 = layers.concatenate([u5, att5])
    c5 = layers.Conv2D(256, 3, padding='same', activation='relu')(u5)
    c5 = layers.Conv2D(256, 3, padding='same', activation='relu')(c5)
    
    u6 = layers.Conv2DTranspose(128, 2, strides=2)(c5)
    att6 = attention_gate(u6, c2, 128)
    u6 = layers.concatenate([u6, att6])
    c6 = layers.Conv2D(128, 3, padding='same', activation='relu')(u6)
    c6 = layers.Conv2D(128, 3, padding='same', activation='relu')(c6)
    
    u7 = layers.Conv2DTranspose(64, 2, strides=2)(c6)
    att7 = attention_gate(u7, c1, 64)
    u7 = layers.concatenate([u7, att7])
    c7 = layers.Conv2D(64, 3, padding='same', activation='relu')(u7)
    c7 = layers.Conv2D(64, 3, padding='same', activation='relu')(c7)
    
    outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(c7)
    
    model = keras.Model(inputs, outputs)
    return model


def create_simple_fpn():
    """Simple FPN-like architecture"""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Bottom-up pathway
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(p3)
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(c4)
    
    # Top-down pathway
    t1 = layers.Conv2DTranspose(256, 2, strides=2)(c4)
    t1 = layers.Add()([t1, c3])
    
    t2 = layers.Conv2DTranspose(128, 2, strides=2)(t1)
    t2 = layers.Add()([t2, c2])
    
    t3 = layers.Conv2DTranspose(64, 2, strides=2)(t2)
    t3 = layers.Add()([t3, c1])
    
    # Output
    out = layers.Conv2D(64, 3, padding='same', activation='relu')(t3)
    outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(out)
    
    model = keras.Model(inputs, outputs)
    return model


def create_linknet():
    """Simple LinkNet - works well for segmentation"""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Initial
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Encoder
    x1 = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D(2)(x1)
    
    x2 = layers.Conv2D(128, 3, padding='same', activation='relu')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D(2)(x2)
    
    x3 = layers.Conv2D(256, 3, padding='same', activation='relu')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.MaxPooling2D(2)(x3)
    
    x4 = layers.Conv2D(512, 3, padding='same', activation='relu')(x3)
    x4 = layers.BatchNormalization()(x4)
    
    # Decoder with residuals
    d1 = layers.Conv2DTranspose(256, 2, strides=2)(x4)
    d1 = layers.Add()([d1, x3])
    d1 = layers.Conv2D(256, 3, padding='same', activation='relu')(d1)
    
    d2 = layers.Conv2DTranspose(128, 2, strides=2)(d1)
    d2 = layers.Add()([d2, x2])
    d2 = layers.Conv2D(128, 3, padding='same', activation='relu')(d2)
    
    d3 = layers.Conv2DTranspose(64, 2, strides=2)(d2)
    d3 = layers.Add()([d3, x1])
    d3 = layers.Conv2D(64, 3, padding='same', activation='relu')(d3)
    
    outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(d3)
    
    model = keras.Model(inputs, outputs)
    return model


def augment_data(images, masks):
    """Augment training data"""
    aug_images = []
    aug_masks = []
    
    for img, mask in zip(images, masks):
        # Original
        aug_images.append(img)
        aug_masks.append(mask)
        
        # Horizontal flip
        aug_images.append(np.fliplr(img).copy())
        aug_masks.append(np.fliplr(mask).copy())
        
        # Vertical flip
        aug_images.append(np.flipud(img).copy())
        aug_masks.append(np.flipud(mask).copy())
        
        # Rotate 90
        aug_images.append(np.rot90(img).copy())
        aug_masks.append(np.rot90(mask).copy())
    
    return np.array(aug_images), np.array(aug_masks)


def calculate_iou(y_true, y_pred, num_classes=8):
    """Calculate Mean IoU"""
    ious = []
    for cls in range(num_classes):
        true_cls = y_true == cls
        pred_cls = y_pred == cls
        
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    return np.nanmean(ious)


def train_model(model, X_train, Y_train, model_name, epochs=EPOCHS):
    """Train a model"""
    print(f"\nTraining {model_name}...")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=0.15
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_train[-len(X_train)//5:], Y_train[-len(Y_train)//5:], verbose=0)
    print(f"{model_name} - Val Accuracy: {val_acc:.4f}")
    
    return history, model


def main():
    # Load data
    print("\nLoading dataset...")
    images, masks = load_suim_images()
    
    if images is None or len(images) == 0:
        print("Error: Could not load images!")
        print("Please ensure SUIM dataset is properly downloaded.")
        return
    
    print(f"Loaded {len(images)} images")
    print(f"Mask shape: {masks.shape}, unique values: {np.unique(masks)}")
    
    # Augment
    print("\nAugmenting data...")
    images, masks = augment_data(images, masks)
    print(f"After augmentation: {len(images)} samples")
    
    # Train models
    models_info = [
        ("U-Net", create_simple_unet),
        ("Attention U-Net", create_attention_unet),
        ("FPN", create_simple_fpn),
        ("LinkNet", create_linknet),
    ]
    
    results = {}
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for name, model_fn in models_info:
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print('='*50)
        
        try:
            model = model_fn()
            model.summary()
            
            history, trained_model = train_model(model, images, masks, name)
            
            # Save
            model_path = os.path.join(checkpoint_dir, f"{name.lower().replace(' ', '_')}_fixed.keras")
            trained_model.save(model_path)
            print(f"Saved: {model_path}")
            
            # Quick evaluation
            preds = trained_model.predict(images[:5], verbose=0)
            pred_masks = np.argmax(preds, axis=-1)
            miou = calculate_iou(masks[:5], pred_masks)
            print(f"{name} Quick mIoU: {miou:.4f}")
            
            results[name] = miou
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for name, miou in results.items():
        print(f"{name}: mIoU = {miou:.4f}")


if __name__ == "__main__":
    main()
