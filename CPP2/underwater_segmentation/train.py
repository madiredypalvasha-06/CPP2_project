#!/usr/bin/env python3
"""
Underwater Semantic Segmentation - Professional Version
U-Net + DeepLabV3 + Hybrid with TTA
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, 
                                     Concatenate, BatchNormalization, Activation, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau


# ============ CONFIG ============
class Config:
    IMG_DIR = "/Users/palvashamadireddy/Downloads/SUIM-master/data/test/images"
    MSK_DIR = "/Users/palvashamadireddy/Downloads/SUIM-master/data/test/masks"
    OUTPUT_DIR = "results"
    CHECKPOINT_DIR = "checkpoints"
    
    IMG_SIZE = 256
    NUM_CLASSES = 8
    EPOCHS = 25
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    
    CLASS_NAMES = ['Background', 'Fish', 'Plants', 'Rocks', 'Coral', 'Wrecks', 'Water', 'Other']
    CLASS_COLORS = np.array([
        [0, 0, 0],        # 0: Background - Black
        [255, 0, 0],     # 1: Fish - Red
        [0, 255, 0],     # 2: Plants - Green
        [128, 128, 128], # 3: Rocks - Gray
        [255, 255, 0],   # 4: Coral - Yellow
        [255, 0, 255],   # 5: Wrecks - Magenta
        [0, 255, 255],   # 6: Water - Cyan
        [255, 128, 0]    # 7: Other - Orange
    ], dtype=np.uint8)
    
    RGB_MAP = {
        (0, 0, 0): 0, (255, 0, 0): 1, (0, 255, 0): 2, (128, 128, 128): 3,
        (255, 255, 0): 4, (255, 0, 255): 5, (0, 255, 255): 6, (255, 128, 0): 7,
        (0, 0, 255): 3, (255, 255, 255): 0, (194, 178, 128): 3
    }

config = Config()


# ============ DATA ============
def rgb_to_class(mask):
    out = np.zeros(mask.shape[:2], dtype=np.uint8)
    for rgb, cls in config.RGB_MAP.items():
        out[np.all(mask == rgb, axis=-1)] = cls
    return out


def load_data():
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    X, Y = [], []
    files = sorted(os.listdir(config.IMG_DIR))
    print(f"Found {len(files)} images")
    
    for f in files:
        if not f.endswith(".jpg"): continue
        
        img_path = os.path.join(config.IMG_DIR, f)
        msk_path = os.path.join(config.MSK_DIR, f.replace(".jpg", ".bmp"))
        if not os.path.exists(msk_path):
            msk_path = msk_path.replace(".bmp", ".png")
            if not os.path.exists(msk_path): continue
        
        img = cv2.imread(img_path)
        msk = cv2.imread(msk_path)
        
        if img is None or msk is None: continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
        
        img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        msk = cv2.resize(msk, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        msk = rgb_to_class(msk)
        img = img.astype(np.float32) / 255.0
        
        # Original
        X.append(img.copy())
        Y.append(msk.copy())
        
        # Strong augmentation (12x)
        for _ in range(12):
            img_aug = img.copy()
            msk_aug = msk.copy()
            
            if np.random.rand() > 0.5:
                img_aug = np.fliplr(img_aug).copy()
                msk_aug = np.fliplr(msk_aug).copy()
            
            if np.random.rand() > 0.5:
                img_aug = np.flipud(img_aug).copy()
                msk_aug = np.flipud(msk_aug).copy()
            
            if np.random.rand() > 0.5:
                k = np.random.choice([1, 2, 3])
                img_aug = np.rot90(img_aug, k).copy()
                msk_aug = np.rot90(msk_aug, k).copy()
            
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.7, 1.3)
                img_aug = np.clip(img_aug * factor, 0, 1)
            
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.7, 1.3)
                mean = img_aug.mean()
                img_aug = np.clip((img_aug - mean) * factor + mean, 0, 1)
            
            if np.random.rand() > 0.6:
                hsv = cv2.cvtColor((img_aug * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                hsv = hsv.astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.6, 1.4), 0, 255)
                hsv = hsv.astype(np.uint8)
                img_aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            
            X.append(img_aug)
            Y.append(msk_aug)
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)
    
    print(f"Total: {len(X)} samples")
    print("="*60 + "\n")
    return X, Y


# ============ MODELS ============
def conv_block(x, filters):
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def unet():
    i = Input((config.IMG_SIZE, config.IMG_SIZE, 3))
    
    c1 = conv_block(i, 64)
    p1 = MaxPooling2D()(c1)
    
    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D()(c2)
    
    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D()(c3)
    
    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D()(c4)
    
    c5 = conv_block(p4, 1024)
    
    u1 = UpSampling2D()(c5)
    u1 = Concatenate()([u1, c4])
    c6 = conv_block(u1, 512)
    
    u2 = UpSampling2D()(c6)
    u2 = Concatenate()([u2, c3])
    c7 = conv_block(u2, 256)
    
    u3 = UpSampling2D()(c7)
    u3 = Concatenate()([u3, c2])
    c8 = conv_block(u3, 128)
    
    u4 = UpSampling2D()(c8)
    u4 = Concatenate()([u4, c1])
    c9 = conv_block(u4, 64)
    
    o = Conv2D(config.NUM_CLASSES, 1, activation='softmax')(c9)
    return Model(i, o)


def deeplab():
    i = Input((config.IMG_SIZE, config.IMG_SIZE, 3))
    
    # Simple encoder
    x = Conv2D(64, 3, padding='same', activation='relu')(i)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    p1 = MaxPooling2D()(x)  # 128
    
    x = Conv2D(128, 3, padding='same', activation='relu')(p1)
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    p2 = MaxPooling2D()(x)  # 64
    
    x = Conv2D(256, 3, padding='same', activation='relu')(p2)
    x = BatchNormalization()(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)  # 64
    
    # ASPP-like module with dilated convolutions
    a1 = Conv2D(256, 1, activation='relu', padding='same')(x)
    a2 = Conv2D(256, 3, dilation_rate=4, activation='relu', padding='same')(x)
    a3 = Conv2D(256, 3, dilation_rate=8, activation='relu', padding='same')(x)
    
    x = Concatenate()([a1, a2, a3])
    x = Conv2D(256, 1, activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)
    
    # Upsample back to 256x256
    x = UpSampling2D(4)(x)  # 64 -> 256
    
    o = Conv2D(config.NUM_CLASSES, 1, activation='softmax')(x)
    return Model(i, o)


# ============ TTA ============
def tta_predict(model, img):
    predictions = []
    
    # Original
    predictions.append(model.predict(img, verbose=0))
    
    # Horizontal flip
    img_flip_h = tf.image.flip_left_right(img).numpy()
    p = model.predict(img_flip_h, verbose=0)
    predictions.append(tf.image.flip_left_right(tf.constant(p)).numpy())
    
    # Vertical flip
    img_flip_v = tf.image.flip_up_down(img).numpy()
    p = model.predict(img_flip_v, verbose=0)
    predictions.append(tf.image.flip_up_down(tf.constant(p)).numpy())
    
    return np.mean(predictions, axis=0)


# ============ METRICS ============
def calculate_metrics(y_true, y_pred):
    iou_per_class = []
    for i in range(config.NUM_CLASSES):
        t = (y_true == i).astype(np.float32)
        p = (y_pred == i).astype(np.float32)
        inter = np.sum(t * p)
        union = np.sum(t) + np.sum(p) - inter
        iou_per_class.append(inter / union if union > 0 else 0.0)
    
    dice_per_class = []
    for i in range(config.NUM_CLASSES):
        t = (y_true == i).astype(np.float32)
        p = (y_pred == i).astype(np.float32)
        inter = np.sum(t * p)
        total = np.sum(t) + np.sum(p)
        dice_per_class.append(2 * inter / total if total > 0 else 0.0)
    
    return {
        'mean_iou': np.mean(iou_per_class),
        'mean_dice': np.mean(dice_per_class),
        'pixel_accuracy': np.mean((y_true == y_pred).astype(np.float32)),
        'per_class_iou': iou_per_class
    }


# ============ VISUALIZATION ============
def create_benchmark_output(X, Y, predictions_dict):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    n = min(4, len(X))
    models_to_show = ['U-Net', 'DeepLabV3', 'Hybrid']  # Only show these 3
    n_cols = 2 + len(models_to_show)
    
    fig, axes = plt.subplots(n, n_cols, figsize=(5*n_cols, 5*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['#2196F3', '#FF5722', '#4CAF50']
    
    for idx in range(n):
        axes[idx, 0].imshow((X[idx] * 255).astype(np.uint8))
        axes[idx, 0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(config.CLASS_COLORS[Y[idx]])
        axes[idx, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[idx, 1].axis('off')
        
        for m_idx, name in enumerate(models_to_show):
            if name in predictions_dict:
                axes[idx, m_idx + 2].imshow(config.CLASS_COLORS[predictions_dict[name][idx]])
                axes[idx, m_idx + 2].set_title(name, fontsize=14, fontweight='bold', color=colors[m_idx])
                axes[idx, m_idx + 2].axis('off')
    
    plt.suptitle('Underwater Image Semantic Segmentation\n(SUIM Dataset Benchmark)', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{config.OUTPUT_DIR}/segmentation_results.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {config.OUTPUT_DIR}/segmentation_results.png")


def plot_training(histories):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#2196F3', '#FF5722']
    
    ax = axes[0]
    for i, (name, h) in enumerate(histories.items()):
        ax.plot(h.history['loss'], color=colors[i], linewidth=2, label=name)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for i, (name, h) in enumerate(histories.items()):
        ax.plot(h.history['accuracy'], color=colors[i], linewidth=2, label=name)
    ax.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{config.OUTPUT_DIR}/training_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {config.OUTPUT_DIR}/training_curves.png")


def create_legend():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    
    for i, (name, color) in enumerate(zip(config.CLASS_NAMES, config.CLASS_COLORS)):
        rect = np.zeros((30, 60, 3), dtype=np.uint8)
        rect[:] = color
        ax.imshow(rect, extent=[0, 1, 8-i-1, 8-i])
        ax.text(1.1, 8-i-0.5, name, va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 8)
    ax.set_title('Class Color Legend', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{config.OUTPUT_DIR}/legend.png', dpi=150)
    plt.close()
    print(f"Saved: {config.OUTPUT_DIR}/legend.png")


def plot_metrics_comparison(metrics_dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(metrics_dict.keys())
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    
    # Mean IoU
    ax = axes[0, 0]
    ious = [metrics_dict[n]['mean_iou'] for n in names]
    ax.bar(range(len(names)), ious, color=colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15)
    ax.set_title('Mean IoU', fontweight='bold')
    ax.set_ylabel('IoU')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Dice
    ax = axes[0, 1]
    dice = [metrics_dict[n]['mean_dice'] for n in names]
    ax.bar(range(len(names)), dice, color=colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15)
    ax.set_title('Mean Dice Score', fontweight='bold')
    ax.set_ylabel('Dice')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Pixel Accuracy
    ax = axes[1, 0]
    pix = [metrics_dict[n]['pixel_accuracy'] for n in names]
    ax.bar(range(len(names)), pix, color=colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15)
    ax.set_title('Pixel Accuracy', fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Per-class IoU for best model
    ax = axes[1, 1]
    best_model = max(metrics_dict.keys(), key=lambda k: metrics_dict[k]['mean_iou'])
    per_class = metrics_dict[best_model]['per_class_iou']
    ax.bar(range(config.NUM_CLASSES), per_class, color='#4CAF50')
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=30, ha='right')
    ax.set_title(f'Per-Class IoU ({best_model})', fontweight='bold')
    ax.set_ylabel('IoU')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{config.OUTPUT_DIR}/metrics_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {config.OUTPUT_DIR}/metrics_comparison.png")


# ============ MAIN ============
def main():
    print("\n" + "="*70)
    print("UNDERWATER SEMANTIC SEGMENTATION")
    print("U-Net + DeepLabV3 + Hybrid with TTA")
    print("="*70)
    print(f"Epochs: {config.EPOCHS} | Batch: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE}")
    print("="*70 + "\n")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    X, Y = load_data()
    
    histories = {}
    models = {}
    
    # Train U-Net
    print("="*60)
    print("TRAINING U-NET")
    print("="*60)
    
    model_unet = unet()
    model_unet.compile(optimizer=Adam(config.LEARNING_RATE),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    
    h1 = model_unet.fit(X, Y, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                        verbose=1, callbacks=[lr_scheduler])
    model_unet.save(f"{config.CHECKPOINT_DIR}/unet.keras")
    histories['U-Net'] = h1
    models['U-Net'] = model_unet
    print("U-Net saved!\n")
    
    # Train DeepLabV3
    print("="*60)
    print("TRAINING DEEP LAB V3")
    print("="*60)
    
    model_deeplab = deeplab()
    model_deeplab.compile(optimizer=Adam(config.LEARNING_RATE),
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    h2 = model_deeplab.fit(X, Y, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                           verbose=1, callbacks=[lr_scheduler])
    model_deeplab.save(f"{config.CHECKPOINT_DIR}/deeplab.keras")
    histories['DeepLabV3'] = h2
    models['DeepLabV3'] = model_deeplab
    print("DeepLabV3 saved!\n")
    
    # Generate predictions
    print("="*60)
    print("GENERATING RESULTS")
    print("="*60)
    
    predictions = {}
    
    # Regular predictions
    p_unet = model_unet.predict(X, verbose=0)
    p_deeplab = model_deeplab.predict(X, verbose=0)
    
    predictions['U-Net'] = np.argmax(p_unet, axis=-1)
    predictions['DeepLabV3'] = np.argmax(p_deeplab, axis=-1)
    predictions['Hybrid'] = np.argmax((p_unet + p_deeplab)/2, axis=-1)
    
    # With TTA
    print("Applying TTA...")
    pred_unet_tta = []
    for i in range(len(X)):
        p = tta_predict(model_unet, X[i:i+1])
        pred_unet_tta.append(np.argmax(p[0], axis=-1))
    predictions['U-Net TTA'] = np.array(pred_unet_tta)
    
    pred_deeplab_tta = []
    for i in range(len(X)):
        p = tta_predict(model_deeplab, X[i:i+1])
        pred_deeplab_tta.append(np.argmax(p[0], axis=-1))
    predictions['DeepLabV3 TTA'] = np.array(pred_deeplab_tta)
    
    pred_hybrid_tta = []
    for i in range(len(X)):
        p1 = tta_predict(model_unet, X[i:i+1])
        p2 = tta_predict(model_deeplab, X[i:i+1])
        p_avg = (p1 + p2) / 2
        pred_hybrid_tta.append(np.argmax(p_avg[0], axis=-1))
    predictions['Hybrid TTA'] = np.array(pred_hybrid_tta)
    
    # Visualizations
    create_benchmark_output(X, Y, predictions)
    plot_training(histories)
    create_legend()
    
    # Metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    metrics_dict = {}
    for name, pred in predictions.items():
        m = calculate_metrics(Y, pred)
        metrics_dict[name] = m
        
        print(f"\n{name}:")
        print(f"  Mean IoU:       {m['mean_iou']:.4f}")
        print(f"  Dice Score:     {m['mean_dice']:.4f}")
        print(f"  Pixel Accuracy: {m['pixel_accuracy']:.4f}")
    
    plot_metrics_comparison(metrics_dict)
    
    # Save report
    with open(f"{config.OUTPUT_DIR}/metrics_report.txt", 'w') as f:
        f.write("="*60 + "\nMETRICS REPORT\n="*60 + "\n\n")
        for name, m in metrics_dict.items():
            f.write(f"{name}:\n")
            f.write(f"  Mean IoU: {m['mean_iou']:.4f}\n")
            f.write(f"  Dice: {m['mean_dice']:.4f}\n")
            f.write(f"  Pixel Acc: {m['pixel_accuracy']:.4f}\n\n")
    
    print(f"\nSaved: {config.OUTPUT_DIR}/metrics_report.txt")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nOutput files:")
    print("  - segmentation_results.png")
    print("  - training_curves.png")
    print("  - metrics_comparison.png")
    print("  - legend.png")
    print("  - metrics_report.txt")
    print("="*60)


if __name__ == "__main__":
    main()
