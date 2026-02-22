#!/usr/bin/env python3
"""
Underwater Semantic Segmentation - Enhanced Version
U-Net + Attention U-Net + DeepLabV3+ + FPN with TTA and Ensemble
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
                                     Concatenate, BatchNormalization, Activation, Dropout,
                                     Add, Multiply, GlobalAveragePooling2D, Reshape, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K


# ============ CONFIG ============
class Config:
    IMG_DIR = "/Users/palvashamadireddy/Downloads/SUIM-master/data/test/images"
    MSK_DIR = "/Users/palvashamadireddy/Downloads/SUIM-master/data/test/masks"
    OUTPUT_DIR = "results"
    CHECKPOINT_DIR = "checkpoints"
    
    IMG_SIZE = 384
    NUM_CLASSES = 8
    EPOCHS = 40
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.15
    
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


# ============ LOSS FUNCTIONS ============
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        return -K.mean(alpha_t * K.pow(1 - pt, gamma) * K.log(pt))
    return loss


def combined_loss(y_true, y_pred):
    cce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return cce


# ============ DATA ============
def rgb_to_class(mask):
    out = np.zeros(mask.shape[:2], dtype=np.uint8)
    for rgb, cls in config.RGB_MAP.items():
        out[np.all(mask == rgb, axis=-1)] = cls
    return out


def compute_class_weights(Y):
    class_counts = np.bincount(Y.flatten(), minlength=config.NUM_CLASSES)
    total = np.sum(class_counts)
    weights = total / (config.NUM_CLASSES * class_counts + 1e-6)
    return weights / weights.sum() * config.NUM_CLASSES


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
        
        X.append(img.copy())
        Y.append(msk.copy())
        
        for _ in range(8):
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
            
            if np.random.rand() > 0.5:
                hsv = cv2.cvtColor((img_aug * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                hsv = hsv.astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-20, 20)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.6, 1.4), 0, 255)
                hsv = hsv.astype(np.uint8)
                img_aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            
            if np.random.rand() > 0.7:
                noise = np.random.normal(0, 0.02, img_aug.shape)
                img_aug = np.clip(img_aug + noise, 0, 1)
            
            X.append(img_aug)
            Y.append(msk_aug)
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)
    
    print(f"Total: {len(X)} samples")
    print(f"Class distribution: {np.bincount(Y.flatten(), minlength=config.NUM_CLASSES)}")
    print("="*60 + "\n")
    return X, Y


# ============ ATTENTION GATE ============
def attention_gate(g, x, filters):
    theta_x = Conv2D(filters, 3, strides=2, padding='same')(x)
    phi_g = Conv2D(filters, 1, padding='same')(g)
    
    add = Add()([theta_x, phi_g])
    act = Activation('relu')(add)
    
    psi = Conv2D(1, 1, padding='same')(act)
    sigmoid = Activation('sigmoid')(psi)
    
    upsample = UpSampling2D(2, interpolation='bilinear')(sigmoid)
    x_upsampled = UpSampling2D(2, interpolation='bilinear')(x)
    multiply = Multiply()([upsample, x_upsampled])
    return multiply


# ============ MODELS ============
def conv_block(x, filters, use_bn=True):
    x = Conv2D(filters, 3, padding='same')(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same')(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def attention_unet():
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
    
    # Standard U-Net decoder (skip connections at each level)
    u1 = UpSampling2D()(c5)
    u1 = Concatenate()([u1, c4])
    u1 = conv_block(u1, 512)
    
    u2 = UpSampling2D()(u1)
    u2 = Concatenate()([u2, c3])
    u2 = conv_block(u2, 256)
    
    u3 = UpSampling2D()(u2)
    u3 = Concatenate()([u3, c2])
    u3 = conv_block(u3, 128)
    
    u4 = UpSampling2D()(u3)
    u4 = Concatenate()([u4, c1])
    u4 = conv_block(u4, 64)
    
    o = Conv2D(config.NUM_CLASSES, 1, activation='softmax')(u4)
    return Model(i, o)


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


def deeplabv3_plus():
    i = Input((config.IMG_SIZE, config.IMG_SIZE, 3))
    
    # Encoder (shallower to preserve spatial info)
    x = Conv2D(64, 3, padding='same', activation='relu')(i)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    c1 = x  # 384
    
    x = MaxPooling2D()(x)
    
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    c2 = x  # 192
    
    x = MaxPooling2D()(x)
    
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    c3 = x  # 96
    
    x = MaxPooling2D()(x)
    
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)  # 48
    
    # ASPP with smaller dilation rates (feature map is 48x48)
    a1 = Conv2D(256, 1, activation='relu', padding='same')(x)
    a1 = BatchNormalization()(a1)
    
    a2 = Conv2D(256, 3, dilation_rate=2, activation='relu', padding='same')(x)
    a2 = BatchNormalization()(a2)
    
    a3 = Conv2D(256, 3, dilation_rate=4, activation='relu', padding='same')(x)
    a3 = BatchNormalization()(a3)
    
    gap = GlobalAveragePooling2D()(x)
    gap = Reshape((1, 1, 512))(gap)
    gap = Conv2D(256, 1, activation='relu', padding='same')(gap)
    gap = BatchNormalization()(gap)
    gap = Lambda(lambda t: tf.image.resize(t, [48, 48], method='bilinear'))(gap)
    
    x = Concatenate()([a1, a2, a3, gap])
    x = Conv2D(256, 1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Decoder with skip connections
    x = UpSampling2D(2)(x)  # 48 -> 96
    x = Concatenate()([x, c3])
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D(2)(x)  # 96 -> 192
    x = Concatenate()([x, c2])
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D(2)(x)  # 192 -> 384
    x = Concatenate()([x, c1])
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    o = Conv2D(config.NUM_CLASSES, 1, activation='softmax')(x)
    return Model(i, o)


def fpn():
    i = Input((config.IMG_SIZE, config.IMG_SIZE, 3))
    
    # Backbone (simpler)
    c1 = conv_block(i, 64)
    p1 = MaxPooling2D()(c1)
    
    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D()(c2)
    
    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D()(c3)
    
    c4 = conv_block(p3, 512)
    
    # Top-down pathway
    p4 = Conv2D(256, 1, padding='same')(c4)
    p3 = Conv2D(256, 1, padding='same')(c3)
    p2 = Conv2D(256, 1, padding='same')(c2)
    
    # Lateral connections
    m4 = p4
    m3 = Add()([UpSampling2D()(m4), p3])
    m2 = Add()([UpSampling2D()(m3), p2])
    
    # Final feature maps
    o4 = Conv2D(256, 3, padding='same')(m4)
    o3 = Conv2D(256, 3, padding='same')(m3)
    o2 = Conv2D(256, 3, padding='same')(m2)
    
    # Combine feature maps
    o = Concatenate()([o2, UpSampling2D(2)(o3), UpSampling2D(4)(o4)])
    o = Conv2D(256, 3, padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    # Upsample to original size
    o = UpSampling2D(4)(o)
    
    o = Conv2D(config.NUM_CLASSES, 1, activation='softmax')(o)
    return Model(i, o)


# ============ ENHANCED TTA ============
def tta_predict(model, img):
    predictions = []
    
    predictions.append(model.predict(img, verbose=0))
    
    img_flip_h = tf.image.flip_left_right(img).numpy()
    p = model.predict(img_flip_h, verbose=0)
    predictions.append(tf.image.flip_left_right(tf.constant(p)).numpy())
    
    img_flip_v = tf.image.flip_up_down(img).numpy()
    p = model.predict(img_flip_v, verbose=0)
    predictions.append(tf.image.flip_up_down(tf.constant(p)).numpy())
    
    img_flip_hv = tf.image.flip_left_right(tf.image.flip_up_down(img)).numpy()
    p = model.predict(img_flip_hv, verbose=0)
    p = tf.image.flip_up_down(tf.constant(p))
    p = tf.image.flip_left_right(p).numpy()
    predictions.append(p)
    
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
        'per_class_iou': iou_per_class,
        'per_class_dice': dice_per_class
    }


# ============ VISUALIZATION ============
def create_benchmark_output(X, Y, predictions_dict):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    n = min(6, len(X))
    models_to_show = ['Attention U-Net', 'U-Net', 'DeepLabV3+', 'FPN', 'Ensemble']
    n_cols = 2 + len(models_to_show[:4])
    
    fig, axes = plt.subplots(n, n_cols, figsize=(4*n_cols, 4*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['#E91E63', '#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    
    for idx in range(n):
        axes[idx, 0].imshow((X[idx] * 255).astype(np.uint8))
        axes[idx, 0].set_title('Input', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(config.CLASS_COLORS[Y[idx]])
        axes[idx, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        
        for m_idx, name in enumerate(models_to_show[:n_cols-2]):
            if name in predictions_dict:
                axes[idx, m_idx + 2].imshow(config.CLASS_COLORS[predictions_dict[name][idx]])
                axes[idx, m_idx + 2].set_title(name, fontsize=12, fontweight='bold', color=colors[m_idx])
                axes[idx, m_idx + 2].axis('off')
    
    plt.suptitle('Underwater Semantic Segmentation - Enhanced Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{config.OUTPUT_DIR}/segmentation_results.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {config.OUTPUT_DIR}/segmentation_results.png")


def plot_training(histories):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#E91E63', '#2196F3', '#FF5722', '#4CAF50']
    
    ax = axes[0]
    for i, (name, h) in enumerate(histories.items()):
        ax.plot(h.history['loss'], color=colors[i % len(colors)], linewidth=2, label=name)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for i, (name, h) in enumerate(histories.items()):
        if 'val_loss' in h.history:
            ax.plot(h.history['val_loss'], color=colors[i % len(colors)], linewidth=2, linestyle='--', label=f'{name} (Val)')
    ax.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
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
    colors = ['#E91E63', '#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#00BCD4']
    
    ax = axes[0, 0]
    ious = [metrics_dict[n]['mean_iou'] for n in names]
    ax.bar(range(len(names)), ious, color=colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_title('Mean IoU', fontweight='bold')
    ax.set_ylabel('IoU')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(ious):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    ax = axes[0, 1]
    dice = [metrics_dict[n]['mean_dice'] for n in names]
    ax.bar(range(len(names)), dice, color=colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_title('Mean Dice Score', fontweight='bold')
    ax.set_ylabel('Dice')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(dice):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    ax = axes[1, 0]
    pix = [metrics_dict[n]['pixel_accuracy'] for n in names]
    ax.bar(range(len(names)), pix, color=colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_title('Pixel Accuracy', fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3, axis='y')
    
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
    print("UNDERWATER SEMANTIC SEGMENTATION - ENHANCED")
    print("Attention U-Net + U-Net + DeepLabV3+ + FPN + Ensemble")
    print("="*70)
    print(f"Image Size: {config.IMG_SIZE} | Epochs: {config.EPOCHS}")
    print(f"Batch: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE} | Val Split: {config.VAL_SPLIT}")
    print("="*70 + "\n")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    X, Y = load_data()
    
    indices = np.random.permutation(len(X))
    val_size = int(len(X) * config.VAL_SPLIT)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    
    print(f"Training: {len(X_train)} | Validation: {len(X_val)}")
    
    histories = {}
    models = {}
    model_paths = {}
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]
    
    # Train Attention U-Net
    print("="*60)
    print("TRAINING ATTENTION U-NET")
    print("="*60)
    
    model_attn = attention_unet()
    model_attn.compile(optimizer=Adam(config.LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    h1 = model_attn.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=1, callbacks=callbacks)
    model_attn.save(f"{config.CHECKPOINT_DIR}/attention_unet.keras")
    histories['Attention U-Net'] = h1
    models['Attention U-Net'] = model_attn
    model_paths['Attention U-Net'] = f"{config.CHECKPOINT_DIR}/attention_unet.keras"
    print("Attention U-Net saved!\n")
    
    # Train U-Net
    print("="*60)
    print("TRAINING U-NET")
    print("="*60)
    
    model_unet = unet()
    model_unet.compile(optimizer=Adam(config.LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    h2 = model_unet.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=1, callbacks=callbacks)
    model_unet.save(f"{config.CHECKPOINT_DIR}/unet.keras")
    histories['U-Net'] = h2
    models['U-Net'] = model_unet
    model_paths['U-Net'] = f"{config.CHECKPOINT_DIR}/unet.keras"
    print("U-Net saved!\n")
    
    # Train DeepLabV3+
    print("="*60)
    print("TRAINING DEEPLABV3+")
    print("="*60)
    
    model_deeplab = deeplabv3_plus()
    model_deeplab.compile(optimizer=Adam(config.LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    h3 = model_deeplab.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=1, callbacks=callbacks)
    model_deeplab.save(f"{config.CHECKPOINT_DIR}/deeplabv3plus.keras")
    histories['DeepLabV3+'] = h3
    models['DeepLabV3+'] = model_deeplab
    model_paths['DeepLabV3+'] = f"{config.CHECKPOINT_DIR}/deeplabv3plus.keras"
    print("DeepLabV3+ saved!\n")
    
    # Train FPN
    print("="*60)
    print("TRAINING FPN")
    print("="*60)
    
    model_fpn = fpn()
    model_fpn.compile(optimizer=Adam(config.LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    h4 = model_fpn.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=1, callbacks=callbacks)
    model_fpn.save(f"{config.CHECKPOINT_DIR}/fpn.keras")
    histories['FPN'] = h4
    models['FPN'] = model_fpn
    model_paths['FPN'] = f"{config.CHECKPOINT_DIR}/fpn.keras"
    print("FPN saved!\n")
    
    # Generate predictions
    print("="*60)
    print("GENERATING RESULTS")
    print("="*60)
    
    predictions = {}
    
    p_attn = model_attn.predict(X, verbose=0)
    p_unet = model_unet.predict(X, verbose=0)
    p_deeplab = model_deeplab.predict(X, verbose=0)
    p_fpn = model_fpn.predict(X, verbose=0)
    
    predictions['Attention U-Net'] = np.argmax(p_attn, axis=-1)
    predictions['U-Net'] = np.argmax(p_unet, axis=-1)
    predictions['DeepLabV3+'] = np.argmax(p_deeplab, axis=-1)
    predictions['FPN'] = np.argmax(p_fpn, axis=-1)
    predictions['Ensemble'] = np.argmax((p_attn + p_unet + p_deeplab + p_fpn) / 4, axis=-1)
    
    print("Applying TTA...")
    for name, model in models.items():
        pred_tta = []
        for i in range(len(X)):
            p = tta_predict(model, X[i:i+1])
            pred_tta.append(np.argmax(p[0], axis=-1))
        predictions[f'{name} TTA'] = np.array(pred_tta)
    
    create_benchmark_output(X, Y, predictions)
    plot_training(histories)
    create_legend()
    
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
    
    best_model = max(metrics_dict.keys(), key=lambda k: metrics_dict[k]['mean_iou'])
    print(f"\n*** Best Model: {best_model} (IoU: {metrics_dict[best_model]['mean_iou']:.4f}) ***")
    
    with open(f"{config.OUTPUT_DIR}/metrics_report.txt", 'w') as f:
        f.write("="*60 + "\nMETRICS REPORT - ENHANCED MODELS\n" + "="*60 + "\n\n")
        f.write(f"Best Model: {best_model}\n\n")
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
