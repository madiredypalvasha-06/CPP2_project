#!/usr/bin/env python3
"""
Fixed Streamlit Web Application for Underwater Semantic Segmentation
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

st.set_page_config(
    page_title="Underwater Semantic Segmentation",
    page_icon="🌊",
    layout="wide"
)

CLASS_NAMES = ['Background', 'Fish', 'Plants', 'Rocks', 'Coral', 'Wrecks', 'Water', 'Other']
CLASS_COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [128, 128, 128],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 128, 0]
], dtype=np.uint8)

IMG_SIZE = 256


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model("checkpoints/segmentation_final.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image):
    """Preprocess image"""
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def predict(model, image):
    """Predict segmentation"""
    img = preprocess_image(image)
    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch, verbose=0)
    mask = np.argmax(pred[0], axis=-1)
    return mask


def get_class_distribution(mask):
    """Get class distribution"""
    unique, counts = np.unique(mask, return_counts=True)
    total = counts.sum()
    distribution = {}
    for cls, cnt in zip(unique, counts):
        percentage = (cnt / total) * 100
        distribution[CLASS_NAMES[cls]] = percentage
    return distribution


def main():
    # Custom CSS
    st.markdown("""
    <style>
    .title {font-size:36px;font-weight:bold;color:#1e3a5f;text-align:center;padding:20px;}
    .subtitle {font-size:18px;color:#555;text-align:center;margin-bottom:30px;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="title">🌊 Underwater Semantic Segmentation</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep Learning for Marine Image Analysis | Woxsen University</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model not found! Please run train_simple.py first.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Project Info")
    st.sidebar.info("Dataset: SUIM\nClasses: 8\nAccuracy: ~82%")
    
    st.sidebar.markdown("### 🎨 Legend")
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        st.sidebar.markdown(
            f'<div style="display:flex;align-items:center;margin:3px 0;">'
            f'<div style="width:15px;height:15px;background-color:rgb({color[0]},{color[1]},{color[2]});margin-right:8px;"></div>'
            f'<span style="font-size:11px;">{name}</span></div>',
            unsafe_allow_html=True
        )
    
    # Upload
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader("Choose underwater image...", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🖼️ Original Image")
            st.image(image, use_container_width=True)
        
        with st.spinner("Processing..."):
            mask = predict(model, image)
            colored_mask = CLASS_COLORS[mask]
            distribution = get_class_distribution(mask)
        
        with col2:
            st.markdown("### 🎯 Segmentation Result")
            st.image(colored_mask, use_container_width=True, clamp=True)
        
        # Results
        st.markdown("---")
        st.markdown("### 📈 Analysis")
        
        col_a, col_b = st.columns([2, 1])
        
        with col_a:
            st.markdown("#### Class Distribution")
            for cls_name, percentage in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                if percentage > 0.1:
                    st.progress(percentage / 100)
                    st.markdown(f"**{cls_name}:** {percentage:.1f}%")
        
        with col_b:
            st.metric("Image Size", f"{IMG_SIZE}×{IMG_SIZE}")
            st.metric("Classes", len([v for v in distribution.values() if v > 0.1]))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#666;padding:20px;'>
        <p><b>Conceptual Project 2</b> | B.Tech AI & ML | Woxsen University</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
