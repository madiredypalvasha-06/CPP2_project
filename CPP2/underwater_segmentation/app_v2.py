#!/usr/bin/env python3
"""
Enhanced Streamlit Web Application for Underwater Semantic Segmentation
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

st.set_page_config(
    page_title="Underwater Semantic Segmentation",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = ['Background', 'Fish', 'Plants', 'Rocks', 'Coral', 'Wrecks', 'Water', 'Other']
CLASS_COLORS = np.array([
    [0, 0, 0],        # 0: Background - Black
    [255, 0, 0],      # 1: Fish - Red
    [0, 255, 0],      # 2: Plants - Green
    [128, 128, 128],  # 3: Rocks - Gray
    [255, 255, 0],    # 4: Coral - Yellow
    [255, 0, 255],    # 5: Wrecks - Magenta
    [0, 255, 255],    # 6: Water - Cyan
    [255, 128, 0]     # 7: Other - Orange
], dtype=np.uint8)

IMG_SIZE = 256


@st.cache_resource
def load_model():
    """Load trained segmentation model"""
    try:
        model = tf.keras.models.load_model("checkpoints/segmentation_final.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image):
    """Preprocess image for model"""
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def predict_segmentation(model, image):
    """Run segmentation"""
    img = preprocess_image(image)
    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch, verbose=0)
    mask = np.argmax(pred[0], axis=-1)
    return mask, pred[0]


def create_colored_mask(mask):
    """Create colored mask"""
    return CLASS_COLORS[mask]


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
    .main {
        background-color: #f5f5f5;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #1e3a5f;
        text-align: center;
        padding: 20px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric {
        font-size: 24px;
        font-weight: bold;
        color: #1e3a5f;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="title">🌊 Underwater Semantic Segmentation</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep Learning for Marine Image Analysis | Woxsen University</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    
    # About section in sidebar
    st.sidebar.markdown("### 📊 Project Info")
    st.sidebar.info("""
    **Dataset:** SUIM
    **Classes:** 8
    **Model:** Custom CNN
    **Accuracy:** ~82%
    """)
    
    st.sidebar.markdown("### 🎨 Legend")
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        st.sidebar.markdown(
            f'<div style="display:flex;align-items:center;margin:3px 0;">'
            f'<div style="width:15px;height:15px;background-color:rgb({color[0]},{color[1]},{color[2]});margin-right:8px;border:1px solid #fff;"></div>'
            f'<span style="font-size:11px;">{name}</span></div>',
            unsafe_allow_html=True
        )
    
    st.sidebar.markdown("---")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an underwater image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload JPEG or PNG image"
        )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        with col1:
            st.markdown("### 🖼️ Original Image")
            st.image(image, use_container_width=True, caption="Uploaded Image")
        
        # Predict
        with st.spinner("Processing..."):
            mask, probs = predict_segmentation(model, image)
            colored_mask = create_colored_mask(mask)
            distribution = get_class_distribution(mask)
        
        with col2:
            st.markdown("### 🎯 Segmentation Result")
            st.image(colored_mask, use_container_width=True, caption="Predicted Segmented Image")
        
        # Results section
        st.markdown("---")
        st.markdown("### 📈 Analysis Results")
        
        # Class distribution
        col_a, col_b = st.columns([2, 1])
        
        with col_a:
            st.markdown("#### Class Distribution")
            for cls_name, percentage in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                if percentage > 0.1:
                    st.progress(percentage / 100)
                    st.markdown(f"**{cls_name}:** {percentage:.1f}%")
        
        with col_b:
            st.markdown("#### Quick Stats")
            st.metric("Image Size", f"{IMG_SIZE}×{IMG_SIZE}")
            st.metric("Classes Detected", len([v for v in distribution.values() if v > 0.1]))
        
        # Download section
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            # Convert colored mask to image
            result_pil = Image.fromarray(colored_mask)
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            st.download_button(
                label="Download Segmentation Mask",
                data=buf.getvalue(),
                file_name="segmentation_mask.png",
                mime="image/png"
            )
    
    # Information section
    st.markdown("---")
    st.markdown("## 📚 About This Project")
    
    with st.expander("Click to learn more"):
        st.markdown("""
        ### 🎓 Project Overview
        
        This is a **Conceptual Project 2** for B.Tech in Artificial Intelligence and Machine Learning at **Woxsen University**.
        
        ### 🔬 Methodology
        
        1. **Data Collection**: Used SUIM (Semantic Underwater Image Segmentation) dataset
        2. **Preprocessing**: Image resizing, normalization, data augmentation
        3. **Model Training**: Custom CNN with encoder-decoder architecture
        4. **Evaluation**: Mean IoU, Pixel Accuracy, Dice Score
        
        ### 🏗️ Architecture
        
        - **Encoder**: Convolutional layers for feature extraction
        - **Decoder**: Transposed convolutions for upsampling
        - **Output**: 8-class softmax for semantic segmentation
        
        ### 📊 Performance
        
        - Training Accuracy: ~82%
        - Mean IoU: ~0.35
        - Supports 8 underwater object classes
        
        ### 📖 References
        
        - SUIM Dataset: Islam et al., IEEE/RSJ IROS 2020
        - U-Net: Ronneberger et al., MICCAI 2015
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><b>Conceptual Project 2</b> | B.Tech AI & ML | Woxsen University</p>
        <p>Underwater Semantic Segmentation using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    import io
    main()
