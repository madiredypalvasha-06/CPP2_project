#!/usr/bin/env python3
"""
Underwater Semantic Segmentation - Streamlit Web Application
Run: streamlit run app.py
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
    [0, 0, 0],        # 0: Background - Black
    [255, 0, 0],     # 1: Fish - Red
    [0, 255, 0],     # 2: Plants - Green
    [128, 128, 128], # 3: Rocks - Gray
    [255, 255, 0],   # 4: Coral - Yellow
    [255, 0, 255],   # 5: Wrecks - Magenta
    [0, 255, 255],   # 6: Water - Cyan
    [255, 128, 0]    # 7: Other - Orange
], dtype=np.uint8)

IMG_SIZE = 256


@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    checkpoint_dir = "checkpoints"
    
    try:
        models['U-Net'] = tf.keras.models.load_model(f"{checkpoint_dir}/unet.keras", compile=False)
    except:
        st.warning("U-Net model not found. Please train the model first.")
    
    try:
        models['DeepLabV3+'] = tf.keras.models.load_model(f"{checkpoint_dir}/deeplabv3plus.keras", compile=False)
    except:
        pass
    
    try:
        models['Attention U-Net'] = tf.keras.models.load_model(f"{checkpoint_dir}/attention_unet.keras", compile=False)
    except:
        pass
    
    try:
        models['FPN'] = tf.keras.models.load_model(f"{checkpoint_dir}/fpn.keras", compile=False)
    except:
        pass
    
    return models


def preprocess_image(image):
    """Preprocess image for model input"""
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def predict_segmentation(models, image, use_tta=False):
    """Run segmentation prediction"""
    img = preprocess_image(image)
    img_batch = np.expand_dims(img, axis=0)
    
    predictions = {}
    
    for name, model in models.items():
        if use_tta:
            pred = tta_predict(model, img_batch)
        else:
            pred = model.predict(img_batch, verbose=0)
        
        mask = np.argmax(pred[0], axis=-1)
        predictions[name] = mask
    
    # Ensemble prediction
    if len(predictions) > 1:
        ensemble_pred = np.zeros((IMG_SIZE, IMG_SIZE, 8))
        for pred in predictions.values():
            ensemble_pred += model.predict(img_batch, verbose=0)
        ensemble_pred /= len(predictions)
        predictions['Ensemble'] = np.argmax(ensemble_pred[0], axis=-1)
    
    return predictions


def tta_predict(model, img):
    """Test Time Augmentation"""
    predictions = []
    
    predictions.append(model.predict(img, verbose=0))
    
    img_flip_h = tf.image.flip_left_right(img).numpy()
    p = model.predict(img_flip_h, verbose=0)
    predictions.append(tf.image.flip_left_right(tf.constant(p)).numpy())
    
    img_flip_v = tf.image.flip_up_down(img).numpy()
    p = model.predict(img_flip_v, verbose=0)
    predictions.append(tf.image.flip_up_down(tf.constant(p)).numpy())
    
    return np.mean(predictions, axis=0)


def create_colored_mask(mask):
    """Convert class mask to colored image"""
    return CLASS_COLORS[mask]


def main():
    st.title("🌊 Underwater Semantic Segmentation")
    st.markdown("""
    ## Automated Underwater Image Segmentation using Deep Learning
    
    This application uses state-of-the-art deep learning models for semantic segmentation 
    of underwater images. Upload an underwater image to see the segmentation results.
    """)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    model_options = ["U-Net", "Attention U-Net", "DeepLabV3+", "FPN", "Ensemble"]
    selected_models = st.sidebar.multiselect(
        "Select Models",
        model_options,
        default=["U-Net"]
    )
    
    use_tta = st.sidebar.checkbox("Use Test Time Augmentation (TTA)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Class Legend")
    
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        st.sidebar.markdown(
            f'<div style="display:flex;align-items:center;margin:5px 0;">'
            f'<div style="width:20px;height:20px;background-color:rgb({color[0]},{color[1]},{color[2]});margin-right:10px;border:1px solid #fff;"></div>'
            f'<span>{name}</span></div>',
            unsafe_allow_html=True
        )
    
    # Main content
    uploaded_file = st.file_uploader(
        "📤 Upload Underwater Image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an underwater image for segmentation"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Load models
        with st.spinner("Loading models..."):
            models = load_models()
        
        if not models:
            st.error("No models found! Please train the models first using train.py")
            return
        
        # Filter selected models
        models = {k: v for k, v in models.items() if k in selected_models}
        
        if st.button("🔍 Run Segmentation", type="primary"):
            with st.spinner("Running segmentation..."):
                predictions = predict_segmentation(models, image, use_tta)
                
                with col2:
                    st.subheader("🎯 Segmentation Result")
                    
                    if "Ensemble" in predictions:
                        result_img = create_colored_mask(predictions["Ensemble"])
                        st.image(result_img, use_container_width=True, caption="Ensemble Result")
                    elif predictions:
                        first_model = list(predictions.keys())[0]
                        result_img = create_colored_mask(predictions[first_model])
                        st.image(result_img, use_container_width=True, caption=first_model)
                
                # Show all model results
                st.markdown("### 📊 Results from All Models")
                
                # Create grid for results
                n_models = len(predictions)
                if n_models > 0:
                    cols = st.columns(min(n_models, 3))
                    
                    for idx, (name, mask) in enumerate(predictions.items()):
                        with cols[idx % 3]:
                            colored_mask = create_colored_mask(mask)
                            st.image(colored_mask, caption=name, use_container_width=True)
                
                # Statistics
                st.markdown("### 📈 Class Distribution")
                
                for name, mask in predictions.items():
                    unique, counts = np.unique(mask, return_counts=True)
                    total = counts.sum()
                    
                    st.markdown(f"**{name}**")
                    
                    cols = st.columns(4)
                    for i, (cls, cnt) in enumerate(zip(unique, counts)):
                        percentage = (cnt / total) * 100
                        with cols[i % 4]:
                            st.markdown(f"- {CLASS_NAMES[cls]}: {percentage:.1f}%")
    
    # Information section
    st.markdown("---")
    st.markdown("## ℹ️ About")
    
    st.markdown("""
    ### Models Used
    
    1. **U-Net**: Classic encoder-decoder architecture with skip connections
    2. **Attention U-Net**: U-Net with attention gates for better feature selection
    3. **DeepLabV3+**: Atrous Spatial Pyramid Pooling for multi-scale features
    4. **FPN**: Feature Pyramid Network for multi-scale detection
    5. **Ensemble**: Combination of all models for improved accuracy
    
    ### Dataset
    
    Trained on the SUIM (Semantic Underwater Image Segmentation) dataset with 8 classes:
    - Background, Fish, Plants, Rocks, Coral, Wrecks, Water, Other
    
    ### Performance Metrics
    
    - Mean IoU (Intersection over Union)
    - Dice Score
    - Pixel Accuracy
    
    ### How to Use
    
    1. Upload an underwater image using the file uploader
    2. Select which models to use from the sidebar
    3. Optionally enable Test Time Augmentation (TTA)
    4. Click "Run Segmentation" to see results
    5. View the segmentation mask and class distribution
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Developed for Conceptual Project 2 - B.Tech AI/ML</p>
        <p>Underwater Semantic Segmentation using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
