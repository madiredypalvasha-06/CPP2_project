# Underwater Semantic Segmentation using Deep Learning

## 📋 Project Overview

This is a comprehensive deep learning project for semantic segmentation of underwater images. The project is developed as part of Conceptual Project 2 for B.Tech in Artificial Intelligence and Machine Learning at Woxsen University.

## 🎯 Features

- **Multiple Models**: U-Net, Attention U-Net, and custom CNN architectures
- **8 Classes**: Background, Fish, Plants, Rocks, Coral, Wrecks, Water, Other
- **Web Application**: User-friendly Streamlit interface
- **Real-time Segmentation**: Upload images and get instant segmentation results

## 📁 Project Structure

```
CPP2/underwater_segmentation/
├── app_v2.py                 # Main web application
├── train_simple.py          # Training script
├── checkpoints/
│   └── segmentation_final.keras  # Trained model
├── results/
│   └── segmentation_output.png    # Sample results
├── SUIM-master/             # Dataset
├── RESEARCH_PAPER_ULTIMATE.docx   # Research Paper
└── PROJECT_REPORT_ULTIMATE.docx   # Project Report
```

## 🚀 How to Run the Website

### Method 1: Using Streamlit (Recommended)

1. **Navigate to the project directory:**
```bash
cd CPP2/underwater_segmentation
```

2. **Run the web application:**
```bash
streamlit run app_v2.py
```

3. **Open your browser:**
The application will automatically open in your default browser at `http://localhost:8501`

### Method 2: Using Python directly

```bash
cd CPP2/underwater_segmentation
python3.11 -m streamlit run app_v2.py
```

## 🎮 Using the Web Application

1. **Upload Image**: Click on "Choose an underwater image..." to upload a JPEG or PNG image
2. **View Results**: The application will display:
   - Original image
   - Segmented result with color-coded classes
   - Class distribution statistics
3. **Download**: You can download the segmentation mask

## 📊 Model Performance

- **Training Accuracy**: ~82%
- **Mean IoU**: ~0.35
- **Pixel Accuracy**: ~80%

## 🔧 Training the Model (Optional)

If you want to retrain the model:

```bash
cd CPP2/underwater_segmentation
python3.11 train_simple.py
```

This will:
1. Load the SUIM dataset
2. Apply data augmentation
3. Train the segmentation model
4. Save results to `checkpoints/segmentation_final.keras`

## 📄 Documents

- **Research Paper**: `RESEARCH_PAPER_ULTIMATE.docx`
- **Project Report**: `PROJECT_REPORT_ULTIMATE.docx`

## 📚 Dataset

This project uses the SUIM (Semantic Underwater Image Segmentation) dataset:
- Source: https://github.com/xahidbuffon/SUIM
- Paper: IEEE/RSJ IROS 2020
- Classes: 8 (Background, Fish, Plants, Rocks, Coral, Wrecks, Water, Other)

## 🛠️ Requirements

- Python 3.11+
- TensorFlow 2.x
- Streamlit
- OpenCV
- NumPy
- Matplotlib

Install requirements:
```bash
pip install tensorflow streamlit opencv-python numpy matplotlib
```

## 👨‍🎓 Student Details

- **Name**: Palvasha Madireddy
- **Course**: B.Tech, Artificial Intelligence and Machine Learning
- **University**: Woxsen University
- **Year**: 2026

## 📞 Support

For any issues or questions:
- GitHub Repository: https://github.com/madiredypalvasha-06/CPP2_project
- SUIM Dataset: https://github.com/xahidbuffon/SUIM

---

**Note**: This project is developed for educational purposes as part of the B.Tech curriculum.
