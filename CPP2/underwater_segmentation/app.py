#!/usr/bin/env python3
"""
Flask Web Application for Underwater Semantic Segmentation
Run: python app.py
"""

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template_string, request, send_from_directory
from PIL import Image
import io
import os
import webbrowser
import threading
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CLASS_NAMES = ['Background', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5']

CLASS_COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [135, 206, 235],
    [128, 128, 128],
    [255, 192, 203],
    [255, 255, 0],
    [255, 255, 255]
], dtype=np.uint8)

IMG_SIZE = 256

model = None

def load_model():
    global model
    try:
        model = tf.keras.models.load_model("checkpoints/unet_enhanced.keras", compile=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def preprocess_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def predict(image):
    global model
    if model is None:
        return None
    img = preprocess_image(image)
    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch, verbose=0)
    mask = np.argmax(pred[0], axis=-1)
    return mask

def get_class_distribution(mask):
    unique, counts = np.unique(mask, return_counts=True)
    total = counts.sum()
    distribution = {}
    for cls, cnt in zip(unique, counts):
        percentage = (cnt / total) * 100
        distribution[CLASS_NAMES[cls]] = percentage
    return distribution

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Underwater Semantic Segmentation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header { 
            text-align: center; 
            padding: 40px 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        h1 { font-size: 2.5em; margin-bottom: 10px; color: #4fc3f7; }
        .subtitle { color: #b0bec5; font-size: 1.1em; }
        .upload-section {
            background: rgba(255,255,255,0.1);
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-btn {
            background: linear-gradient(135deg, #4fc3f7, #2196f3);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 30px;
            font-size: 1.2em;
            cursor: pointer;
            transition: transform 0.3s;
        }
        .upload-btn:hover { transform: scale(1.05); }
        input[type="file"] { display: none; }
        .results {
            display: none;
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 20px;
            margin-top: 30px;
        }
        .results.show { display: block; }
        .image-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 20px;
        }
        .image-box {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 15px;
            text-align: center;
        }
        .image-box h3 { color: #4fc3f7; margin-bottom: 15px; }
        .image-box img { 
            max-width: 100%; 
            border-radius: 10px;
            width: 300px;
            height: 300px;
            object-fit: cover;
        }
        .analysis {
            margin-top: 30px;
            padding: 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
        }
        .analysis h3 { color: #4fc3f7; margin-bottom: 20px; }
        .class-item {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }
        .class-name { width: 150px; }
        .progress-bar {
            flex: 1;
            height: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4fc3f7, #2196f3);
            transition: width 0.5s;
        }
        .class-percent { width: 80px; text-align: right; }
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 20px;
            justify-content: center;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 5px;
        }
        .info-box {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            text-align: center;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #b0bec5;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Underwater Semantic Segmentation</h1>
            <p class="subtitle">Deep Learning for Marine Image Analysis</p>
        </header>

        <div class="upload-section">
            <label class="upload-btn">
                Upload Underwater Image
                <input type="file" id="fileInput" accept="image/*">
            </label>
            <p style="margin-top: 15px; color: #b0bec5;">Supported formats: JPG, PNG, BMP</p>
        </div>

        <div class="results" id="results">
            <div class="image-grid">
                <div class="image-box">
                    <h3>Original Image</h3>
                    <img id="originalImg" src="" alt="Original">
                </div>
                <div class="image-box">
                    <h3>Segmentation Result</h3>
                    <img id="resultImg" src="" alt="Segmentation" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGR5PSIuM2VtIiBmaWxsPSIjZmZmIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+SW1hZ2UgZXJyb3I8L3RleHQ+PC9zdmc+'">
                </div>
            </div>

            <div class="analysis">
                <h3>Class Distribution</h3>
                <div id="classDist"></div>
            </div>

            <div class="info-box">
                <p><b>Dataset:</b> SUIM | <b>Classes:</b> 8 | <b>Model Accuracy:</b> ~82%</p>
            </div>

            <div class="legend">
                {% for name, color in legend %}
                <div class="legend-item">
                    <div class="legend-color" style="background: rgb({{color[0]}},{{color[1]}},{{color[2]}})"></div>
                    <span>{{name}}</span>
                </div>
                {% endfor %}
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const results = document.getElementById('results');
        
        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('image', file);

            results.classList.add('show');
            document.getElementById('originalImg').src = URL.createObjectURL(file);
            document.getElementById('resultImg').src = '/static/loading.gif';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                console.log('Response:', data);

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                document.getElementById('resultImg').src = 'data:image/png;base64,' + data.image;
                console.log('Image set, length:', data.image ? data.image.length : 0);
                
                let distHtml = '';
                for (const [cls, pct] of Object.entries(data.distribution)) {
                    if (pct > 0.1) {
                        distHtml += `
                            <div class="class-item">
                                <span class="class-name">${cls}</span>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${pct}%"></div>
                                </div>
                                <span class="class-percent">${pct.toFixed(1)}%</span>
                            </div>
                        `;
                    }
                }
                document.getElementById('classDist').innerHTML = distHtml;
            } catch (err) {
                alert('Error processing image: ' + err);
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    legend = list(zip(CLASS_NAMES, CLASS_COLORS.tolist()))
    return render_template_string(HTML_TEMPLATE, legend=legend)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return {'error': 'No image provided'}
    
    file = request.files['image']
    try:
        image = Image.open(file.stream)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        mask = predict(image)
        
        if mask is None:
            return {'error': 'Model not loaded'}
        
        mask = np.clip(mask, 0, len(CLASS_COLORS) - 1)
        
        colored_mask = CLASS_COLORS[mask]
        colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode('.png', colored_mask)
        if not success:
            return {'error': 'Failed to encode image'}
        import base64
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        distribution = get_class_distribution(mask)
        
        return {'image': img_b64, 'distribution': distribution}
    except Exception as e:
        return {'error': f'Error: {str(e)}'}

@app.route('/static/loading.gif')
def loading():
    return send_from_directory('.', 'static/loading.gif' if os.path.exists('static/loading.gif') else __file__)

def open_browser():
    time.sleep(2)
    webbrowser.open('http://127.0.0.1:5001')

if __name__ == '__main__':
    print("=" * 50)
    print("Underwater Semantic Segmentation")
    print("=" * 50)
    print("\nLoading model...")
    load_model()
    
    if model:
        print("\nStarting server...")
        print("Opening browser at http://127.0.0.1:5001")
        print("\nPress Ctrl+C to stop the server\n")
        
        threading.Thread(target=open_browser, daemon=True).start()
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        print("\nCannot start server - Model not found!")
        print("Please run train_improved.py first to train the model.")
