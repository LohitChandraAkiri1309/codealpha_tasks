import os
import base64
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
import cv2

from model import CharCNN
from segment import segment_characters

app = Flask(__name__)

# Define label mappings
MNIST_LABELS = [str(i) for i in range(10)]
EMNIST_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Global model variables
mnist_model = None
emnist_model = None
mnist_loaded = False
emnist_loaded = False

def init_models():
    global mnist_model, emnist_model, mnist_loaded, emnist_loaded
    
    # 1. Initialize MNIST Model (10 classes: 0-9)
    mnist_model = CharCNN(num_classes=10)
    mnist_path = "models/mnist_cnn.pth"
    if os.path.exists(mnist_path):
        try:
            mnist_model.load_state_dict(torch.load(mnist_path, map_location=torch.device('cpu')))
            mnist_model.eval()
            mnist_loaded = True
            print("Successfully loaded MNIST model weights.")
        except Exception as e:
            print(f"Error loading MNIST model: {e}")
    else:
        print("Warning: MNIST weights file not found. Running with randomized weights. Run python train.py.")
        mnist_model.eval()

    # 2. Initialize EMNIST Model (26 classes: A-Z)
    emnist_model = CharCNN(num_classes=26)
    emnist_path = "models/emnist_cnn.pth"
    if os.path.exists(emnist_path):
        try:
            emnist_model.load_state_dict(torch.load(emnist_path, map_location=torch.device('cpu')))
            emnist_model.eval()
            emnist_loaded = True
            print("Successfully loaded EMNIST model weights.")
        except Exception as e:
            print(f"Error loading EMNIST model: {e}")
    else:
        print("Warning: EMNIST weights file not found. Running with randomized weights. Run python train.py.")
        emnist_model.eval()

def load_model_weights_if_needed():
    global mnist_loaded, emnist_loaded
    mnist_path = "models/mnist_cnn.pth"
    if not mnist_loaded and os.path.exists(mnist_path):
        try:
            mnist_model.load_state_dict(torch.load(mnist_path, map_location=torch.device('cpu')))
            mnist_model.eval()
            mnist_loaded = True
            print("Successfully hot-loaded MNIST model weights.")
        except Exception as e:
            print(f"Error hot-loading MNIST model: {e}")
            
    emnist_path = "models/emnist_cnn.pth"
    if not emnist_loaded and os.path.exists(emnist_path):
        try:
            emnist_model.load_state_dict(torch.load(emnist_path, map_location=torch.device('cpu')))
            emnist_model.eval()
            emnist_loaded = True
            print("Successfully hot-loaded EMNIST model weights.")
        except Exception as e:
            print(f"Error hot-loading EMNIST model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    load_model_weights_if_needed()
    return jsonify({
        'mnist': {
            'loaded': mnist_loaded,
            'classes': MNIST_LABELS
        },
        'emnist': {
            'loaded': emnist_loaded,
            'classes': EMNIST_LABELS
        }
    })

def decode_base64_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    return img_data

@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Predict a single drawn or uploaded character.
    Uses segment_characters to isolate the character, center, and resize it.
    """
    load_model_weights_if_needed()
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        model_type = data.get('model_type', 'mnist') # 'mnist' or 'emnist'
        img_bytes = decode_base64_image(data['image'])
        
        # Segment characters to auto-crop, auto-center and pad correctly
        annotated_img, boxes, chars = segment_characters(img_bytes, is_bytes=True)
        
        if len(chars) == 0:
            return jsonify({
                'prediction': 'N/A',
                'confidence': 0.0,
                'distribution': [],
                'message': 'No character detected in the drawing canvas.'
            })
            
        # Run prediction on the first detected character
        char_img = chars[0]
        
        # Match training normalization: (x - 0.1307) / 0.3081
        char_tensor = torch.tensor(char_img).unsqueeze(0).unsqueeze(0) # [1, 1, 28, 28]
        char_tensor = (char_tensor - 0.1307) / 0.3081
        
        # Select model and labels
        if model_type == 'emnist':
            model = emnist_model
            labels = EMNIST_LABELS
        else:
            model = mnist_model
            labels = MNIST_LABELS
            
        with torch.no_grad():
            outputs = model(char_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze().numpy()
            
        # Get highest class
        pred_idx = int(np.argmax(probabilities))
        prediction = labels[pred_idx]
        confidence = float(probabilities[pred_idx])
        
        # Prepare probability distribution for charts
        dist = [{'label': labels[i], 'prob': float(probabilities[i])} for i in range(len(labels))]
        dist.sort(key=lambda x: x['prob'], reverse=True)
        
        # Return base64 preprocessed 28x28 image for the frontend preview
        char_img_255 = (char_img * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', char_img_255)
        preprocessed_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'distribution': dist[:5], # Send top 5 confidences
            'preprocessed_preview': preprocessed_b64,
            'model_loaded': emnist_loaded if model_type == 'emnist' else mnist_loaded
        })
        
    except Exception as e:
        print(f"Error in single prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_sequence', methods=['POST'])
def predict_sequence():
    """
    Segment a word or sequence of characters from the drawing canvas/upload,
    predict each character left-to-right, and return the combined text.
    """
    load_model_weights_if_needed()
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        model_type = data.get('model_type', 'emnist') # Usually EMNIST for words
        img_bytes = decode_base64_image(data['image'])
        
        # Run segmentation
        annotated_img, boxes, chars = segment_characters(img_bytes, is_bytes=True)
        
        if len(chars) == 0:
            return jsonify({
                'sequence': '',
                'predictions': [],
                'message': 'No characters detected.'
            })
            
        # Select model and labels
        if model_type == 'emnist':
            model = emnist_model
            labels = EMNIST_LABELS
        else:
            model = mnist_model
            labels = MNIST_LABELS
            
        predictions = []
        sequence_chars = []
        
        for idx, char_img in enumerate(chars):
            # Preprocess to match training loader
            char_tensor = torch.tensor(char_img).unsqueeze(0).unsqueeze(0) # [1, 1, 28, 28]
            char_tensor = (char_tensor - 0.1307) / 0.3081
            
            with torch.no_grad():
                outputs = model(char_tensor)
                probabilities = F.softmax(outputs, dim=1).squeeze().numpy()
                
            pred_idx = int(np.argmax(probabilities))
            pred_char = labels[pred_idx]
            conf = float(probabilities[pred_idx])
            
            sequence_chars.append(pred_char)
            
            # Save preprocessed image preview as base64
            char_img_255 = (char_img * 255).astype(np.uint8)
            _, buf = cv2.imencode('.png', char_img_255)
            char_b64 = base64.b64encode(buf).decode('utf-8')
            
            predictions.append({
                'index': idx,
                'char': pred_char,
                'confidence': conf,
                'bbox': boxes[idx], # [x, y, w, h]
                'preview': char_b64
            })
            
        # Convert annotated image (with boxes) to base64
        _, annot_buf = cv2.imencode('.png', annotated_img)
        annotated_b64 = base64.b64encode(annot_buf).decode('utf-8')
        
        sequence_str = "".join(sequence_chars)
        
        return jsonify({
            'sequence': sequence_str,
            'predictions': predictions,
            'annotated_image': annotated_b64,
            'model_loaded': emnist_loaded if model_type == 'emnist' else mnist_loaded
        })
        
    except Exception as e:
        print(f"Error in sequence prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
