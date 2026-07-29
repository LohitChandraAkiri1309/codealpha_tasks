# Neural Characters - Handwritten OCR Dashboard

Neural Characters is a Deep Learning-based handwritten digit and character recognition platform. Using PyTorch convolutional neural networks (CNNs), it allows users to draw character sequences on an interactive canvas, upload custom images, and segment + recognize each individual character (digits 0-9 and letters A-Z).

---

## Features

- **Interactive Drawing Canvas**: Draw free-form sequences of characters directly in the web UI.
- **Advanced Character Segmentation**: Pre-processes user inputs, detects letter/number boundaries, clusters overlaps (e.g., dots on 'i's or double strokes), and extracts individual characters.
- **Deep Learning Recognition**: Uses PyTorch CNN models (`CharCNN` model architecture) to perform high-accuracy OCR predictions on MNIST (digits) and EMNIST Split Letters (A-Z) datasets.
- **Real-Time Confidence Feedback**: View classified characters, their bounding boxes, and the model's confidence scores in real-time.
- **Training Script**: Includes a standalone training utility `train.py` to train/fine-tune custom model checkpoints.

---

## How to Setup and Run

### 1. Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 2. Open the Project Folder
Open your terminal or command prompt and navigate to the project directory:
```bash
cd "Neural Characters - subtitled as the Handwritten OCR Dashboard or Deep Learning Handwritten Recognition Dashboard"
```

### 3. Create a Virtual Environment (Recommended)
It is highly recommended to isolate your dependencies using a Python virtual environment:
- **Windows**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies
Install all the required Python libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 5. Train the Machine Learning Models
To generate the model weights, run the training script:
```bash
python train.py
```
This script will:
- Automatically download the standard **MNIST** and **EMNIST split-letters** datasets into a `./data` subdirectory.
- Train the CNN classifiers for 2 epochs on a random subset of 20,000 samples.
- Save the trained weights to `models/mnist_cnn.pth` and `models/emnist_cnn.pth`.

*(Note: If you run the web app without training first, it will run using randomized weights with poor classification accuracy.)*

### 6. Run the Application
Start the Flask development server:
```bash
python app.py
```

### 7. View the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

> [!WARNING]
> **Port Conflict:** By default, this application runs on port `5000`. If you plan to run **AuraCredit AI** concurrently, you will need to change the port number in `app.py` for one of the projects to avoid a port collision.

---

## Technical Details

- **Backend**: Python Flask server.
- **Deep Learning Framework**: PyTorch (`torch` and `torchvision`).
- **Computer Vision**: OpenCV (`cv2`) for image binarization, Otsu's thresholding, contour extraction, bounding box detection, and character dilation.
- **Model Architecture**: `CharCNN` consisting of two Convolutional-ReLU-Pooling blocks, Dropout layers, and two Dense (Fully Connected) layers.
