import sys
import os
import numpy as np
import torch
import cv2

# Import project modules
from model import CharCNN
from segment import segment_characters

def test_model_forward():
    print("Testing CharCNN architecture...")
    model_mnist = CharCNN(num_classes=10)
    model_emnist = CharCNN(num_classes=26)
    
    # Generate mock batch of 4 grayscale images
    dummy_input = torch.randn(4, 1, 28, 28)
    
    out_mnist = model_mnist(dummy_input)
    out_emnist = model_emnist(dummy_input)
    
    assert out_mnist.shape == (4, 10), f"Expected shape (4, 10), got {out_mnist.shape}"
    assert out_emnist.shape == (4, 26), f"Expected shape (4, 26), got {out_emnist.shape}"
    print("OK: Model forward pass tests passed.")

def test_segmentation():
    print("\nTesting character segmentation pipeline with mock image...")
    # Create a blank black image
    mock_img = np.zeros((100, 300, 3), dtype=np.uint8)
    
    # Draw two white letters/shapes (e.g. rectangles) to simulate characters
    # Rectangle 1: representing '1'
    cv2.rectangle(mock_img, (30, 20), (45, 80), (255, 255, 255), -1)
    
    # Rectangle 2: representing '2'
    cv2.rectangle(mock_img, (100, 20), (135, 80), (255, 255, 255), -1)
    
    # Double stroke: let's draw two close shapes that should be merged (like i's dot and stem)
    cv2.rectangle(mock_img, (200, 20), (210, 30), (255, 255, 255), -1) # dot
    cv2.rectangle(mock_img, (200, 40), (210, 80), (255, 255, 255), -1) # stem
    
    # Encode mock image to bytes
    _, buf = cv2.imencode(".png", mock_img)
    img_bytes = buf.tobytes()
    
    # Run segmentation
    annotated_img, boxes, chars = segment_characters(img_bytes, is_bytes=True)
    
    print(f"Detected {len(boxes)} segmented components (Expected: 3, after merging dot and stem)")
    for i, box in enumerate(boxes):
        print(f"  Box {i}: {box}")
        
    # We expect exactly 3 components (first rect, second rect, and merged third rect + dot)
    assert len(chars) == 3, f"Expected 3 segmented characters, got {len(chars)}"
    
    # Check that character dimensions are all 28x28 normalized
    for idx, char_img in enumerate(chars):
        assert char_img.shape == (28, 28), f"Expected (28, 28) shape for character, got {char_img.shape}"
        assert char_img.min() >= 0.0 and char_img.max() <= 1.0, "Pixels must be normalized between 0 and 1"
        
    print("OK: Segmentation logic, vertical box merging, and padding normalization tests passed.")

if __name__ == "__main__":
    try:
        test_model_forward()
        test_segmentation()
        print("\nAll pipeline verification tests passed successfully!")
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)
