import cv2
import numpy as np

def segment_characters(image_path_or_bytes, is_bytes=False):
    """
    Load an image, preprocess it, segment it into individual characters,
    and return the bounding boxes and the normalized 28x28 character images.
    """
    if is_bytes:
        nparr = np.frombuffer(image_path_or_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(image_path_or_bytes)
        
    if img is None:
        raise ValueError("Could not read image")
        
    # Keep a copy of original image for drawing bounding boxes
    annotated_img = img.copy()
    
    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    # 2. Binarize (Otsu's thresholding)
    # Check if the corners are white or dark to decide on inversion
    h, w = gray.shape
    corner_avg = (int(gray[0, 0]) + int(gray[0, w-1]) + int(gray[h-1, 0]) + int(gray[h-1, w-1])) / 4
    if corner_avg > 127:
        # White background (like paper/canvas clear) - invert so text is white on black
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Dark background (neon drawing on black canvas) - keep text white on black
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    # Perform a light dilation to connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
        
    # 3. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Extract bounding boxes
    boxes = []
    min_w, min_h = 3, 3 # Filter out noise
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw >= min_w and bh >= min_h:
            boxes.append([x, y, bw, bh])
            
    if not boxes:
        return annotated_img, [], []
        
    # 5. Merge overlapping/split bounding boxes (e.g. dot on 'i', double strokes, fraction bars)
    # A simple clustering: if two boxes overlap heavily in the X dimension, we check if they should be merged.
    merged_boxes = []
    # Sort boxes by x coordinate first
    boxes.sort(key=lambda b: b[0])
    
    while len(boxes) > 0:
        box = boxes.pop(0)
        x1, y1, w1, h1 = box
        
        merged = False
        # Look ahead at subsequent boxes to see if they overlap horizontally
        for i in range(len(boxes)):
            x2, y2, w2, h2 = boxes[i]
            
            # Check horizontal overlap
            # Two boxes overlap horizontally if:
            overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
            
            # If they overlap significantly or are very close (e.g. within 5 pixels)
            # and they are vertically stacked or close (to handle dots of i/j)
            horizontal_overlap_threshold = -4 # Allow small gap
            if overlap >= horizontal_overlap_threshold:
                # Merge boxes
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                
                # Update current box and remove merged one from list
                x1, y1, w1, h1 = new_x, new_y, new_w, new_h
                boxes.pop(i)
                merged = True
                break
                
        if merged:
            # Re-insert the merged box at the beginning to let it check for more merges
            boxes.insert(0, [x1, y1, w1, h1])
        else:
            merged_boxes.append([x1, y1, w1, h1])
            
    # Sort final boxes left-to-right
    merged_boxes.sort(key=lambda b: b[0])
    
    # 6. Crop, pad, and resize each character
    processed_chars = []
    for box in merged_boxes:
        bx, by, bw, bh = box
        
        # Crop character from thresholded image
        crop = thresh[by:by+bh, bx:bx+bw]
        
        # Center in square padding
        size = max(bw, bh)
        pad = int(size * 0.18) # standard padding to keep spacing
        if pad < 3:
            pad = 3
        square_size = size + 2 * pad
        
        char_square = np.zeros((square_size, square_size), dtype=np.uint8)
        
        # Paste cropped character into the center of the square
        dx = pad + (size - bw) // 2
        dy = pad + (size - bh) // 2
        char_square[dy:dy+bh, dx:dx+bw] = crop
        
        # Resize to 28x28 (standard input for MNIST/EMNIST)
        char_28 = cv2.resize(char_square, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to 0.0 - 1.0 (float32)
        char_norm = char_28.astype(np.float32) / 255.0
        
        processed_chars.append(char_norm)
        
        # Draw bounding boxes on annotated image (cyan color, 2px stroke)
        cv2.rectangle(annotated_img, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
        
    return annotated_img, merged_boxes, processed_chars
