# Surgical Tool Detection PoC - Code Generation Prompt

Create a Python application for a surgical tool detection Proof of Concept using YOLO11x. This application should help detect missing surgical tools by comparing before and after operation images.

## Requirements:

### Core Functionality:
1. **Image Input System**: Allow users to upload two images - "Before Operation" and "After Operation"
2. **Object Detection**: Use YOLO11x to detect common objects (as proxy for surgical tools) in both images
3. **Object Counting**: Count detected objects in each image
4. **Comparison Logic**: Compare detected objects between before/after images and identify missing items
5. **Visual Output**: Display both images side by side with detection boxes and show difference analysis

### Technical Specifications:

#### Libraries to Use:
- `ultralytics` (for YOLO11x)
- `opencv-python` (cv2)
- `PIL` or `Pillow` (for image handling)
- `streamlit` (for web interface)
- `numpy`
- `matplotlib` (for visualization)

#### Detection Categories:
Focus on detecting these common objects (as surgical tool proxies):
- scissors
- knife
- spoon (as forceps proxy)
- bottle (as container proxy)
- cup
- bowl
- fork
- Any small handheld objects

#### Application Structure:
1. **Main Interface**: Streamlit web app with file upload widgets
2. **Detection Module**: YOLO11x detection function
3. **Comparison Module**: Logic to compare detected objects
4. **Visualization Module**: Display results with bounding boxes

### Specific Features to Implement:

#### 1. Image Upload Section:
- Two file upload widgets: "Before Operation Image" and "After Operation Image"
- Support for JPG, PNG, JPEG formats
- Image preview functionality

#### 2. Detection Processing:
- Use YOLO11x model (yolo11x.pt)
- Set confidence threshold to 0.3 (adjustable)
- Extract object classes and coordinates
- Store detection results in structured format

#### 3. Object Tracking Logic:
- Count objects by class in both images
- Create a comparison table showing:
  - Object type
  - Count in "Before" image
  - Count in "After" image
  - Difference (missing items)
- Highlight missing objects in red

#### 4. Visual Display:
- Show both images side by side
- Draw bounding boxes on detected objects
- Use different colors: Green for matched items, Red for missing items
- Display summary statistics

#### 5. Results Summary:
- Total objects detected before: X
- Total objects detected after: Y
- Missing objects: Z
- List of missing object types with counts
- Alert message if any objects are missing

### Code Structure Requirements:

```python
# Main structure should include:

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class SurgicalToolDetector:
    def __init__(self):
        # Initialize YOLO11x model
        pass
    
    def detect_objects(self, image):
        # Detect objects in image and return results
        pass
    
    def compare_detections(self, before_results, after_results):
        # Compare detection results and find missing objects
        pass
    
    def visualize_results(self, image, results, missing_objects=None):
        # Draw bounding boxes and highlight missing objects
        pass

def main():
    # Streamlit app main function
    pass

if __name__ == "__main__":
    main()
```

### UI Requirements:
- Clean, medical-themed interface
- Progress bars during processing
- Clear error messages
- Professional color scheme (blues, whites)
- Large, clear buttons and text
- Responsive layout

### Error Handling:
- Handle missing YOLO model download
- Validate image uploads
- Handle cases where no objects are detected
- Provide helpful error messages

### Output Format:
The application should show:
1. Both original images with detection boxes
2. A comparison table
3. Summary statistics
4. Missing items alert (if any)
5. Confidence scores for detections

### Additional Features:
- Save results to a report file
- Export detection data as CSV
- Adjustable confidence threshold slider
- Option to download annotated images

## Expected Deliverable:
A complete, runnable Python script that can be executed with `streamlit run app.py` and provides a fully functional surgical tool detection PoC interface.

Make sure the code is well-commented, follows Python best practices, and includes clear instructions for setup and usage.