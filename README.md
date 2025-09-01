# Object Detection Web App

A modern web application for object detection using YOLOv11x model with image upload, camera capture, and clipboard paste functionality.

## Features

### 🔄 **Two-Image Workflow** (Matches Your Original Colab Cells)
- **Step 1**: Process First Image (Cell 5 equivalent)
- **Step 2**: Process Second Image (Cell 6 equivalent) 
- **Step 3**: Automatic Comparison (Cell 7 equivalent)
- **Step 4**: Parameter Adjustment (Cell 8 equivalent)
- **Step 5**: Results Saving (Cell 9 equivalent)

### 📤 **Multiple Input Methods**
- **File Upload**: Drag & drop or click to upload images
- **Camera Capture**: Take photos directly from webcam
- **Clipboard Paste**: Paste images with Ctrl+V or paste button

### 🎯 **Advanced Object Detection**
- **YOLOv11x Model**: Maximum detection accuracy (same as your Colab)
- **Configurable Parameters**: Real-time adjustment of confidence, IoU, and max detections
- **Parameter Re-analysis**: Adjust settings and re-process both images (Cell 8 feature)

### 📊 **Comprehensive Comparison**
- **Side-by-side Analysis**: Visual comparison of both annotated images
- **Detailed Change Report**: Object count changes, new/removed classes
- **Statistical Summary**: Total objects, unique classes, and differences

### 💾 **Results Export** 
- **Report Download**: Text file with complete analysis (Cell 9 feature)
- **Image Download**: Save annotated images with detection boxes
- **Workflow Reset**: Start fresh analysis with new image pairs

## Prerequisites

- Python 3.8 or higher
- Webcam (for camera capture functionality)
- Modern web browser with camera permissions

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the YOLO model (will be automatically downloaded on first run):
   - The app uses `yolo11x.pt` which will be downloaded automatically by ultralytics

## Usage

1. Start the web application:
```bash
python app.py
```

2. Open your browser and go to: `http://localhost:8000`

3. Choose your input method:
   - **Upload**: Click "Upload Image" to select a file
   - **Camera**: Click "Take Photo" to capture from your webcam
   - **Paste**: Copy an image and click "Paste Image" or use Ctrl+V

4. Adjust detection parameters if needed:
   - **Confidence Threshold**: Lower values detect more objects (may include false positives)
   - **IoU Threshold**: Controls overlap filtering between detections
   - **Max Detections**: Maximum number of objects to detect

5. View results:
   - Original and annotated images side by side
   - Detection statistics and detailed object list
   - Save results for comparison

6. Compare images:
   - Save your first detection result
   - Process a second image
   - Click "Compare with Previous" to see differences

## API Endpoints

- `GET /`: Main web interface
- `POST /detect`: Upload and detect objects in image file
- `POST /detect-base64`: Detect objects in base64 encoded image
- `POST /compare`: Compare two detection results
- `GET /health`: Health check endpoint

## Configuration

You can modify detection parameters in the web interface:

- **Confidence Threshold**: 0.05 - 0.95 (default: 0.15)
- **IoU Threshold**: 0.1 - 0.9 (default: 0.45)  
- **Max Detections**: 10 - 1000 (default: 500)

## Technical Details

- **Backend**: FastAPI with Python
- **Frontend**: Vanilla JavaScript with modern CSS
- **AI Model**: YOLOv11x (ultralytics)
- **Image Processing**: OpenCV and Pillow
- **UI Framework**: Custom responsive design

## Browser Compatibility

- Chrome 80+ (recommended)
- Firefox 75+
- Safari 13+
- Edge 80+

## Troubleshooting

### Camera Issues
- Ensure camera permissions are granted
- Check if camera is being used by another application
- Try refreshing the page and granting permissions again

### Model Loading Issues
- Ensure stable internet connection for first-time model download
- Check available disk space (model is ~80MB)
- Verify Python and dependency versions

### Performance
- Large images may take longer to process
- Consider reducing max detections for faster processing
- Ensure sufficient RAM (recommended: 4GB+)

## File Structure

```
OD_PoC/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Web interface template
├── static/
│   ├── css/
│   │   └── style.css  # Application styles
│   └── js/
│       └── app.js     # Frontend JavaScript
└── README.md          # This file
```

## License

This project is for educational and demonstration purposes.
