# Object Detection Comparison PoC

A comprehensive web application for object detection and comparison using YOLO11x. This application allows users to compare objects between two images, identify missing or added items, and perform real-time object detection on video streams.

![Object Detection Demo](https://img.shields.io/badge/YOLO-11x-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green) ![Python](https://img.shields.io/badge/Python-3.8+-yellow)

## Features

### 🔍 Core Detection Capabilities
- **Dual Image Comparison**: Upload two images and identify differences in detected objects
- **Real-time Video Processing**: Live object detection via webcam with WebSocket streaming
- **Advanced YOLO11x Integration**: State-of-the-art object detection with 80+ object classes
- **Confidence Threshold Control**: Adjustable detection sensitivity (0.1 - 1.0)
- **Batch Processing**: Support for multiple detection parameters

### 📊 Analysis & Visualization
- **Object Counting**: Automatic counting of detected objects by class
- **Difference Analysis**: Identify missing, added, and changed objects between images
- **Visual Annotations**: Bounding boxes with confidence scores
- **Comparison Reports**: Detailed analysis with statistics and summaries
- **Export Functionality**: Save results as reports and annotated images

### 🌐 Web Interface
- **Modern UI**: Clean, responsive web interface
- **Multiple Input Methods**: File upload, drag-and-drop, camera capture, clipboard paste
- **Real-time Preview**: Live camera feed with object detection overlay
- **Network Access**: LAN sharing capabilities for remote access
- **Progress Indicators**: Real-time processing status and feedback

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam (optional, for real-time detection)
- 4GB+ RAM recommended
- Internet connection (for initial YOLO model download)

### Installation & Launch

#### Windows
```bash
# Clone or download the project
cd OD_PoC

# Run the automated setup
run.bat
```

#### Linux/macOS
```bash
# Clone or download the project
cd OD_PoC

# Make the script executable
chmod +x run.sh

# Run the automated setup
./run.sh
```

#### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Start the application
python app.py
```

### Access the Application
- **Local Access**: http://localhost:3000
- **Network Access**: Visit http://localhost:3000/network-access for LAN sharing instructions

## Usage Guide

### 1. Image Comparison Workflow

#### Step 1: Upload First Image
- Click "Choose File" or drag-and-drop your first image
- Supported formats: JPG, PNG, JPEG
- Or use camera capture / paste from clipboard

#### Step 2: Upload Second Image
- Upload the second image using any supported method
- The system will automatically enable comparison once both images are processed

#### Step 3: Review Results
- View detected objects with bounding boxes
- Check the comparison table showing differences
- Analyze missing/added objects statistics

#### Step 4: Adjust Parameters (Optional)
- Fine-tune confidence threshold (default: 0.15)
- Modify IoU threshold for overlapping detections
- Adjust maximum detections limit

#### Step 5: Export Results
- Download annotated images
- Export detailed reports
- Save comparison data

### 2. Real-time Video Detection
1. Click "Start Live Detection"
2. Allow camera access when prompted
3. Adjust detection parameters in real-time
4. View live object detection overlay
5. Monitor detection statistics

### 3. Network Sharing
1. Visit `/network-access` endpoint
2. Share the provided IP addresses with other devices on your network
3. Access the application from any device on the same network

## Technical Details

### Object Detection Classes
The YOLO11x model detects 80+ object classes including:
- **People & Animals**: person, cat, dog, horse, bird, etc.
- **Vehicles**: car, truck, bus, motorcycle, bicycle, etc.
- **Household Items**: chair, table, bed, couch, tv, etc.
- **Food & Drinks**: apple, banana, cup, bottle, etc.
- **Electronics**: laptop, phone, keyboard, mouse, etc.
- **Sports Equipment**: ball, racket, skateboard, etc.

### API Endpoints

#### Main Endpoints
- `GET /` - Main application interface
- `POST /detect/{image_number}` - Process uploaded images (1 or 2)
- `POST /detect-base64/{image_number}` - Process base64 encoded images
- `POST /compare` - Compare two processed images
- `POST /adjust-parameters` - Re-analyze with new parameters
- `GET /status` - Get current workflow status

#### Utility Endpoints
- `GET /health` - Health check and model status
- `GET /network-access` - Network sharing guide
- `POST /reset-workflow` - Clear current session
- `POST /save-results` - Export results

#### WebSocket
- `WS /ws/video/{client_id}` - Real-time video processing

### Configuration Options

#### Detection Parameters
```python
# Adjustable parameters
conf_threshold: float = 0.15    # Confidence threshold (0.1 - 1.0)
iou_threshold: float = 0.45     # IoU threshold for NMS
max_detection: int = 500        # Maximum detections per image
```

#### Model Settings
- **Model**: YOLO11x (yolo11x.pt)
- **Input Size**: 640px (adjustable)
- **Processing**: Optimized for both accuracy and speed
- **Classes**: 80 COCO dataset classes

## File Structure

```
OD_PoC/
├── app.py                  # Main FastAPI application
├── requirements.txt        # Python dependencies
├── yolo11x.pt             # YOLO11x model file
├── run.bat                # Windows startup script
├── run.sh                 # Linux/macOS startup script
├── instruction.md         # Development specifications
├── quick_test.md         # Testing guide
├── certs/                # SSL certificates (if needed)
├── static/               # Static web assets
│   ├── css/style.css    # Application styles
│   └── js/app.js        # Frontend JavaScript
└── templates/           # HTML templates
    ├── index.html       # Main application interface
    └── network_access.html # Network sharing guide
```

## System Requirements

### Minimum Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space (for model files)
- **GPU**: Optional (CUDA-compatible for faster processing)

### Recommended Environment
- **Python**: 3.9+
- **Platform**: Windows 10+, Ubuntu 20.04+, macOS 11+
- **Browser**: Modern browser with WebRTC support
- **Network**: Stable internet for initial model download

## Troubleshooting

### Common Issues

#### Model Loading Error
```bash
# Re-download YOLO model
pip install ultralytics --upgrade
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
```

#### Port Already in Use
- Change port in `app.py` line 747: `port=3000` to another port
- Or kill existing process using the port

#### Camera Access Denied
- Check browser permissions for camera access
- Ensure no other applications are using the camera
- Refresh the page and allow camera access

#### Memory Issues
- Reduce `max_detection` parameter
- Lower image resolution before upload
- Close other memory-intensive applications

### Performance Optimization
- Use GPU acceleration if available
- Reduce image size for faster processing
- Adjust confidence threshold for better speed/accuracy balance
- Use smaller YOLO model variant for speed (yolo11n.pt instead of yolo11x.pt)

## Contributing

This is a Proof of Concept project. For improvements or issues:
1. Test the application thoroughly
2. Document any bugs or feature requests
3. Consider performance optimizations
4. Ensure compatibility across different platforms

## License

This project is for demonstration and educational purposes. Please ensure compliance with YOLO and dependency licenses for commercial use.

## Version History

- **v1.0.0** - Initial release with dual image comparison
- **v1.1.0** - Added real-time video processing
- **v1.2.0** - Enhanced network sharing capabilities
- **Current** - Optimized performance and expanded detection classes

---

For technical support or questions about the implementation, refer to the `instruction.md` file or test the application using `quick_test.md`.
