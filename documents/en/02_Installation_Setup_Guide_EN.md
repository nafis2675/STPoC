# Installation and Setup Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Start Guide](#quick-start-guide)
3. [Detailed Installation Steps](#detailed-installation-steps)
4. [Configuration Options](#configuration-options)
5. [Network Setup](#network-setup)
6. [Troubleshooting Installation](#troubleshooting-installation)
7. [Performance Optimization](#performance-optimization)

## System Requirements

### Minimum Requirements
```yaml
Hardware:
  - CPU: 4+ cores (Intel i5/AMD Ryzen 5 or better)
  - RAM: 4GB minimum, 8GB recommended
  - Storage: 2GB free space for model files
  - GPU: Optional (NVIDIA CUDA-compatible for acceleration)
  - Webcam: Optional (for real-time detection)

Operating System:
  - Windows: Windows 10+ (64-bit)
  - macOS: macOS 11+ (Big Sur or later)
  - Linux: Ubuntu 20.04+ or equivalent

Network:
  - Internet: Required for initial YOLO model download
  - LAN: Optional (for multi-device access)
```

### Recommended Requirements
```yaml
Hardware:
  - CPU: 8+ cores (Intel i7/AMD Ryzen 7)
  - RAM: 16GB for optimal performance
  - GPU: NVIDIA RTX series or GTX 1060+ with 6GB VRAM
  - SSD: For faster model loading and processing

Software:
  - Python: 3.9+ (3.10 recommended)
  - Browser: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
  - Git: For version control (optional)
```

## Quick Start Guide

### Option 1: Automated Setup (Recommended)

#### Windows
```batch
# 1. Download and extract the project
# 2. Double-click run.bat
run.bat
```

#### Linux/macOS
```bash
# 1. Download and extract the project
# 2. Make script executable and run
chmod +x run.sh
./run.sh
```

### Option 2: Manual Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start the application
python app.py

# 3. Open browser to http://localhost:3000
```

### First Access
1. Open your web browser
2. Navigate to `http://localhost:3000`
3. Select your detection mode (Image Comparison or Real-time Video)
4. Upload an image or start camera to test functionality

## Detailed Installation Steps

### Step 1: Environment Preparation

#### Python Environment Setup
```bash
# Check Python version (should be 3.8+)
python --version

# Create virtual environment (recommended)
python -m venv od_env

# Activate virtual environment
# Windows:
od_env\Scripts\activate
# Linux/macOS:
source od_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### Alternative: Using Conda
```bash
# Create conda environment
conda create -n od_env python=3.10
conda activate od_env

# Install pip packages
pip install -r requirements.txt
```

### Step 2: Dependency Installation

#### Core Dependencies Analysis
```yaml
requirements.txt breakdown:
  fastapi>=0.100.0         # Web framework
  uvicorn[standard]>=0.20.0 # ASGI server
  python-multipart>=0.0.5  # File upload support
  ultralytics>=8.0.0       # YOLO implementation
  opencv-python>=4.8.0     # Computer vision
  pillow>=10.0.0          # Image processing
  numpy>=1.21.0           # Numerical computing
  torch>=2.0.0            # Deep learning framework
  torchvision>=0.15.0     # Vision models
  torchaudio>=2.0.0       # Audio processing
  jinja2>=3.0.0           # HTML templating
```

#### Installation with Version Verification
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify critical packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import ultralytics; print('YOLO available')"
```

#### GPU Support (Optional but Recommended)
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If CUDA not available, install PyTorch with CUDA
# Visit: https://pytorch.org/get-started/locally/
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: YOLO Model Setup

#### Automatic Model Download
```bash
# The first run will automatically download yolo11x.pt
# This may take 5-10 minutes depending on internet speed
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
```

#### Manual Model Download (if needed)
```bash
# Download directly from Ultralytics
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11x.pt
# Or use curl on Windows:
curl -L -o yolo11x.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11x.pt
```

#### Verify Model Installation
```python
# Test script - save as test_model.py
from ultralytics import YOLO
import cv2
import numpy as np

try:
    # Load model
    model = YOLO('yolo11x.pt')
    print("✅ YOLO model loaded successfully")
    
    # Test with dummy image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    results = model(test_img)
    print("✅ Model inference test passed")
    
    # Print available classes
    print(f"📊 Model supports {len(model.names)} object classes")
    
except Exception as e:
    print(f"❌ Model test failed: {e}")
```

### Step 4: Application Configuration

#### Basic Configuration Check
```python
# config_check.py - Run this to verify setup
import os
import sys
from pathlib import Path

def check_configuration():
    print("🔍 Checking Configuration...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check required files
    required_files = [
        'app.py',
        'requirements.txt',
        'yolo11x.pt',
        'static/css/style.css',
        'static/js/app.js',
        'templates/index.html'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ Missing: {file}")
            return False
    
    # Check imports
    try:
        import fastapi, uvicorn, ultralytics, cv2, PIL, numpy, torch
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    if check_configuration():
        print("🎉 Configuration check passed! Ready to start.")
    else:
        print("🔧 Please fix the issues above before starting.")
```

### Step 5: First Launch

#### Starting the Application
```bash
# Method 1: Direct Python execution
python app.py

# Method 2: Using Uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 3000

# Method 3: Development mode with auto-reload
uvicorn app:app --host 0.0.0.0 --port 3000 --reload
```

#### Verification Steps
```bash
# Check server is running
curl http://localhost:3000/health

# Expected response:
# {"status":"healthy","model_loaded":true}
```

## Configuration Options

### Port Configuration
```python
# In app.py, modify the last line:
if __name__ == "__main__":
    import uvicorn
    # Change port here if 3000 is occupied
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Example: port 8000
```

### Detection Parameters
```python
# Default parameters in ObjectDetectionService
DEFAULT_CONF_THRESHOLD = 0.50    # Confidence threshold
DEFAULT_IOU_THRESHOLD = 0.15     # IoU threshold for NMS
DEFAULT_MAX_DETECTION = 500      # Maximum detections per image
DEFAULT_IMAGE_SIZE = 640         # Input image size for YOLO
```

### Session Configuration
```python
# Session cleanup configuration
MAX_SESSION_AGE_HOURS = 24       # Auto cleanup after 24 hours
CLEANUP_INTERVAL_SECONDS = 3600  # Check every hour
```

### Performance Settings
```python
# Real-time processing settings
VIDEO_PROCESSING_FPS = 5         # Processing FPS for video
VIDEO_DISPLAY_FPS = 30           # Display refresh rate
FRAME_PROCESSING_QUALITY = 0.6   # JPEG quality for frame processing
REALTIME_IMAGE_SIZE = 416        # Smaller size for real-time processing
```

## Network Setup

### Local Access Configuration
```python
# Default binding (localhost only)
uvicorn.run(app, host="127.0.0.1", port=3000)

# LAN access (all interfaces)
uvicorn.run(app, host="0.0.0.0", port=3000)
```

### Firewall Configuration

#### Windows Firewall
```batch
# Allow Python through Windows Firewall
netsh advfirewall firewall add rule name="Object Detection App" dir=in action=allow program="C:\Python39\python.exe" enable=yes

# Or allow specific port
netsh advfirewall firewall add rule name="OD Port 3000" dir=in action=allow protocol=TCP localport=3000
```

#### Linux iptables
```bash
# Allow port 3000
sudo iptables -A INPUT -p tcp --dport 3000 -j ACCEPT

# For Ubuntu/Debian with ufw
sudo ufw allow 3000
```

#### macOS
```bash
# macOS typically allows local network access by default
# Check System Preferences > Security & Privacy > Firewall if needed
```

### Network Access Verification
```bash
# Find your local IP address
# Windows:
ipconfig | findstr IPv4

# Linux/macOS:
hostname -I
# or
ifconfig | grep inet

# Test from another device
curl http://[YOUR_IP]:3000/health
```

## Troubleshooting Installation

### Common Issues and Solutions

#### Issue 1: Python Version Compatibility
```bash
# Symptoms: Package installation fails
# Solution: Verify Python version
python --version

# If Python < 3.8, update Python:
# Windows: Download from python.org
# Ubuntu: sudo apt update && sudo apt install python3.10
# macOS: brew install python@3.10
```

#### Issue 2: PyTorch Installation Issues
```bash
# Symptoms: CUDA/GPU not detected
# Solution: Reinstall PyTorch with proper CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (if no GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Issue 3: OpenCV Import Error
```bash
# Symptoms: "ImportError: libGL.so.1"
# Linux solution:
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# Alternative OpenCV installation:
pip uninstall opencv-python
pip install opencv-python-headless
```

#### Issue 4: YOLO Model Download Fails
```bash
# Symptoms: Model download timeout/fails
# Solution: Manual download and placement
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11x.pt
# Place in project root directory
```

#### Issue 5: Port Already in Use
```bash
# Symptoms: "Port 3000 is already in use"
# Find process using port:
# Windows:
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Linux/macOS:
lsof -i :3000
kill -9 <PID>

# Or change port in app.py
```

#### Issue 6: Camera Access Denied
```bash
# Symptoms: Camera not accessible in browser
# Solutions:
1. Use HTTPS instead of HTTP (required for mobile)
2. Grant camera permissions in browser
3. Check if other applications are using camera
4. Try different browser
```

### Debug Mode Setup
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints to app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=3000,
        log_level="debug",
        access_log=True
    )
```

### Memory and Performance Issues
```bash
# Check memory usage
# Windows:
tasklist /fi "imagename eq python.exe"

# Linux/macOS:
ps aux | grep python
top -p $(pgrep python)

# Reduce memory usage:
# - Lower max_detection parameter
# - Use smaller YOLO model (yolo11n.pt instead of yolo11x.pt)
# - Reduce image resolution before processing
```

## Performance Optimization

### Hardware Optimization

#### GPU Acceleration Setup
```python
# Check GPU availability and setup
import torch

def setup_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Set GPU memory management
        torch.backends.cudnn.benchmark = True
        return device
    else:
        print("⚠️  GPU not available, using CPU")
        return torch.device("cpu")

# Apply in ObjectDetectionService
device = setup_gpu()
model = YOLO('yolo11x.pt')
model.to(device)
```

#### Memory Optimization
```python
# Configure memory-efficient settings
MEMORY_EFFICIENT_CONFIG = {
    'max_sessions': 10,           # Limit concurrent sessions
    'image_max_size': 1920,       # Resize large images
    'video_buffer_size': 5,       # Limit video frame buffer
    'cleanup_interval': 1800      # More frequent cleanup (30 min)
}
```

### Software Optimization

#### Model Optimization
```python
# Use appropriate model size based on hardware
MODEL_CONFIGS = {
    'high_performance': 'yolo11x.pt',    # Best accuracy, slower
    'balanced': 'yolo11l.pt',            # Good balance
    'fast': 'yolo11m.pt',                # Faster processing
    'mobile': 'yolo11n.pt'               # Fastest, mobile-friendly
}

# Switch model based on requirements:
# In ObjectDetectionService.__init__():
self.model_name = MODEL_CONFIGS['balanced']  # Change as needed
```

#### Processing Optimization
```python
# Optimize detection parameters for speed
FAST_DETECTION_CONFIG = {
    'conf_threshold': 0.3,        # Higher threshold = fewer detections
    'iou_threshold': 0.5,         # Higher threshold = less overlap processing
    'max_detection': 100,         # Limit maximum detections
    'imgsz': 416,                 # Smaller input size = faster processing
    'half': True                  # Use FP16 precision (if GPU supports)
}
```

### Network Optimization
```python
# Configure for better network performance
NETWORK_CONFIG = {
    'websocket_timeout': 30,
    'max_frame_size': 1024*1024,  # 1MB max frame size
    'compression_quality': 0.7,   # JPEG compression for frames
    'batch_processing': False     # Disable batching for real-time
}
```

---

This installation guide provides comprehensive setup instructions for the Object Detection Web Application. Follow the steps in order for the smoothest installation experience. For architecture details, refer to the Architecture Documentation.
