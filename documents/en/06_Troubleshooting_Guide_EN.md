# Comprehensive Troubleshooting Guide

## Table of Contents
1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Runtime Problems](#runtime-problems)
4. [Performance Issues](#performance-issues)
5. [Network and Connectivity](#network-and-connectivity)
6. [Camera and Media Problems](#camera-and-media-problems)
7. [Model and AI Issues](#model-and-ai-issues)
8. [Browser and Frontend Issues](#browser-and-frontend-issues)
9. [System-Specific Problems](#system-specific-problems)
10. [Advanced Debugging](#advanced-debugging)

## Quick Diagnostics

### System Health Check Script
Create and run this diagnostic script to quickly identify common issues:

```python
#!/usr/bin/env python3
"""
Object Detection App - System Health Check
Run this script to diagnose common issues
"""

import sys
import subprocess
import importlib
import platform
import os
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("❌ ERROR: Python 3.8+ required")
        return False
    elif version >= (3, 8) and version < (3, 12):
        print("✅ Python version compatible")
        return True
    else:
        print("⚠️  WARNING: Python 3.12+ may have package compatibility issues")
        return True

def check_required_packages():
    """Check if all required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'ultralytics', 'cv2', 'PIL', 
        'numpy', 'torch', 'torchvision', 'jinja2'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_gpu_availability():
    """Check GPU and CUDA availability"""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ CUDA Available: {device_count} device(s)")
            print(f"   Primary GPU: {device_name}")
            print(f"   Memory: {memory:.1f} GB")
            return True
        else:
            print("⚠️  No CUDA GPU detected - using CPU")
            return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def check_model_file():
    """Check if YOLO model file exists"""
    model_path = Path("yolo11x.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ YOLO model found: {size_mb:.1f} MB")
        return True
    else:
        print("❌ YOLO model file (yolo11x.pt) not found")
        print("   Run: python -c \"from ultralytics import YOLO; YOLO('yolo11x.pt')\"")
        return False

def check_port_availability():
    """Check if default port 3000 is available"""
    import socket
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('localhost', 3000))
        print("✅ Port 3000 available")
        return True
    except OSError:
        print("❌ Port 3000 is in use")
        print("   Change port in app.py or stop conflicting service")
        return False
    finally:
        sock.close()

def check_memory():
    """Check available system memory"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        
        print(f"Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
        
        if available_gb < 2:
            print("⚠️  WARNING: Low memory may cause issues")
            return False
        else:
            print("✅ Memory sufficient")
            return True
    except ImportError:
        print("ℹ️  Install psutil for memory check: pip install psutil")
        return True

def main():
    print("=" * 50)
    print("Object Detection App - System Health Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("GPU/CUDA", check_gpu_availability),
        ("YOLO Model", check_model_file),
        ("Port Availability", check_port_availability),
        ("System Memory", check_memory)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n{name}:")
        try:
            result = check_func()
            if isinstance(result, tuple):
                passed, details = result
                all_passed &= passed
            else:
                all_passed &= result
        except Exception as e:
            print(f"❌ Check failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed! System ready to run.")
    else:
        print("⚠️  Some issues detected. See details above.")
        print("📖 Refer to troubleshooting guide for solutions.")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

Save this as `health_check.py` and run: `python health_check.py`

## Installation Issues

### 1. Python Version Problems

#### Issue: "Python version too old"
```bash
# Symptoms
ModuleNotFoundError: No module named '_ctypes'
SyntaxError: invalid syntax (f-strings, type hints)

# Diagnosis
python --version
# Shows Python < 3.8

# Solutions
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.10 python3.10-pip python3.10-venv

# Windows:
# Download from python.org and install
# Or use Microsoft Store

# macOS:
brew install python@3.10
```

#### Issue: Multiple Python versions conflict
```bash
# Symptoms
pip installs packages but import fails
Different Python version when running script

# Diagnosis
which python
which pip
python --version
pip --version

# Solutions
# Use explicit python3/pip3
python3 -m pip install -r requirements.txt
python3 app.py

# Or create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. Package Installation Failures

#### Issue: PyTorch installation fails
```bash
# Common errors:
ERROR: Could not find a version that satisfies the requirement torch
OSError: [WinError 5] Access is denied
ERROR: Microsoft Visual C++ 14.0 is required

# Solutions:

# 1. Install Visual C++ Build Tools (Windows)
# Download from Microsoft: "Microsoft C++ Build Tools"

# 2. Use conda instead of pip
conda create -n od_env python=3.10
conda activate od_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install CPU-only version first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Use pre-compiled wheels
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

#### Issue: OpenCV installation problems
```bash
# Common errors:
ImportError: libGL.so.1: cannot open shared object file
ImportError: No module named 'cv2'

# Solutions:

# Linux - missing system dependencies:
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Alternative OpenCV installation:
pip uninstall opencv-python
pip install opencv-python-headless  # No GUI dependencies

# Build from source (if needed):
pip install cmake
pip install opencv-python --no-binary opencv-python
```

#### Issue: Ultralytics installation problems
```bash
# Common errors:
ERROR: Failed building wheel for ultralytics
ModuleNotFoundError: No module named 'ultralytics'

# Solutions:

# 1. Update pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision  # Install PyTorch first
pip install ultralytics

# 2. Install from GitHub (latest version)
pip install git+https://github.com/ultralytics/ultralytics.git

# 3. Manual dependency installation
pip install numpy pillow pyyaml requests matplotlib seaborn
pip install ultralytics --no-deps  # Skip dependency check
```

### 3. Environment Setup Issues

#### Issue: Virtual environment problems
```bash
# Command not found errors:
'python' is not recognized as an internal or external command
bash: python: command not found

# Solutions:

# Windows - Add Python to PATH:
# 1. Find Python installation directory
# 2. Add to System PATH environment variable
# 3. Restart command prompt

# Linux/Mac - Install Python properly:
sudo apt install python3 python3-pip python3-venv  # Ubuntu
brew install python  # macOS

# Create and use virtual environment:
python3 -m venv od_env
source od_env/bin/activate
pip install --upgrade pip
```

## Runtime Problems

### 1. Application Startup Issues

#### Issue: FastAPI server won't start
```bash
# Common errors:
ModuleNotFoundError: No module named 'fastapi'
Address already in use
Permission denied

# Diagnosis and Solutions:

# 1. Check if packages installed in correct environment
which python
pip list | grep fastapi

# 2. Check port conflicts
lsof -i :3000  # Linux/Mac
netstat -ano | findstr :3000  # Windows

# Kill conflicting process:
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows

# 3. Change port in app.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Change port

# 4. Run with different parameters
python -m uvicorn app:app --host 0.0.0.0 --port 3000
```

#### Issue: YOLO model loading fails
```bash
# Common errors:
FileNotFoundError: [Errno 2] No such file or directory: 'yolo11x.pt'
RuntimeError: CUDA error: out of memory
AttributeError: 'YOLO' object has no attribute 'names'

# Solutions:

# 1. Download model manually:
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"

# 2. Check file permissions and location:
ls -la yolo11x.pt
# Should be in project root directory

# 3. Handle CUDA memory issues:
import torch
torch.cuda.empty_cache()

# Or use CPU:
model = YOLO('yolo11x.pt')
model.to('cpu')

# 4. Use smaller model:
model = YOLO('yolo11n.pt')  # Nano version - much smaller
```

### 2. Processing Failures

#### Issue: Image upload fails
```bash
# Common errors:
HTTP 413 Request Entity Too Large
HTTP 422 Unprocessable Entity
"Invalid image format"

# Solutions:

# 1. Check file size limits:
# In nginx.conf (if using nginx):
client_max_body_size 100M;

# In FastAPI (app.py):
from fastapi import File, UploadFile

@app.post("/detect/")
async def detect(file: UploadFile = File(..., max_length=50*1024*1024)):  # 50MB limit

# 2. Validate image format:
allowed_types = {'image/jpeg', 'image/png', 'image/jpg', 'image/webp'}
if file.content_type not in allowed_types:
    raise HTTPException(400, "Unsupported file type")

# 3. Handle corrupted images:
try:
    image = Image.open(io.BytesIO(await file.read()))
    image.verify()  # Check if image is valid
except Exception as e:
    raise HTTPException(400, f"Invalid image: {str(e)}")
```

#### Issue: Detection processing hangs
```bash
# Symptoms:
Request never completes
High CPU/GPU usage
Memory keeps increasing

# Diagnosis:
import psutil
import time

def monitor_process():
    process = psutil.Process()
    start_time = time.time()
    
    while time.time() - start_time < 30:  # Monitor for 30 seconds
        cpu_percent = process.cpu_percent()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"CPU: {cpu_percent}%, Memory: {memory_mb:.1f}MB")
        time.sleep(1)

# Solutions:

# 1. Set processing timeout:
import asyncio

async def process_with_timeout():
    try:
        result = await asyncio.wait_for(
            detection_service.detect_objects(image),
            timeout=30.0  # 30 second timeout
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(408, "Processing timeout")

# 2. Limit image size:
max_pixels = 1920 * 1080
if image.width * image.height > max_pixels:
    ratio = (max_pixels / (image.width * image.height)) ** 0.5
    new_size = (int(image.width * ratio), int(image.height * ratio))
    image = image.resize(new_size, Image.LANCZOS)

# 3. Monitor memory and cleanup:
if torch.cuda.is_available():
    torch.cuda.empty_cache()
import gc
gc.collect()
```

### 3. Session Management Problems

#### Issue: Session data lost
```bash
# Symptoms:
"Session not found" errors
Data disappears between requests
Inconsistent behavior

# Solutions:

# 1. Check session ID handling:
# Frontend - store session ID properly:
const sessionId = localStorage.getItem('od_session_id');
if (!sessionId) {
    const response = await fetch('/create-session', {method: 'POST'});
    const data = await response.json();
    localStorage.setItem('od_session_id', data.session_id);
}

# 2. Debug session storage:
def debug_session_manager():
    print(f"Active sessions: {len(session_manager.sessions)}")
    for sid, data in session_manager.sessions.items():
        print(f"Session {sid}: {list(data.keys())}")

# 3. Increase session timeout:
# In app.py:
MAX_SESSION_AGE_HOURS = 48  # Increase from 24 to 48 hours

# 4. Implement session persistence:
import pickle

class PersistentSessionManager(SessionManager):
    def save_sessions(self):
        with open('sessions.pkl', 'wb') as f:
            pickle.dump(self.sessions, f)
    
    def load_sessions(self):
        try:
            with open('sessions.pkl', 'rb') as f:
                self.sessions = pickle.load(f)
        except FileNotFoundError:
            pass
```

## Performance Issues

### 1. Slow Processing Speed

#### Issue: Detection takes too long
```python
# Diagnosis - Add timing measurements:
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@timing_decorator
def detect_objects(self, image):
    # Your detection code here
    pass

# Solutions:

# 1. Optimize image size:
def optimize_image_for_detection(image):
    # Resize large images
    max_size = 1280
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.LANCZOS)
    return image

# 2. Use GPU acceleration:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Optimize YOLO parameters:
results = model.predict(
    source=image,
    imgsz=640,        # Smaller input size
    conf=0.5,         # Higher confidence threshold
    iou=0.5,          # Higher IoU threshold  
    max_det=100,      # Limit detections
    half=True,        # Use FP16 precision
    verbose=False     # Disable logging
)

# 4. Use smaller model for real-time:
model = YOLO('yolo11n.pt')  # Nano - fastest
# vs
model = YOLO('yolo11x.pt')  # Extra large - most accurate
```

#### Issue: High memory usage
```python
# Diagnosis - Monitor memory:
import tracemalloc
import psutil

def monitor_memory():
    tracemalloc.start()
    
    # Your code here
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    print(f"Memory - Current: {current / 1024 / 1024:.1f}MB")
    print(f"Memory - Peak: {peak / 1024 / 1024:.1f}MB")
    print(f"Process Memory: {memory_mb:.1f}MB")

# Solutions:

# 1. Implement memory cleanup:
def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

# Call after each detection:
result = detect_objects(image)
cleanup_memory()

# 2. Limit concurrent processing:
import asyncio

# Global semaphore to limit concurrent detections
detection_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent

async def detect_with_limit(image):
    async with detection_semaphore:
        return detection_service.detect_objects(image)

# 3. Implement LRU cache with size limit:
from functools import lru_cache
import hashlib

def image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

@lru_cache(maxsize=10)  # Cache last 10 results
def cached_detection(image_hash, params_hash):
    # Detection logic here
    pass
```

### 2. Network Performance

#### Issue: Slow file uploads
```python
# Solutions:

# 1. Implement client-side image compression:
function compressImage(file, maxWidth = 1920, quality = 0.8) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = () => {
            const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
            canvas.width = img.width * ratio;
            canvas.height = img.height * ratio;
            
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(resolve, 'image/jpeg', quality);
        };
        
        img.src = URL.createObjectURL(file);
    });
}

# 2. Add progress tracking:
async function uploadWithProgress(file, onProgress) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();
        formData.append('image', file);
        
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                onProgress(percentComplete);
            }
        });
        
        xhr.onload = () => resolve(JSON.parse(xhr.responseText));
        xhr.onerror = reject;
        
        xhr.open('POST', '/detect/1');
        xhr.send(formData);
    });
}

# 3. Implement chunked upload for large files:
# Server-side FastAPI:
from fastapi import Request

@app.post("/upload-chunk")
async def upload_chunk(request: Request):
    chunk_number = int(request.headers.get("chunk-number"))
    total_chunks = int(request.headers.get("total-chunks"))
    file_id = request.headers.get("file-id")
    
    chunk_data = await request.body()
    
    # Store chunk
    chunk_dir = Path(f"temp/{file_id}")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    with open(chunk_dir / f"chunk_{chunk_number}", "wb") as f:
        f.write(chunk_data)
    
    # Check if all chunks received
    if len(list(chunk_dir.glob("chunk_*"))) == total_chunks:
        # Reassemble file
        assembled_file = reassemble_chunks(chunk_dir, total_chunks)
        return {"status": "complete", "file": assembled_file}
    
    return {"status": "chunk_received"}
```

## Network and Connectivity

### 1. Local Network Access Issues

#### Issue: Cannot access from other devices
```bash
# Common problems:
Connection refused
Timeout errors
Firewall blocking

# Diagnosis:

# 1. Check server binding:
# In app.py, ensure:
uvicorn.run(app, host="0.0.0.0", port=3000)  # Not "127.0.0.1"

# 2. Find local IP address:
# Windows:
ipconfig | findstr IPv4

# Linux/Mac:
hostname -I
ifconfig | grep inet

# 3. Test connectivity:
# From another device:
curl http://192.168.1.100:3000/health  # Replace with your IP

# Solutions:

# 1. Configure firewall (Windows):
netsh advfirewall firewall add rule name="Object Detection App" dir=in action=allow program="python.exe" enable=yes
netsh advfirewall firewall add rule name="OD Port 3000" dir=in action=allow protocol=TCP localport=3000

# 2. Configure firewall (Linux):
sudo ufw allow 3000
# Or iptables:
sudo iptables -A INPUT -p tcp --dport 3000 -j ACCEPT

# 3. Router/Network configuration:
# Check if devices are on same subnet
# Ensure no network isolation enabled
# Check for corporate firewalls
```

#### Issue: WebSocket connection fails
```javascript
// Common errors:
WebSocket connection to 'ws://192.168.1.100:3000/ws/video/client123' failed
Error during WebSocket handshake

// Diagnosis and Solutions:

// 1. Check WebSocket URL construction:
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${wsProtocol}//${window.location.host}/ws/video/${clientId}`;

// 2. Implement connection retry:
class ReconnectingWebSocket {
    constructor(url) {
        this.url = url;
        this.reconnectInterval = 1000;
        this.maxReconnectInterval = 30000;
        this.reconnectDecay = 1.5;
        this.connect();
    }
    
    connect() {
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectInterval = 1000;
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected, reconnecting...');
            setTimeout(() => {
                this.reconnectInterval = Math.min(
                    this.maxReconnectInterval,
                    this.reconnectInterval * this.reconnectDecay
                );
                this.connect();
            }, this.reconnectInterval);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
}

// 3. Handle network changes:
window.addEventListener('online', () => {
    console.log('Network connectivity restored');
    // Reconnect WebSocket
});

window.addEventListener('offline', () => {
    console.log('Network connectivity lost');
    // Show offline message
});
```

### 2. HTTPS and Security Issues

#### Issue: Camera access denied on mobile
```javascript
// Problem: Mobile browsers require HTTPS for camera access

// Solutions:

// 1. Check if secure context:
if (!window.isSecureContext && /Mobi|Android/i.test(navigator.userAgent)) {
    showHTTPSWarning();
}

function showHTTPSWarning() {
    alert(`
        Camera access requires HTTPS on mobile devices.
        
        Solutions:
        1. Use file upload instead of camera
        2. Access via HTTPS
        3. Use desktop browser
        
        The file upload feature works on all devices!
    `);
}

// 2. Implement HTTPS with self-signed certificate:
# Generate certificate:
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Run with HTTPS:
uvicorn app:app --host 0.0.0.0 --port 443 --ssl-keyfile=key.pem --ssl-certfile=cert.pem

// 3. Graceful fallback for insecure context:
async function requestCamera() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera API not available');
        }
        
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        return stream;
    } catch (error) {
        if (error.name === 'NotAllowedError') {
            showCameraPermissionHelp();
        } else {
            showCameraFallback();
        }
        throw error;
    }
}

function showCameraFallback() {
    const message = `
        Camera not available. Please:
        1. Use the "Upload Image" button instead
        2. Take a photo with your phone's camera app
        3. Upload the photo using the file selector
    `;
    showUserMessage(message);
}
```

## Camera and Media Problems

### 1. Camera Access Issues

#### Issue: Camera permission denied
```javascript
// Comprehensive camera troubleshooting:

class CameraManager {
    constructor() {
        this.stream = null;
        this.constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment'  // Back camera on mobile
            }
        };
    }
    
    async requestCamera() {
        try {
            // Check for API availability
            if (!navigator.mediaDevices?.getUserMedia) {
                throw new Error('getUserMedia not supported');
            }
            
            // Request camera access
            this.stream = await navigator.mediaDevices.getUserMedia(this.constraints);
            return this.stream;
            
        } catch (error) {
            return this.handleCameraError(error);
        }
    }
    
    handleCameraError(error) {
        const errorMessages = {
            'NotAllowedError': 'Camera access denied. Please allow camera permission and refresh.',
            'NotFoundError': 'No camera found on this device.',
            'NotReadableError': 'Camera is already in use by another application.',
            'OverconstrainedError': 'Camera constraints cannot be satisfied.',
            'SecurityError': 'Camera access blocked by security policy.',
            'TypeError': 'Camera constraints are invalid.'
        };
        
        const message = errorMessages[error.name] || `Camera error: ${error.message}`;
        console.error('Camera error:', error);
        
        // Show user-friendly message
        this.showCameraErrorDialog(error.name, message);
        
        // Suggest alternatives
        this.showAlternatives();
        
        throw new Error(message);
    }
    
    showCameraErrorDialog(errorType, message) {
        const dialog = document.createElement('div');
        dialog.className = 'camera-error-dialog';
        dialog.innerHTML = `
            <div class="dialog-content">
                <h3>Camera Access Issue</h3>
                <p><strong>Error:</strong> ${errorType}</p>
                <p>${message}</p>
                
                <div class="error-solutions">
                    <h4>Troubleshooting Steps:</h4>
                    <ol>
                        <li>Refresh the page and allow camera access</li>
                        <li>Check if another app is using the camera</li>
                        <li>Restart your browser</li>
                        <li>Check browser permissions in settings</li>
                        <li>Use file upload instead</li>
                    </ol>
                </div>
                
                <button onclick="this.parentElement.parentElement.remove()">Close</button>
            </div>
        `;
        document.body.appendChild(dialog);
    }
    
    async enumerateDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            
            console.log('Available cameras:');
            videoDevices.forEach((device, index) => {
                console.log(`${index}: ${device.label || 'Camera ' + index}`);
            });
            
            return videoDevices;
        } catch (error) {
            console.error('Cannot enumerate devices:', error);
            return [];
        }
    }
    
    // Try different camera constraints if first attempt fails
    async tryAlternativeConstraints() {
        const alternatives = [
            // Try basic constraints
            { video: true },
            
            // Try lower resolution
            { video: { width: 640, height: 480 } },
            
            // Try front camera
            { video: { facingMode: 'user' } },
            
            // Try without facing mode
            { video: { width: { max: 1280 }, height: { max: 720 } } }
        ];
        
        for (const constraints of alternatives) {
            try {
                console.log('Trying constraints:', constraints);
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                console.log('Success with alternative constraints');
                return stream;
            } catch (error) {
                console.log('Failed with constraints:', constraints, error.name);
                continue;
            }
        }
        
        throw new Error('All camera constraint alternatives failed');
    }
}
```

#### Issue: Camera stream quality problems
```javascript
// Camera quality optimization:

class CameraOptimizer {
    constructor() {
        this.supportedConstraints = navigator.mediaDevices?.getSupportedConstraints() || {};
    }
    
    getOptimalConstraints() {
        const constraints = { video: {} };
        
        // Resolution constraints
        if (this.supportedConstraints.width && this.supportedConstraints.height) {
            constraints.video.width = { ideal: 1280, max: 1920 };
            constraints.video.height = { ideal: 720, max: 1080 };
        }
        
        // Frame rate
        if (this.supportedConstraints.frameRate) {
            constraints.video.frameRate = { ideal: 30, max: 60 };
        }
        
        // Camera selection
        if (this.supportedConstraints.facingMode) {
            constraints.video.facingMode = 'environment';  // Back camera
        }
        
        return constraints;
    }
    
    async optimizeStream(stream) {
        const videoTrack = stream.getVideoTracks()[0];
        
        if (videoTrack && videoTrack.getCapabilities) {
            const capabilities = videoTrack.getCapabilities();
            console.log('Camera capabilities:', capabilities);
            
            // Apply optimal settings
            const constraints = {};
            
            if (capabilities.width && capabilities.height) {
                constraints.width = Math.min(1280, capabilities.width.max);
                constraints.height = Math.min(720, capabilities.height.max);
            }
            
            if (capabilities.frameRate) {
                constraints.frameRate = Math.min(30, capabilities.frameRate.max);
            }
            
            try {
                await videoTrack.applyConstraints(constraints);
                console.log('Applied optimal constraints:', constraints);
            } catch (error) {
                console.warn('Could not apply optimal constraints:', error);
            }
        }
        
        return stream;
    }
}
```

## Model and AI Issues

### 1. YOLO Model Problems

#### Issue: Model download fails
```python
# Common download issues and solutions:

import os
import requests
from pathlib import Path
import hashlib

class ModelDownloader:
    def __init__(self):
        self.model_urls = {
            'yolo11n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt',
            'yolo11s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11s.pt',
            'yolo11m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11m.pt',
            'yolo11l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11l.pt',
            'yolo11x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11x.pt'
        }
        
    def download_model(self, model_name='yolo11x.pt', max_retries=3):
        """Download model with retry logic and progress tracking"""
        
        if model_name not in self.model_urls:
            raise ValueError(f"Unknown model: {model_name}")
            
        url = self.model_urls[model_name]
        model_path = Path(model_name)
        
        # Check if model already exists
        if model_path.exists():
            print(f"Model {model_name} already exists")
            return str(model_path)
        
        print(f"Downloading {model_name} from {url}")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Show progress
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\rDownloading: {progress:.1f}%", end='', flush=True)
                
                print(f"\n✅ Downloaded {model_name} successfully")
                return str(model_path)
                
            except Exception as e:
                print(f"❌ Download attempt {attempt + 1} failed: {e}")
                if model_path.exists():
                    model_path.unlink()  # Remove partial file
                
                if attempt < max_retries - 1:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise Exception(f"Failed to download {model_name} after {max_retries} attempts")
    
    def verify_model(self, model_path):
        """Verify model file integrity"""
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            
            # Test with dummy input
            import numpy as np
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(dummy_image)
            
            print(f"✅ Model {model_path} verification successful")
            return True
            
        except Exception as e:
            print(f"❌ Model {model_path} verification failed: {e}")
            return False

# Usage:
downloader = ModelDownloader()
try:
    model_path = downloader.download_model('yolo11x.pt')
    if downloader.verify_model(model_path):
        print("Model ready to use")
    else:
        print("Model verification failed, please re-download")
except Exception as e:
    print(f"Model download failed: {e}")
    print("Try using a smaller model: yolo11n.pt or yolo11s.pt")
```

#### Issue: Model inference errors
```python
# Common inference problems and solutions:

class ModelDiagnostics:
    def __init__(self, model):
        self.model = model
    
    def diagnose_inference_error(self, image, error):
        """Diagnose and suggest solutions for inference errors"""
        
        print(f"🔍 Diagnosing inference error: {type(error).__name__}")
        print(f"Error message: {str(error)}")
        
        # Check image properties
        if hasattr(image, 'shape'):
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")
        elif hasattr(image, 'size'):
            print(f"Image size: {image.size}")
            print(f"Image mode: {image.mode}")
        
        # Common error patterns and solutions
        error_solutions = {
            'CUDA out of memory': self._solve_cuda_memory,
            'RuntimeError': self._solve_runtime_error,
            'TypeError': self._solve_type_error,
            'ValueError': self._solve_value_error,
            'AttributeError': self._solve_attribute_error
        }
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Find matching solution
        for pattern, solution_func in error_solutions.items():
            if pattern.lower() in error_message or pattern == error_type:
                return solution_func(error, image)
        
        # Generic solution
        return self._generic_solution(error, image)
    
    def _solve_cuda_memory(self, error, image):
        """Solutions for CUDA memory errors"""
        solutions = [
            "1. Clear GPU cache: torch.cuda.empty_cache()",
            "2. Use smaller image size",
            "3. Switch to CPU: model.to('cpu')",
            "4. Use smaller YOLO model (yolo11n.pt)",
            "5. Reduce batch size or max_det parameter",
            "6. Close other GPU applications"
        ]
        
        # Implement automatic fixes
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🔧 Cleared CUDA cache")
        
        return solutions
    
    def _solve_runtime_error(self, error, image):
        """Solutions for runtime errors"""
        solutions = [
            "1. Check image format and dimensions",
            "2. Ensure image is valid (not corrupted)",
            "3. Try converting image: image.convert('RGB')",
            "4. Check model is properly loaded",
            "5. Verify model file integrity"
        ]
        
        # Try automatic image conversion
        try:
            if hasattr(image, 'convert'):
                fixed_image = image.convert('RGB')
                print("🔧 Converted image to RGB")
                return solutions, fixed_image
        except:
            pass
        
        return solutions
    
    def _solve_type_error(self, error, image):
        """Solutions for type errors"""
        solutions = [
            "1. Convert image to numpy array: np.array(image)",
            "2. Convert to PIL Image: Image.fromarray(image)",
            "3. Check image data type: image.dtype",
            "4. Ensure proper image format"
        ]
        
        # Try automatic type conversion
        try:
            import numpy as np
            from PIL import Image as PILImage
            
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                    print("🔧 Converted array to uint8")
            elif hasattr(image, 'convert'):
                image_array = np.array(image)
                print("🔧 Converted PIL to numpy")
                return solutions, image_array
                
        except Exception as conv_error:
            print(f"Auto-conversion failed: {conv_error}")
        
        return solutions
    
    def _generic_solution(self, error, image):
        """Generic troubleshooting steps"""
        return [
            "1. Check error details above",
            "2. Verify image is valid and properly formatted", 
            "3. Try with a different image",
            "4. Check model file integrity",
            "5. Update ultralytics package: pip install --upgrade ultralytics",
            "6. Restart Python session",
            "7. Check system resources (memory, disk space)"
        ]

# Usage in detection service:
def detect_objects_with_diagnostics(self, image, **params):
    try:
        results = self.model.predict(source=image, **params)
        return results
    except Exception as e:
        diagnostics = ModelDiagnostics(self.model)
        solutions = diagnostics.diagnose_inference_error(image, e)
        
        print("🚨 Inference failed. Suggested solutions:")
        for solution in solutions:
            print(f"   {solution}")
        
        # Try with fallback parameters
        try:
            print("🔧 Trying with fallback parameters...")
            fallback_params = {
                'imgsz': 320,      # Smaller size
                'conf': 0.5,       # Higher confidence
                'max_det': 50,     # Fewer detections
                'half': False      # Disable FP16
            }
            results = self.model.predict(source=image, **fallback_params)
            print("✅ Inference succeeded with fallback parameters")
            return results
            
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            raise e  # Re-raise original error
```

## Browser and Frontend Issues

### 1. JavaScript Errors

#### Issue: WebSocket connection problems
```javascript
// Robust WebSocket implementation with error handling:

class RobustWebSocket {
    constructor(url, protocols = []) {
        this.url = url;
        this.protocols = protocols;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectInterval = 1000;
        this.maxReconnectInterval = 30000;
        this.reconnectDecay = 1.5;
        this.isIntentionallyClosed = false;
        
        this.connect();
    }
    
    connect() {
        try {
            this.ws = new WebSocket(this.url, this.protocols);
            this.setupEventHandlers();
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.scheduleReconnect();
        }
    }
    
    setupEventHandlers() {
        this.ws.onopen = (event) => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            this.reconnectInterval = 1000;
            this.onopen && this.onopen(event);
        };
        
        this.ws.onmessage = (event) => {
            this.onmessage && this.onmessage(event);
        };
        
        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            
            if (!this.isIntentionallyClosed) {
                this.scheduleReconnect();
            }
            
            this.onclose && this.onclose(event);
        };
        
        this.ws.onerror = (event) => {
            console.error('WebSocket error:', event);
            this.onerror && this.onerror(event);
        };
    }
    
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            return;
        }
        
        this.reconnectAttempts++;
        
        console.log(`Reconnecting in ${this.reconnectInterval}ms (attempt ${this.reconnectAttempts})`);
        
        setTimeout(() => {
            this.connect();
        }, this.reconnectInterval);
        
        this.reconnectInterval = Math.min(
            this.maxReconnectInterval,
            this.reconnectInterval * this.reconnectDecay
        );
    }
    
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(data);
                return true;
            } catch (error) {
                console.error('Failed to send message:', error);
                return false;
            }
        } else {
            console.warn('WebSocket not connected, cannot send message');
            return false;
        }
    }
    
    close() {
        this.isIntentionallyClosed = true;
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Usage:
const ws = new RobustWebSocket('ws://localhost:3000/ws/video/client123');

ws.onopen = () => {
    console.log('Connected to detection service');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleDetectionResult(data);
};

ws.onerror = (error) => {
    showUserMessage('Connection error. Trying to reconnect...', 'warning');
};
```

#### Issue: File upload problems
```javascript
// Comprehensive file upload handling:

class FileUploadManager {
    constructor() {
        this.maxFileSize = 50 * 1024 * 1024; // 50MB
        this.allowedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
        this.compressionQuality = 0.8;
    }
    
    async handleFile(file) {
        try {
            // Validate file
            this.validateFile(file);
            
            // Compress if needed
            const processedFile = await this.processFile(file);
            
            // Upload with progress
            const result = await this.uploadWithProgress(processedFile);
            
            return result;
            
        } catch (error) {
            this.handleUploadError(error, file);
            throw error;
        }
    }
    
    validateFile(file) {
        // Check file type
        if (!this.allowedTypes.includes(file.type)) {
            throw new Error(`Unsupported file type: ${file.type}. Allowed: ${this.allowedTypes.join(', ')}`);
        }
        
        // Check file size
        if (file.size > this.maxFileSize) {
            throw new Error(`File too large: ${(file.size / 1024 / 1024).toFixed(1)}MB. Maximum: ${this.maxFileSize / 1024 / 1024}MB`);
        }
        
        // Check if file is actually an image
        if (!file.type.startsWith('image/')) {
            throw new Error('File does not appear to be a valid image');
        }
    }
    
    async processFile(file) {
        // Check if compression is needed
        const needsCompression = file.size > 5 * 1024 * 1024; // 5MB
        
        if (!needsCompression) {
            return file;
        }
        
        console.log('Compressing large image...');
        return await this.compressImage(file);
    }
    
    async compressImage(file) {
        return new Promise((resolve, reject) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = () => {
                // Calculate new dimensions (max 1920x1080)
                const maxWidth = 1920;
                const maxHeight = 1080;
                
                let { width, height } = img;
                
                if (width > maxWidth || height > maxHeight) {
                    const ratio = Math.min(maxWidth / width, maxHeight / height);
                    width = Math.round(width * ratio);
                    height = Math.round(height * ratio);
                }
                
                canvas.width = width;
                canvas.height = height;
                
                // Draw and compress
                ctx.drawImage(img, 0, 0, width, height);
                
                canvas.toBlob(
                    (blob) => {
                        if (blob) {
                            console.log(`Compressed from ${(file.size / 1024 / 1024).toFixed(1)}MB to ${(blob.size / 1024 / 1024).toFixed(1)}MB`);
                            resolve(blob);
                        } else {
                            reject(new Error('Image compression failed'));
                        }
                    },
                    'image/jpeg',
                    this.compressionQuality
                );
            };
            
            img.onerror = () => {
                reject(new Error('Invalid image file'));
            };
            
            img.src = URL.createObjectURL(file);
        });
    }
    
    async uploadWithProgress(file) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            formData.append('image', file);
            formData.append('session_id', this.getSessionId());
            
            const xhr = new XMLHttpRequest();
            
            // Progress tracking
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    this.updateProgressBar(percentComplete);
                }
            });
            
            // Success handler
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        const result = JSON.parse(xhr.responseText);
                        resolve(result);
                    } catch (error) {
                        reject(new Error('Invalid response from server'));
                    }
                } else {
                    reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`));
                }
            });
            
            // Error handler
            xhr.addEventListener('error', () => {
                reject(new Error('Network error during upload'));
            });
            
            // Timeout handler
            xhr.addEventListener('timeout', () => {
                reject(new Error('Upload timeout'));
            });
            
            xhr.timeout = 120000; // 2 minute timeout
            xhr.open('POST', '/detect/1');
            xhr.send(formData);
        });
    }
    
    handleUploadError(error, file) {
        console.error('Upload error:', error);
        
        const errorMessages = {
            'file-too-large': 'The file is too large. Please choose a smaller image or the app will automatically compress it.',
            'invalid-type': 'This file type is not supported. Please use JPEG, PNG, or WebP images.',
            'network-error': 'Network error occurred. Please check your connection and try again.',
            'server-error': 'Server error occurred. Please try again later.',
            'timeout': 'Upload timed out. Please try with a smaller file or check your connection.'
        };
        
        let errorType = 'unknown';
        const errorMsg = error.message.toLowerCase();
        
        if (errorMsg.includes('file too large')) {
            errorType = 'file-too-large';
        } else if (errorMsg.includes('unsupported') || errorMsg.includes('invalid image')) {
            errorType = 'invalid-type';
        } else if (errorMsg.includes('network')) {
            errorType = 'network-error';
        } else if (errorMsg.includes('timeout')) {
            errorType = 'timeout';
        } else if (errorMsg.includes('500') || errorMsg.includes('server')) {
            errorType = 'server-error';
        }
        
        const userMessage = errorMessages[errorType] || `Upload failed: ${error.message}`;
        this.showUserError(userMessage, errorType);
    }
    
    updateProgressBar(percent) {
        const progressBar = document.getElementById('upload-progress');
        if (progressBar) {
            progressBar.style.width = `${percent}%`;
            progressBar.textContent = `${Math.round(percent)}%`;
        }
    }
    
    showUserError(message, type) {
        const errorDiv = document.createElement('div');
        errorDiv.className = `error-message ${type}`;
        errorDiv.innerHTML = `
            <div class="error-content">
                <strong>Upload Error</strong>
                <p>${message}</p>
                <button onclick="this.parentElement.parentElement.remove()">Dismiss</button>
            </div>
        `;
        
        document.body.appendChild(errorDiv);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 10000);
    }
    
    getSessionId() {
        return localStorage.getItem('od_session_id') || '';
    }
}

// Usage:
const uploadManager = new FileUploadManager();

document.getElementById('file-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    try {
        const result = await uploadManager.handleFile(file);
        console.log('Upload successful:', result);
        // Handle successful upload
    } catch (error) {
        console.error('Upload failed:', error);
        // Error already shown to user by uploadManager
    }
});
```

## System-Specific Problems

### 1. Windows-Specific Issues

#### Issue: Path and permission problems
```batch
REM Common Windows-specific fixes:

REM 1. Long path support
REM Enable in Windows 10/11:
REM Computer Configuration > Administrative Templates > System > Filesystem
REM Enable "Enable Win32 long paths"

REM 2. PowerShell execution policy
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"

REM 3. Python PATH issues
REM Add Python to PATH manually:
setx PATH "%PATH%;C:\Python310;C:\Python310\Scripts"

REM 4. Windows Defender exclusion
REM Add project folder to Windows Defender exclusions:
REM Windows Security > Virus & threat protection > Exclusions
```

```python
# Windows-specific Python fixes:

import os
import sys
import platform

def fix_windows_issues():
    """Apply Windows-specific fixes"""
    
    if platform.system() != 'Windows':
        return
    
    # 1. Fix path separator issues
    def fix_path(path_str):
        return path_str.replace('/', os.sep)
    
    # 2. Handle long paths
    def enable_long_paths():
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\FileSystem",
                0,
                winreg.KEY_SET_VALUE
            )
            winreg.SetValueEx(key, "LongPathsEnabled", 0, winreg.REG_DWORD, 1)
            winreg.CloseKey(key)
            print("✅ Long paths enabled")
        except Exception as e:
            print(f"⚠️  Could not enable long paths: {e}")
    
    # 3. Set proper file permissions
    def set_file_permissions(file_path):
        try:
            os.chmod(file_path, 0o755)
        except Exception as e:
            print(f"Could not set permissions for {file_path}: {e}")
    
    # 4. Handle Windows file locking
    def safe_file_operation(file_path, operation):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                return operation()
            except PermissionError:
                if attempt < max_attempts - 1:
                    time.sleep(0.5)  # Wait and retry
                else:
                    raise
    
    print("Applied Windows-specific fixes")
```

### 2. macOS-Specific Issues

#### Issue: Apple Silicon compatibility
```bash
# macOS Apple Silicon (M1/M2) specific fixes:

# 1. Install Rosetta 2 if needed
softwareupdate --install-rosetta --agree-to-license

# 2. Use native Python
# Install via Homebrew:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.10

# 3. PyTorch with Metal Performance Shaders
pip install torch torchvision torchaudio

# Verify MPS support:
python -c "import torch; print(torch.backends.mps.is_available())"

# 4. Fix camera permissions
# System Preferences > Security & Privacy > Privacy > Camera
# Add Terminal.app or your Python IDE
```

```python
# macOS-specific optimizations:

import platform
import subprocess

def optimize_for_macos():
    """Apply macOS-specific optimizations"""
    
    if platform.system() != 'Darwin':
        return
    
    # Check if Apple Silicon
    is_apple_silicon = platform.processor() == 'arm'
    
    if is_apple_silicon:
        print("Detected Apple Silicon Mac")
        
        # Use Metal Performance Shaders if available
        try:
            import torch
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                print("✅ Using Metal Performance Shaders")
                return device
        except:
            pass
    
    # Fallback optimizations
    try:
        # Increase file descriptor limit
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (10240, 10240))
        print("✅ Increased file descriptor limit")
    except:
        pass
    
    return torch.device("cpu")

# Use in model loading:
device = optimize_for_macos()
model = YOLO('yolo11x.pt')
if device.type != 'cpu':
    model.to(device)
```

### 3. Linux-Specific Issues

#### Issue: Missing system dependencies
```bash
# Common Linux dependency issues:

# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# CentOS/RHEL:
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    python3-pip \
    python3-devel \
    mesa-libGL \
    glib2 \
    libSM \
    libXext \
    libXrender \
    libgomp \
    gtk3 \
    ffmpeg-devel

# Arch Linux:
sudo pacman -S \
    python-pip \
    base-devel \
    mesa \
    glib2 \
    libsm \
    libxext \
    libxrender \
    gomp \
    gtk3 \
    ffmpeg
```

## Advanced Debugging

### 1. Comprehensive Logging Setup

```python
# Advanced logging configuration:

import logging
import logging.handlers
import sys
from pathlib import Path

def setup_comprehensive_logging(log_level=logging.INFO):
    """Setup comprehensive logging for debugging"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Separate error log
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Performance log
    perf_logger = logging.getLogger('performance')
    perf_handler = logging.handlers.RotatingFileHandler(
        log_dir / "performance.log",
        maxBytes=10*1024*1024,
        backupCount=3
    )
    perf_handler.setFormatter(detailed_formatter)
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    return root_logger

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            logging.error(f"Function {func.__name__} failed: {e}")
            raise
        finally:
            end_time = time.time()
            end_memory = get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            perf_logger = logging.getLogger('performance')
            perf_logger.info(
                f"{func.__name__} - Duration: {duration:.3f}s, "
                f"Memory: {memory_delta:+.1f}MB, Success: {success}"
            )
        
        return result
    return wrapper

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0

# Usage:
setup_comprehensive_logging()

@monitor_performance
def detect_objects(image):
    # Your detection code
    pass
```

### 2. System Diagnostics Tool

```python
# Comprehensive system diagnostics:

import psutil
import GPUtil
import platform
import socket
import subprocess
from pathlib import Path

class SystemDiagnostics:
    """Comprehensive system diagnostics for troubleshooting"""
    
    def __init__(self):
        self.report = {}
    
    def run_full_diagnostics(self):
        """Run complete system diagnostics"""
        
        print("🔍 Running comprehensive system diagnostics...")
        
        self.report['system'] = self.check_system_info()
        self.report['python'] = self.check_python_environment()
        self.report['hardware'] = self.check_hardware()
        self.report['network'] = self.check_network()
        self.report['dependencies'] = self.check_dependencies()
        self.report['model'] = self.check_model_status()
        self.report['performance'] = self.check_performance()
        
        return self.generate_report()
    
    def check_system_info(self):
        """Check system information"""
        return {
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'hostname': socket.gethostname()
        }
    
    def check_python_environment(self):
        """Check Python environment"""
        import sys
        
        return {
            'executable': sys.executable,
            'version': sys.version,
            'path': sys.path[:3],  # First 3 entries
            'modules_count': len(sys.modules),
            'virtual_env': 'VIRTUAL_ENV' in os.environ
        }
    
    def check_hardware(self):
        """Check hardware resources"""
        # CPU info
        cpu_info = {
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'usage_percent': psutil.cpu_percent(interval=1)
        }
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = {
            'total_gb': memory.total / 1024**3,
            'available_gb': memory.available / 1024**3,
            'used_percent': memory.percent
        }
        
        # Disk info
        disk = psutil.disk_usage('.')
        disk_info = {
            'total_gb': disk.total / 1024**3,
            'free_gb': disk.free / 1024**3,
            'used_percent': (disk.used / disk.total) * 100
        }
        
        # GPU info
        gpu_info = self.check_gpu_info()
        
        return {
            'cpu': cpu_info,
            'memory': memory_info,
            'disk': disk_info,
            'gpu': gpu_info
        }
    
    def check_gpu_info(self):
        """Check GPU information"""
        gpu_info = {'available': False, 'details': []}
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['cuda_version'] = torch.version.cuda
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info['details'].append({
                        'name': props.name,
                        'memory_total': props.total_memory / 1024**3,
                        'memory_free': (props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
        except ImportError:
            gpu_info['error'] = 'PyTorch not installed'
        except Exception as e:
            gpu_info['error'] = str(e)
        
        return gpu_info
    
    def check_network(self):
        """Check network configuration"""
        network_info = {
            'interfaces': [],
            'connectivity': {}
        }
        
        # Network interfaces
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    network_info['interfaces'].append({
                        'name': interface,
                        'ip': addr.address,
                        'netmask': addr.netmask
                    })
        
        # Test connectivity
        test_urls = [
            ('google.com', 80),
            ('github.com', 443),
            ('localhost', 3000)
        ]
        
        for host, port in test_urls:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                network_info['connectivity'][f"{host}:{port}"] = result == 0
            except Exception as e:
                network_info['connectivity'][f"{host}:{port}"] = str(e)
        
        return network_info
    
    def check_dependencies(self):
        """Check Python dependencies"""
        import pkg_resources
        
        required_packages = [
            'fastapi', 'uvicorn', 'ultralytics', 'opencv-python',
            'pillow', 'numpy', 'torch', 'torchvision', 'jinja2'
        ]
        
        installed_packages = {pkg.project_name: pkg.version for pkg in pkg_resources.working_set}
        
        dependency_info = {
            'total_installed': len(installed_packages),
            'required_status': {},
            'versions': {}
        }
        
        for package in required_packages:
            if package in installed_packages:
                dependency_info['required_status'][package] = 'installed'
                dependency_info['versions'][package] = installed_packages[package]
            else:
                dependency_info['required_status'][package] = 'missing'
        
        return dependency_info
    
    def check_model_status(self):
        """Check YOLO model status"""
        model_info = {'files': {}, 'loading': {}}
        
        # Check model files
        model_files = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']
        
        for model_file in model_files:
            model_path = Path(model_file)
            if model_path.exists():
                size_mb = model_path.stat().st_size / 1024**2
                model_info['files'][model_file] = {
                    'exists': True,
                    'size_mb': size_mb
                }
            else:
                model_info['files'][model_file] = {'exists': False}
        
        # Test model loading
        try:
            from ultralytics import YOLO
            
            # Try to load smallest available model
            for model_file in model_files:
                if model_info['files'][model_file]['exists']:
                    try:
                        start_time = time.time()
                        model = YOLO(model_file)
                        load_time = time.time() - start_time
                        
                        model_info['loading'][model_file] = {
                            'success': True,
                            'load_time': load_time,
                            'classes_count': len(model.names)
                        }
                        break  # Only test one model
                    except Exception as e:
                        model_info['loading'][model_file] = {
                            'success': False,
                            'error': str(e)
                        }
        except ImportError as e:
            model_info['loading']['error'] = f"Cannot import ultralytics: {e}"
        
        return model_info
    
    def check_performance(self):
        """Run performance benchmarks"""
        performance_info = {}
        
        # CPU benchmark
        import time
        start_time = time.time()
        
        # Simple CPU test
        total = 0
        for i in range(1000000):
            total += i * i
        
        cpu_test_time = time.time() - start_time
        performance_info['cpu_benchmark_ms'] = cpu_test_time * 1000
        
        # Memory allocation test
        start_time = time.time()
        test_array = [0] * 10000000  # 10M integers
        del test_array
        memory_test_time = time.time() - start_time
        performance_info['memory_benchmark_ms'] = memory_test_time * 1000
        
        # Disk I/O test
        start_time = time.time()
        test_file = Path('performance_test.tmp')
        try:
            with open(test_file, 'w') as f:
                f.write('x' * 1000000)  # 1MB
            test_file.unlink()
            io_test_time = time.time() - start_time
            performance_info['io_benchmark_ms'] = io_test_time * 1000
        except Exception as e:
            performance_info['io_benchmark_error'] = str(e)
        
        return performance_info
    
    def generate_report(self):
        """Generate formatted diagnostics report"""
        
        print("\n" + "="*80)
        print("SYSTEM DIAGNOSTICS REPORT")
        print("="*80)
        
        # System Info
        sys_info = self.report['system']
        print(f"\n📊 SYSTEM INFO:")
        print(f"   OS: {sys_info['os']} {sys_info['os_version']}")
        print(f"   Architecture: {sys_info['architecture']}")
        print(f"   Python: {sys_info['python_version']}")
        
        # Hardware
        hw_info = self.report['hardware']
        print(f"\n💻 HARDWARE:")
        print(f"   CPU: {hw_info['cpu']['cores_physical']} cores ({hw_info['cpu']['cores_logical']} logical)")
        print(f"   Memory: {hw_info['memory']['available_gb']:.1f}GB available / {hw_info['memory']['total_gb']:.1f}GB total")
        print(f"   Disk: {hw_info['disk']['free_gb']:.1f}GB free / {hw_info['disk']['total_gb']:.1f}GB total")
        
        if hw_info['gpu']['available']:
            for gpu in hw_info['gpu']['details']:
                print(f"   GPU: {gpu['name']} ({gpu['memory_total']:.1f}GB)")
        else:
            print(f"   GPU: Not available")
        
        # Dependencies
        deps_info = self.report['dependencies']
        print(f"\n📦 DEPENDENCIES:")
        missing_deps = [pkg for pkg, status in deps_info['required_status'].items() if status == 'missing']
        if missing_deps:
            print(f"   ❌ Missing: {', '.join(missing_deps)}")
        else:
            print(f"   ✅ All required packages installed")
        
        # Model Status
        model_info = self.report['model']
        print(f"\n🤖 MODEL STATUS:")
        available_models = [name for name, info in model_info['files'].items() if info['exists']]
        if available_models:
            print(f"   ✅ Available models: {', '.join(available_models)}")
        else:
            print(f"   ❌ No YOLO models found")
        
        # Performance
        perf_info = self.report['performance']
        print(f"\n⚡ PERFORMANCE:")
        print(f"   CPU Benchmark: {perf_info['cpu_benchmark_ms']:.1f}ms")
        print(f"   Memory Benchmark: {perf_info['memory_benchmark_ms']:.1f}ms")
        if 'io_benchmark_ms' in perf_info:
            print(f"   I/O Benchmark: {perf_info['io_benchmark_ms']:.1f}ms")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if hw_info['memory']['available_gb'] < 4:
            print("   ⚠️  Low memory - consider closing other applications")
        
        if not hw_info['gpu']['available']:
            print("   ⚠️  No GPU detected - processing will be slower")
        
        if missing_deps:
            print("   ❌ Install missing dependencies: pip install " + " ".join(missing_deps))
        
        if not available_models:
            print("   ❌ Download YOLO model: python -c \"from ultralytics import YOLO; YOLO('yolo11x.pt')\"")
        
        print("\n" + "="*80)
        
        return self.report

# Usage:
diagnostics = SystemDiagnostics()
report = diagnostics.run_full_diagnostics()
```

This comprehensive troubleshooting guide covers the most common issues users might encounter and provides practical solutions with code examples. Each section includes both diagnostic tools and fix implementations to help users resolve problems quickly and efficiently.
