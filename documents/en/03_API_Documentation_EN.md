# API Documentation

## Table of Contents
1. [API Overview](#api-overview)
2. [Authentication & Sessions](#authentication--sessions)
3. [HTTP Endpoints](#http-endpoints)
4. [WebSocket API](#websocket-api)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Rate Limiting & Performance](#rate-limiting--performance)
8. [Code Examples](#code-examples)

## API Overview

The Object Detection API is built on FastAPI and provides both RESTful HTTP endpoints and WebSocket connections for real-time processing. The API supports image comparison workflows and live video detection with comprehensive session management.

### Base Information
```yaml
Base URL: http://localhost:3000
Protocol: HTTP/1.1, WebSocket
Content-Type: application/json (for JSON endpoints)
File Upload: multipart/form-data
Real-time: WebSocket (ws://localhost:3000/ws/video/{client_id})
```

### API Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Client Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ File Upload │  │    JSON     │  │  WebSocket  │    │
│  │   (Form)    │  │  Requests   │  │   Client    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
                           │
                    HTTP/WebSocket
                           │
┌─────────────────────────────────────────────────────────┐
│                 FastAPI Server                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Session   │  │    YOLO     │  │  WebSocket  │    │
│  │  Endpoints  │  │ Processing  │  │   Handler   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Authentication & Sessions

### Session Management
The API uses UUID-based session management for isolating user data and enabling multi-user support.

#### Create Session
```http
POST /create-session
```

**Response:**
```json
{
  "success": true,
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "New session created successfully"
}
```

#### Session Storage
Each session stores:
- `image1`: First image detection results
- `image2`: Second image detection results  
- `created_at`: Session creation timestamp
- Auto-cleanup after 24 hours

### Session Validation
All processing endpoints require a valid `session_id`. Sessions are automatically validated and cleaned up.

## HTTP Endpoints

### 1. Page Rendering Endpoints

#### Landing Page
```http
GET /
```
Returns the main application interface with mode selection.

**Response:** HTML page with workflow selection

#### Image Comparison Page  
```http
GET /image-comparison
```
**Response:** HTML page for image comparison workflow

#### Video Detection Page
```http
GET /video-detection  
```
**Response:** HTML page for real-time video detection

#### Tracking Dashboard
```http
GET /tracking-dashboard
```
**Response:** HTML page for object tracking dashboard

#### Network Access Guide
```http
GET /network-access
```
**Response:** HTML page with LAN sharing instructions and IP addresses

### 2. Detection Endpoints

#### Upload Image Detection
```http
POST /detect/{image_number}
Content-Type: multipart/form-data
```

**Parameters:**
- `image_number`: 1 or 2 (path parameter)
- `image`: Image file (form data)
- `conf_threshold`: Float (0.05-0.95, default: 0.50)
- `iou_threshold`: Float (0.1-0.9, default: 0.15) 
- `max_detection`: Integer (10-1000, default: 500)
- `session_id`: String (required)

**Example Request:**
```bash
curl -X POST "http://localhost:3000/detect/1" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@image1.jpg" \
  -F "conf_threshold=0.50" \
  -F "iou_threshold=0.15" \
  -F "max_detection=500" \
  -F "session_id=123e4567-e89b-12d3-a456-426614174000"
```

**Response:**
```json
{
  "success": true,
  "report": {
    "image_name": "Image 1: image1.jpg",
    "total_objects": 15,
    "unique_classes": 8,
    "class_counts": {
      "person": 3,
      "chair": 4,
      "table": 1,
      "laptop": 2,
      "book": 3,
      "cup": 1,
      "phone": 1
    },
    "detections": [
      {
        "class_name": "person",
        "confidence": 0.95,
        "bbox": [100.5, 200.3, 300.8, 450.2],
        "area": 50062.5
      }
    ]
  },
  "original_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "parameters": {
    "conf_threshold": 0.50,
    "iou_threshold": 0.15,
    "max_detection": 500
  },
  "ready_for_comparison": false,
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### Base64 Image Detection
```http
POST /detect-base64/{image_number}
Content-Type: application/json
```

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "conf_threshold": 0.50,
  "iou_threshold": 0.15,
  "max_detection": 500,
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Response:** Same as upload detection endpoint

### 3. Analysis Endpoints

#### Compare Images
```http
POST /compare
Content-Type: application/json
```

**Request Body:**
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Response:**
```json
{
  "success": true,
  "comparison": {
    "total_objects": {
      "image1": 15,
      "image2": 18, 
      "change": 3
    },
    "class_changes": {
      "person": {
        "image1": 3,
        "image2": 4,
        "change": 1
      },
      "chair": {
        "image1": 4,
        "image2": 3,
        "change": -1
      }
    },
    "new_classes": ["bottle", "mouse"],
    "removed_classes": ["book"]
  },
  "image1": {...}, 
  "image2": {...},
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### Adjust Parameters
```http
POST /adjust-parameters
Content-Type: application/json
```

Re-analyze both images with new detection parameters.

**Request Body:**
```json
{
  "conf_threshold": 0.30,
  "iou_threshold": 0.25,
  "max_detection": 1000,
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Response:**
```json
{
  "success": true,
  "adjusted_results": {
    "image1": {...},
    "image2": {...}
  },
  "comparison": {...},
  "parameters_used": {
    "conf_threshold": 0.30,
    "iou_threshold": 0.25,
    "max_detection": 1000
  },
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### Save Results
```http
POST /save-results
Content-Type: application/json
```

**Request Body:**
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Response:**
```json
{
  "success": true,
  "report_text": "OBJECT DETECTION COMPARISON REPORT\n=====================================\n\nIMAGE 1 REPORT:\nTotal Objects: 15\n...",
  "full_results": {...},
  "comparison": {...},
  "download_ready": true,
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

### 4. Utility Endpoints

#### Reset Workflow
```http
POST /reset-workflow
Content-Type: application/json
```

**Request Body:**
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Workflow reset successfully",
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### Get Status
```http
GET /status/{session_id}
```

**Response:**
```json
{
  "image1_processed": true,
  "image2_processed": false,
  "ready_for_comparison": false,
  "next_step": "image2",
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Clean Up Sessions
```http
POST /cleanup-sessions
```

**Response:**
```json
{
  "success": true,
  "message": "Cleaned up 5 old sessions",
  "removed_sessions": 5
}
```

## WebSocket API

### Connection
```
ws://localhost:3000/ws/video/{client_id}
```

The WebSocket API enables real-time video processing for live object detection.

#### Connection Parameters
- `client_id`: Unique identifier for the client session

#### Message Format
All WebSocket messages use JSON format:

```json
{
  "action": "message_type",
  "data": {...}
}
```

### Client Messages

#### Process Frame
```json
{
  "action": "process_frame",
  "frame": "data:image/jpeg;base64,/9j/4AAQ...",
  "conf_threshold": 0.50,
  "iou_threshold": 0.15,
  "max_detection": 100
}
```

#### Keep-Alive Ping
```json
{
  "action": "ping"
}
```

### Server Messages

#### Detection Result
```json
{
  "success": true,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.92,
      "bbox": [120.5, 180.3, 280.8, 420.2],
      "area": 38440.0
    }
  ],
  "class_counts": {
    "person": 2,
    "chair": 1,
    "table": 1
  },
  "total_objects": 4,
  "timestamp": 1698765432.123
}
```

#### Pong Response
```json
{
  "type": "pong",
  "timestamp": 1698765432.123
}
```

#### Error Response
```json
{
  "success": false,
  "error": "Detection processing failed",
  "timestamp": 1698765432.123
}
```

### WebSocket Connection Management

#### Connection Lifecycle
```python
# Connection establishment
async def connect(websocket: WebSocket, client_id: str):
    await websocket.accept()
    # Store connection and initialize processing state

# Message handling
async def handle_message(message: dict):
    if message["action"] == "process_frame":
        # Process frame and send result
    elif message["action"] == "ping":
        # Send pong response

# Disconnection cleanup
def disconnect(client_id: str):
    # Remove connection and clean up resources
```

## Data Models

### Detection Object
```python
{
  "class_name": str,        # Object class (e.g., "person", "car")
  "confidence": float,      # Detection confidence (0.0-1.0)
  "bbox": [float, float, float, float],  # [x1, y1, x2, y2] coordinates
  "area": float            # Bounding box area in pixels
}
```

### Detection Report
```python
{
  "image_name": str,        # Descriptive name for the image
  "total_objects": int,     # Total number of detected objects
  "unique_classes": int,    # Number of different object classes
  "class_counts": Dict[str, int],  # Count per object class
  "detections": List[Detection]    # List of individual detections
}
```

### Comparison Result
```python
{
  "total_objects": {
    "image1": int,
    "image2": int,
    "change": int
  },
  "class_changes": Dict[str, {
    "image1": int,
    "image2": int,
    "change": int
  }],
  "new_classes": List[str],      # Classes found only in image2
  "removed_classes": List[str]   # Classes found only in image1
}
```

### Session Data Structure
```python
{
  "session_id": str,
  "created_at": float,      # Unix timestamp
  "image1": {
    "report": DetectionReport,
    "original_image": str,   # Base64 encoded image
    "annotated_image": str,  # Base64 encoded annotated image
    "parameters": Dict,      # Detection parameters used
    "detections": List[Detection],
    "class_counts": Dict[str, int]
  },
  "image2": {...}           # Same structure as image1
}
```

## Error Handling

### HTTP Error Responses

#### 400 Bad Request
```json
{
  "detail": "Invalid parameter: conf_threshold must be between 0.05 and 0.95"
}
```

#### 404 Not Found
```json
{
  "detail": "Session not found. Please create a new session."
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Detection failed: YOLO model not loaded"
}
```

### Common Error Scenarios

#### Invalid Session ID
```http
Status: 404 Not Found
{
  "detail": "Session not found. Please create a new session."
}
```

#### Missing Required Parameters
```http
Status: 400 Bad Request
{
  "detail": "Session ID required"
}
```

#### Invalid File Type
```http
Status: 400 Bad Request  
{
  "detail": "Please select a valid image file"
}
```

#### Model Processing Error
```http
Status: 500 Internal Server Error
{
  "detail": "Detection failed: Model inference error"
}
```

### WebSocket Error Handling

#### Connection Errors
- **Connection Refused**: Check if server is running and port is accessible
- **WebSocket Upgrade Failed**: Ensure proper WebSocket headers
- **Timeout**: Implement reconnection logic with exponential backoff

#### Processing Errors
```json
{
  "success": false,
  "error": "Frame processing failed: Invalid image format",
  "timestamp": 1698765432.123
}
```

## Rate Limiting & Performance

### Request Limits
```yaml
Session Creation: 10 per minute per IP
File Upload: 100MB max file size
WebSocket Connections: 5 concurrent per IP
Session Storage: 50MB max per session
Processing Queue: 10 concurrent detections
```

### Performance Considerations

#### Optimization Parameters
```python
# For faster processing, adjust these parameters:
PERFORMANCE_CONFIG = {
  "conf_threshold": 0.5,     # Higher = fewer detections, faster
  "max_detection": 100,      # Lower = less processing time  
  "image_resize": 640,       # Smaller = faster inference
  "batch_size": 1,           # Keep at 1 for real-time
  "half_precision": True     # Use FP16 if GPU supports
}
```

#### Memory Management
- Sessions auto-cleanup after 24 hours
- Image data stored in memory only (no disk writes)
- Periodic cleanup endpoint available
- Real-time processing uses frame buffering

## Code Examples

### Python Client Example

#### Image Detection
```python
import requests
import base64

# Create session
session_resp = requests.post("http://localhost:3000/create-session")
session_id = session_resp.json()["session_id"]

# Upload and detect image
with open("image1.jpg", "rb") as f:
    files = {"image": f}
    data = {
        "conf_threshold": 0.5,
        "iou_threshold": 0.15,
        "max_detection": 500,
        "session_id": session_id
    }
    
    response = requests.post(
        "http://localhost:3000/detect/1",
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"Detected {result['report']['total_objects']} objects")
```

#### Base64 Image Detection
```python
import requests
import base64

# Convert image to base64
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()
    image_b64 = f"data:image/jpeg;base64,{image_data}"

# Create session
session_resp = requests.post("http://localhost:3000/create-session")
session_id = session_resp.json()["session_id"]

# Detect objects
payload = {
    "image": image_b64,
    "conf_threshold": 0.5,
    "iou_threshold": 0.15, 
    "max_detection": 500,
    "session_id": session_id
}

response = requests.post(
    "http://localhost:3000/detect-base64/1",
    json=payload
)

result = response.json()
for detection in result["detections"]:
    print(f"Found {detection['class_name']} with {detection['confidence']:.2f} confidence")
```

### JavaScript WebSocket Example

#### Real-time Video Processing
```javascript
// Connect to WebSocket
const clientId = 'client_' + Math.random().toString(36).substr(2, 9);
const ws = new WebSocket(`ws://localhost:3000/ws/video/${clientId}`);

// Handle connection
ws.onopen = function() {
    console.log('Connected to detection service');
};

// Handle messages
ws.onmessage = function(event) {
    const result = JSON.parse(event.data);
    
    if (result.success) {
        console.log(`Detected ${result.total_objects} objects`);
        // Update UI with detection results
        updateDetectionDisplay(result.detections);
    } else {
        console.error('Detection error:', result.error);
    }
};

// Send frame for processing
function processFrame(canvas) {
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    
    const message = {
        action: 'process_frame',
        frame: frameData,
        conf_threshold: 0.5,
        iou_threshold: 0.15,
        max_detection: 100
    };
    
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
    }
}

// Keep connection alive
setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'ping' }));
    }
}, 30000); // Ping every 30 seconds
```

### curl Examples

#### Health Check
```bash
curl -X GET "http://localhost:3000/health"
```

#### Create Session
```bash
curl -X POST "http://localhost:3000/create-session"
```

#### Image Detection with File Upload
```bash
curl -X POST "http://localhost:3000/detect/1" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@test_image.jpg" \
  -F "conf_threshold=0.5" \
  -F "iou_threshold=0.15" \
  -F "max_detection=500" \
  -F "session_id=your-session-id-here"
```

#### Compare Images
```bash
curl -X POST "http://localhost:3000/compare" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id-here"}'
```

#### Reset Session
```bash
curl -X POST "http://localhost:3000/reset-workflow" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id-here"}'
```

---

This API documentation provides comprehensive coverage of all endpoints and functionality. For implementation examples, see the Architecture Documentation. For troubleshooting API issues, refer to the Troubleshooting Guide.
