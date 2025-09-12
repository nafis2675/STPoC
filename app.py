"""
Object Detection Web App
Converts the Colab notebook functionality into a web application
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.requests import Request
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
from ultralytics import YOLO
from collections import Counter
from typing import List, Dict, Any, Optional
import os
import tempfile
from pathlib import Path
import socket
import ipaddress

# Initialize FastAPI app
app = FastAPI(title="Object Detection App", description="YOLO-based Object Detection and Comparison")

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for model and storage
model = None
image_results = {
    "image1": None,
    "image2": None
}

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # Connect to a remote address to get the local IP
        # This doesn't actually send data, just determines the route
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            # Use Google's DNS server as the target
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
    except Exception:
        ip = '127.0.0.1'
    return ip

def is_private_ipv4(ip: str) -> bool:
    """Return True if the IP is IPv4 and in a private LAN range."""
    try:
        addr = ipaddress.ip_address(ip)
        return addr.version == 4 and addr.is_private
    except ValueError:
        return False

def get_all_private_ipv4_addresses() -> List[str]:
    """Enumerate all private IPv4 addresses for local adapters (best-effort, stdlib only)."""
    candidates = set()

    # Hostname-based discovery
    try:
        hostname = socket.gethostname()
        host_ips = socket.gethostbyname_ex(hostname)[2]
        for ip in host_ips:
            if is_private_ipv4(ip):
                candidates.add(ip)
    except Exception:
        pass

    # getaddrinfo-based discovery
    try:
        for res in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = res[4][0]
            if is_private_ipv4(ip):
                candidates.add(ip)
    except Exception:
        pass

    # Always include the primary detected IP
    primary = get_local_ip()
    if is_private_ipv4(primary) or primary:
        candidates.add(primary)

    # Ensure we have at least one IP
    if not candidates:
        candidates.add(primary or '127.0.0.1')

    # Sort with primary first
    sorted_ips = sorted(candidates, key=lambda ip: (ip != primary, ip))
    return sorted_ips

class ObjectDetectionService:
    def __init__(self):
        self.model = None
        self.model_name = "yolo11x.pt"
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_name)
            print(f"✅ YOLO model {self.model_name} loaded successfully!")
            print(f"Model classes: {len(self.model.names)} classes available")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise e
    
    def detect_objects(self, image: Image.Image, conf_threshold: float = 0.15, 
                      iou_threshold: float = 0.45, max_detection: int = 500) -> Any:
        """Enhanced object detection with maximum sensitivity"""
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Convert PIL to numpy array
        image_array = np.array(image)
        
        # Run inference with optimized parameters
        results = self.model.predict(
            source=image_array,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            max_det=max_detection,
            show_labels=True,
            show_conf=True,
            agnostic_nms=True,
            verbose=False
        )
        
        return results
    
    def analyze_detections(self, results: Any) -> tuple:
        """Analyze detection results and create detailed report"""
        detections = []
        class_counts = Counter()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get class name and confidence
                    class_id = int(box.cls.cpu().numpy()[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf.cpu().numpy()[0])
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    
                    detection_info = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'area': float((x2-x1) * (y2-y1))
                    }
                    detections.append(detection_info)
                    class_counts[class_name] += 1
        
        return detections, dict(class_counts)
    
    def create_detection_report(self, detections: List[Dict], class_counts: Dict, image_name: str) -> Dict:
        """Create a detailed detection report"""
        return {
            'image_name': image_name,
            'total_objects': len(detections),
            'unique_classes': len(class_counts),
            'class_counts': class_counts,
            'detections': detections
        }
    
    def compare_images(self, detections1: List[Dict], class_counts1: Dict, 
                      detections2: List[Dict], class_counts2: Dict) -> Dict:
        """Compare two sets of detections"""
        total1 = len(detections1)
        total2 = len(detections2)
        
        all_classes = set(class_counts1.keys()) | set(class_counts2.keys())
        class_changes = {}
        
        for class_name in all_classes:
            count1 = class_counts1.get(class_name, 0)
            count2 = class_counts2.get(class_name, 0)
            change = count2 - count1
            class_changes[class_name] = {
                'image1': count1,
                'image2': count2,
                'change': change
            }
        
        new_classes = set(class_counts2.keys()) - set(class_counts1.keys())
        removed_classes = set(class_counts1.keys()) - set(class_counts2.keys())
        
        return {
            'total_objects': {'image1': total1, 'image2': total2, 'change': total2 - total1},
            'class_changes': class_changes,
            'new_classes': list(new_classes),
            'removed_classes': list(removed_classes)
        }
    
    def get_annotated_image(self, image: Image.Image, results: Any) -> str:
        """Get base64 encoded annotated image"""
        # Get annotated image from YOLO results
        for r in results:
            annotated_array = r.plot()
            # Convert BGR to RGB
            annotated_array_rgb = annotated_array[..., ::-1]
            annotated_image = Image.fromarray(annotated_array_rgb)
            
            # Convert to base64
            buffered = io.BytesIO()
            annotated_image.save(buffered, format="JPEG", quality=90)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        
        return ""

# Initialize the detection service
detection_service = ObjectDetectionService()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global detection_service
    # Model is already loaded in __init__
    pass

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/network-access", response_class=HTMLResponse)
async def network_access(request: Request):
    """Serve the network access guide with dynamic IP"""
    local_ip = get_local_ip()
    port = 3000
    candidate_urls = [f"http://{ip}:{port}" for ip in get_all_private_ipv4_addresses()]
    return templates.TemplateResponse("network_access.html", {
        "request": request,
        "local_ip": local_ip,
        "port": port,
        "url": f"http://{local_ip}:{port}",
        "candidate_urls": candidate_urls
    })

@app.post("/detect/{image_number}")
async def detect_objects_numbered(
    image_number: int,
    image: UploadFile = File(...),
    conf_threshold: float = Form(0.15),
    iou_threshold: float = Form(0.45),
    max_detection: int = Form(500)
):
    """Detect objects in uploaded image (for two-image workflow)"""
    global image_results
    
    if image_number not in [1, 2]:
        raise HTTPException(status_code=400, detail="Image number must be 1 or 2")
    
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Detect objects
        results = detection_service.detect_objects(
            pil_image, conf_threshold, iou_threshold, max_detection
        )
        
        # Analyze results
        detections, class_counts = detection_service.analyze_detections(results)
        
        # Create report
        report = detection_service.create_detection_report(
            detections, class_counts, f"Image {image_number}: {image.filename or 'uploaded_image'}"
        )
        
        # Get annotated image
        annotated_image_b64 = detection_service.get_annotated_image(pil_image, results)
        
        # Convert original image to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=90)
        original_image_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Store result
        result = {
            "report": report,
            "original_image": f"data:image/jpeg;base64,{original_image_b64}",
            "annotated_image": annotated_image_b64,
            "parameters": {
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "max_detection": max_detection
            },
            "detections": detections,
            "class_counts": class_counts
        }
        
        image_results[f"image{image_number}"] = result
        
        return JSONResponse({
            "success": True,
            **result,
            "ready_for_comparison": image_results["image1"] is not None and image_results["image2"] is not None
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect-base64/{image_number}")
async def detect_objects_base64_numbered(image_number: int, request: Request):
    """Detect objects in base64 encoded image (for camera/paste functionality)"""
    global image_results
    
    if image_number not in [1, 2]:
        raise HTTPException(status_code=400, detail="Image number must be 1 or 2")
    
    try:
        body = await request.json()
        image_data = body.get("image")
        conf_threshold = body.get("conf_threshold", 0.15)
        iou_threshold = body.get("iou_threshold", 0.45)
        max_detection = body.get("max_detection", 500)
        
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Remove data URL prefix if present
        if "data:image" in image_data:
            image_data = image_data.split(",")[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Detect objects
        results = detection_service.detect_objects(
            pil_image, conf_threshold, iou_threshold, max_detection
        )
        
        # Analyze results
        detections, class_counts = detection_service.analyze_detections(results)
        
        # Create report
        report = detection_service.create_detection_report(
            detections, class_counts, f"Image {image_number}: captured_image"
        )
        
        # Get annotated image
        annotated_image_b64 = detection_service.get_annotated_image(pil_image, results)
        
        # Store result
        result = {
            "report": report,
            "original_image": f"data:image/jpeg;base64,{image_data}",
            "annotated_image": annotated_image_b64,
            "parameters": {
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "max_detection": max_detection
            },
            "detections": detections,
            "class_counts": class_counts
        }
        
        image_results[f"image{image_number}"] = result
        
        return JSONResponse({
            "success": True,
            **result,
            "ready_for_comparison": image_results["image1"] is not None and image_results["image2"] is not None
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/compare")
async def compare_images():
    """Compare two previously processed images"""
    global image_results
    
    try:
        if image_results["image1"] is None or image_results["image2"] is None:
            raise HTTPException(status_code=400, detail="Need both Image 1 and Image 2 to be processed first")
        
        # Compare detections
        comparison = detection_service.compare_images(
            image_results["image1"]["detections"],
            image_results["image1"]["class_counts"],
            image_results["image2"]["detections"],
            image_results["image2"]["class_counts"]
        )
        
        return JSONResponse({
            "success": True,
            "comparison": comparison,
            "image1": image_results["image1"],
            "image2": image_results["image2"]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.post("/adjust-parameters")
async def adjust_parameters(request: Request):
    """Re-analyze both images with new parameters (like Cell 8)"""
    global image_results
    
    try:
        body = await request.json()
        new_conf = body.get("conf_threshold", 0.1)
        new_iou = body.get("iou_threshold", 0.3)
        new_max_det = body.get("max_detection", 1000)
        
        if image_results["image1"] is None or image_results["image2"] is None:
            raise HTTPException(status_code=400, detail="Need both images to be processed first")
        
        adjusted_results = {}
        
        # Re-analyze both images with new parameters
        for img_key in ["image1", "image2"]:
            original_result = image_results[img_key]
            
            # Decode the original image
            image_b64 = original_result["original_image"].split(",")[1]
            image_bytes = base64.b64decode(image_b64)
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Re-detect with new parameters
            results = detection_service.detect_objects(
                pil_image, new_conf, new_iou, new_max_det
            )
            
            # Analyze results
            detections, class_counts = detection_service.analyze_detections(results)
            
            # Create report
            report = detection_service.create_detection_report(
                detections, class_counts, f"{img_key.replace('image', 'Image ')} (Adjusted Parameters)"
            )
            
            # Get annotated image
            annotated_image_b64 = detection_service.get_annotated_image(pil_image, results)
            
            adjusted_results[img_key] = {
                "report": report,
                "original_image": original_result["original_image"],
                "annotated_image": annotated_image_b64,
                "parameters": {
                    "conf_threshold": new_conf,
                    "iou_threshold": new_iou,
                    "max_detection": new_max_det
                },
                "detections": detections,
                "class_counts": class_counts
            }
        
        # Compare adjusted results
        comparison = detection_service.compare_images(
            adjusted_results["image1"]["detections"],
            adjusted_results["image1"]["class_counts"],
            adjusted_results["image2"]["detections"],
            adjusted_results["image2"]["class_counts"]
        )
        
        return JSONResponse({
            "success": True,
            "adjusted_results": adjusted_results,
            "comparison": comparison,
            "parameters_used": {
                "conf_threshold": new_conf,
                "iou_threshold": new_iou,
                "max_detection": new_max_det
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parameter adjustment failed: {str(e)}")

@app.post("/save-results")
async def save_results():
    """Save detection results (like Cell 9)"""
    global image_results
    
    try:
        if image_results["image1"] is None or image_results["image2"] is None:
            raise HTTPException(status_code=400, detail="Need both images to be processed first")
        
        # Create comprehensive report
        timestamp = json.dumps(None, default=str)
        
        # Create text report
        report_text = "OBJECT DETECTION COMPARISON REPORT\n"
        report_text += "=" * 50 + "\n\n"
        
        # Image 1 Report
        img1_report = image_results["image1"]["report"]
        report_text += f"IMAGE 1 REPORT:\n"
        report_text += f"Total Objects: {img1_report['total_objects']}\n"
        report_text += f"Unique Classes: {img1_report['unique_classes']}\n"
        report_text += f"Class Counts: {img1_report['class_counts']}\n\n"
        
        # Image 2 Report
        img2_report = image_results["image2"]["report"]
        report_text += f"IMAGE 2 REPORT:\n"
        report_text += f"Total Objects: {img2_report['total_objects']}\n"
        report_text += f"Unique Classes: {img2_report['unique_classes']}\n"
        report_text += f"Class Counts: {img2_report['class_counts']}\n\n"
        
        # Comparison
        comparison = detection_service.compare_images(
            image_results["image1"]["detections"],
            image_results["image1"]["class_counts"],
            image_results["image2"]["detections"],
            image_results["image2"]["class_counts"]
        )
        
        report_text += f"COMPARISON:\n"
        report_text += f"Total Objects Change: {comparison['total_objects']['change']}\n"
        report_text += f"New Classes: {comparison['new_classes']}\n"
        report_text += f"Removed Classes: {comparison['removed_classes']}\n"
        
        return JSONResponse({
            "success": True,
            "report_text": report_text,
            "full_results": image_results,
            "comparison": comparison,
            "download_ready": True
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save results failed: {str(e)}")

@app.post("/reset-workflow")
async def reset_workflow():
    """Reset the workflow to start fresh"""
    global image_results
    
    image_results = {
        "image1": None,
        "image2": None
    }
    
    return JSONResponse({
        "success": True,
        "message": "Workflow reset successfully"
    })

@app.get("/status")
async def get_status():
    """Get current workflow status"""
    global image_results
    
    return JSONResponse({
        "image1_processed": image_results["image1"] is not None,
        "image2_processed": image_results["image2"] is not None,
        "ready_for_comparison": image_results["image1"] is not None and image_results["image2"] is not None,
        "next_step": "image1" if image_results["image1"] is None else "image2" if image_results["image2"] is None else "comparison"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": detection_service.model is not None}

if __name__ == "__main__":
    import uvicorn
    # Try different ports if 8000 is blocked by corporate network
    uvicorn.run(app, host="0.0.0.0", port=3000)
