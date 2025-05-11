from fastapi import FastAPI, Query, Body
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from typing import Dict, List, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

# Define FastAPI app
app = FastAPI(title="Multi-Camera YOLO Detection")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Use YOLO model
model = YOLO("best.pt")

# Camera sources dictionary with thread locks
cameras: Dict[str, Dict] = {}
camera_locks: Dict[str, threading.Lock] = {}
# Store phone detections and screenshots
phone_detections: Dict[str, List[Dict]] = {}

# Define request models
class CameraRequest(BaseModel):
    camera_id: str
    source: str
    confidence: float = 0.5

# Function to process frames from a camera
def process_frames(camera_id: str):
    camera = cameras[camera_id]
    cap = camera["capture"]
    
    while camera["active"]:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
            
        # Run YOLOv8 inference
        results = model(frame, conf=camera["confidence"])
        
        # Check if phone is detected
        if len(results[0].boxes) > 0:
            # Save screenshot
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_path = f"screenshots/{camera_id}_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, frame)
            
            # Store detection result
            if camera_id not in phone_detections:
                phone_detections[camera_id] = []
            phone_detections[camera_id].append({
                "timestamp": timestamp,
                "screenshotUrl": screenshot_path
            })
            # Keep only the last 10 detections
            phone_detections[camera_id] = phone_detections[camera_id][-10:]
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Store the processed frame
        with camera_locks[camera_id]:
            camera["last_frame"] = annotated_frame
            
        time.sleep(0.01)

# Function to add a new camera
def add_camera(camera_id: str, source, confidence: float = 0.5):
    if camera_id in cameras:
        stop_camera(camera_id)
        
    # Convert source to int if it's a digit (for webcam indexes)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return False
    
    cameras[camera_id] = {
        "capture": cap,
        "source": source,
        "active": True,
        "last_frame": None,
        "confidence": confidence,
        "thread": None
    }
    camera_locks[camera_id] = threading.Lock()
    
    # Start processing thread
    cameras[camera_id]["thread"] = threading.Thread(
        target=process_frames, 
        args=(camera_id,),
        daemon=True
    )
    cameras[camera_id]["thread"].start()
    return True

# Function to stop a camera
def stop_camera(camera_id: str):
    if camera_id in cameras:
        cameras[camera_id]["active"] = False
        if cameras[camera_id]["thread"]:
            cameras[camera_id]["thread"].join(timeout=1.0)
        if cameras[camera_id]["capture"]:
            cameras[camera_id]["capture"].release()
        del cameras[camera_id]
        del camera_locks[camera_id]
        return True
    return False

# Generator for video streaming
def generate_frames(camera_id: str):
    if camera_id not in cameras:
        return
    
    while cameras[camera_id]["active"]:
        with camera_locks[camera_id]:
            if cameras[camera_id]["last_frame"] is not None:
                frame = cameras[camera_id]["last_frame"].copy()
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + 
                      buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "FastAPI server is running"}

@app.post("/add_camera")
async def api_add_camera(camera: CameraRequest):
    """Add a new camera to the system using JSON body."""
    try:
        print(f"Received add camera request: {camera}")
        # Convert source to int if it's a digit (for webcam indexes)
        source = camera.source
        print(f"Source before conversion: {source}, type: {type(source)}")
        
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            print(f"Converted source to int: {source}")
            
        print(f"Attempting to add camera with source: {source}, type: {type(source)}")
        success = add_camera(camera.camera_id, source, camera.confidence)
        
        if success:
            video_feed_url = f"/video_feed/{camera.camera_id}"
            print(f"Camera {camera.camera_id} added successfully")
            return JSONResponse({
                "success": True, 
                "message": f"Camera {camera.camera_id} added successfully",
                "video_feed_url": video_feed_url
            })
        else:
            print(f"Failed to connect to camera source {source}")
            return JSONResponse({
                "success": False, 
                "message": f"Failed to connect to camera source {source}"
            })
    except Exception as e:
        import traceback
        print(f"Error adding camera: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({
            "success": False, 
            "message": str(e)
        })

@app.post("/stop_camera")
async def api_stop_camera(camera_id: str = Body(..., embed=True)):
    """Stop and remove a camera from the system using JSON body."""
    success = stop_camera(camera_id)
    return JSONResponse({"success": success})

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    """Stream processed video from a specific camera."""
    if camera_id not in cameras:
        return JSONResponse({"error": "Camera not found"})
    
    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/detections/{camera_id}")
async def get_detections(camera_id: str):
    """Get phone detection results for a specific camera."""
    if camera_id not in phone_detections:
        return JSONResponse({"phoneDetections": []})
    return JSONResponse({"phoneDetections": phone_detections[camera_id]})

# Ensure screenshots directory exists
os.makedirs("screenshots", exist_ok=True)

# Mount static files for serving screenshots
app.mount("/screenshots", StaticFiles(directory="screenshots"), name="screenshots")

@app.get("/list_cameras")
async def list_cameras():
    """List all active cameras."""
    return {
        "cameras": [
            {
                "id": camera_id,
                "source": cameras[camera_id]["source"],
                "confidence": cameras[camera_id]["confidence"],
                "video_feed_url": f"/video_feed/{camera_id}"
            }
            for camera_id in cameras
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting server... Access the API at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)