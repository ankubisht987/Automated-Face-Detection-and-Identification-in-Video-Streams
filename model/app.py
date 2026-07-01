from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import tempfile
import os
from datetime import timedelta
from retinaface import RetinaFace
from insightface.app import FaceAnalysis
from PIL import Image
import io
import warnings
from pathlib import Path
import uuid
import asyncio
from typing import Optional
import json
import base64

warnings.filterwarnings('ignore')

# ========================
# FastAPI App Setup
# ========================
app = FastAPI(title="Face Detection & Recognition API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Global State & Storage
# ========================
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

sessions = {}  # Store session data: {session_id: {...}}

# ========================
# Image Enhancement Utils (from Streamlit code)
# ========================
def apply_gamma_correction(image_bgr, gamma=1.5):
    if gamma <= 0:
        return image_bgr
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image_bgr, table)

def apply_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    y_eq = clahe.apply(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

def enhance_low_light(image_bgr):
    img = apply_gamma_correction(image_bgr, gamma=1.5)
    img = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    return img

def auto_brightness_contrast(image_bgr, clip_hist_percent=1.0):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    accumulator = np.cumsum(hist)
    clip_amount = (clip_hist_percent / 100.0) * (accumulator[-1])
    clip_amount /= 2.0
    minimum_gray = np.searchsorted(accumulator, clip_amount)
    maximum_gray = np.searchsorted(accumulator, accumulator[-1] - clip_amount)
    if maximum_gray == minimum_gray:
        return image_bgr
    alpha = 255.0 / max(1, (maximum_gray - minimum_gray))
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(image_bgr, alpha=alpha, beta=beta)
    return auto_result

def generate_enhancement_variants(image_bgr):
    variants = []
    variants.append(image_bgr)
    for g in (1.3, 1.6, 1.9):
        variants.append(apply_gamma_correction(image_bgr, gamma=g))
    variants.append(apply_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)))
    variants.append(apply_clahe(image_bgr, clip_limit=3.0, tile_grid_size=(8, 8)))
    variants.append(auto_brightness_contrast(image_bgr, clip_hist_percent=1.0))
    variants.append(auto_brightness_contrast(image_bgr, clip_hist_percent=2.0))
    variants.append(cv2.convertScaleAbs(image_bgr, alpha=1.0, beta=20))
    variants.append(cv2.convertScaleAbs(image_bgr, alpha=1.0, beta=-20))
    variants.append(cv2.convertScaleAbs(image_bgr, alpha=1.2, beta=0))
    variants.append(cv2.convertScaleAbs(image_bgr, alpha=0.8, beta=0))
    unique = []
    seen = set()
    for v in variants:
        key = (v.shape, v.dtype.str, int(v.mean()))
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique

# ========================
# Helper function to encode frame to base64
# ========================
def encode_frame_to_base64(frame_bgr):
    """Encode BGR image frame to base64 JPEG string"""
    _, buffer = cv2.imencode('.jpg', frame_bgr)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

# ========================
# Face Recognition Classes
# ========================
class FaceRecognizer:
    def __init__(self):
        self.app = None
        self.reference_embedding = None
    
    def load_model(self):
        try:
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            return True
        except Exception as e:
            print(f"Error loading InsightFace: {e}")
            return False
    
    def extract_embedding(self, face_image_bgr):
        if self.app is None:
            if not self.load_model():
                return None
        try:
            image = np.ascontiguousarray(face_image_bgr)
            faces = self.app.get(image)
            if faces:
                return faces[0].embedding
            return None
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def set_reference_face(self, image_data):
        try:
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                bgr = image_data
            
            if self.app is None:
                if not self.load_model():
                    return False
            
            image = np.ascontiguousarray(bgr)
            faces = self.app.get(image)
            if not faces:
                for var in generate_enhancement_variants(image):
                    faces = self.app.get(np.ascontiguousarray(var))
                    if faces:
                        break
            
            if not faces:
                return False
            
            def bbox_area(face):
                x1, y1, x2, y2 = face.bbox.astype(int)
                return max(0, x2 - x1) * max(0, y2 - y1)
            
            faces.sort(key=bbox_area, reverse=True)
            self.reference_embedding = faces[0].embedding
            return True
        except Exception as e:
            print(f"Error setting reference: {e}")
            return False
    
    def compare_faces(self, face_image_bgr, threshold=0.6):
        if self.reference_embedding is None:
            return False
        try:
            current_embedding = self.extract_embedding(face_image_bgr)
            if current_embedding is None:
                return False
            similarity = np.dot(self.reference_embedding, current_embedding) / (
                np.linalg.norm(self.reference_embedding) * np.linalg.norm(current_embedding)
            )
            return similarity > threshold
        except Exception as e:
            print(f"Error comparing: {e}")
            return False

# ========================
# API Endpoints
# ========================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/api/create-session")
async def create_session():
    """Create a new processing session"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "id": session_id,
        "status": "initialized",
        "reference_set": False,
        "detections": [],
        "preview_frames": [],
        "recognizer": FaceRecognizer(),
    }
    return {
        "session_id": session_id,
        "status": "success"
    }

@app.post("/api/set-reference/{session_id}")
async def set_reference(session_id: str, file: UploadFile = File(...)):
    """Set reference face for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        contents = await file.read()
        recognizer = sessions[session_id]["recognizer"]
        
        if recognizer.set_reference_face(contents):
            sessions[session_id]["reference_set"] = True
            return {
                "status": "success",
                "message": "Reference face set successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="No face detected in reference image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-video/{session_id}")
async def process_video(
    session_id: str,
    file: UploadFile = File(...),
    sample_rate: int = 1,
    similarity_threshold: float = 0.6,
    background_tasks: BackgroundTasks = None
):
    """Start video processing"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not sessions[session_id]["reference_set"]:
        raise HTTPException(status_code=400, detail="Reference face not set")
    
    try:
        # Save video temporarily
        video_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        with open(video_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Start background processing
        background_tasks.add_task(
            process_video_background,
            session_id,
            str(video_path),
            sample_rate,
            similarity_threshold
        )
        
        sessions[session_id]["status"] = "processing"
        return {
            "status": "started",
            "session_id": session_id,
            "message": "Video processing started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_video_background(session_id: str, video_path: str, sample_rate: int, threshold: float):
    """Background task for video processing"""
    try:
        recognizer = sessions[session_id]["recognizer"]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            sessions[session_id]["status"] = "error"
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        detection_count = 0
        detections = []
        preview_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                if recognizer.app is None:
                    recognizer.load_model()
                
                faces = recognizer.app.get(np.ascontiguousarray(frame))
                if not faces:
                    for var in generate_enhancement_variants(frame):
                        faces = recognizer.app.get(np.ascontiguousarray(var))
                        if faces:
                            break
                
                for face in faces:
                    emb = face.embedding
                    if emb is None:
                        continue
                    
                    similarity = np.dot(recognizer.reference_embedding, emb) / (
                        np.linalg.norm(recognizer.reference_embedding) * np.linalg.norm(emb)
                    )
                    
                    if similarity > threshold:
                        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                        timestamp = float(pos_msec) / 1000.0 if pos_msec else (frame_count / max(fps, 1e-6))
                        time_str = str(timedelta(seconds=int(timestamp)))
                        
                        detections.append({
                            "timestamp": timestamp,
                            "time_string": time_str,
                            "frame_number": frame_count,
                            "similarity": float(similarity)
                        })
                        
                        # Store preview frame every 10 detections
                        if detection_count % 10 == 0:
                            # Encode frame to base64 with metadata
                            frame_base64 = encode_frame_to_base64(frame)
                            preview_frames.append({
                                "frame": frame_base64,
                                "timestamp": time_str,
                                "frame_number": frame_count,
                                "similarity": float(similarity)
                            })
                        
                        detection_count += 1
                        break
            
            frame_count += 1
        
        cap.release()
        
        sessions[session_id]["detections"] = detections
        sessions[session_id]["preview_frames"] = preview_frames
        sessions[session_id]["status"] = "completed"
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
    
    except Exception as e:
        print(f"Error in background processing: {e}")
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)

@app.get("/api/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get session status"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "detection_count": len(session["detections"]),
        "reference_set": session["reference_set"]
    }

@app.get("/api/session/{session_id}/results")
async def get_results(session_id: str):
    """Get detection results"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "detections": session["detections"],
        "total_detections": len(session["detections"]),
        "preview_frames": session["preview_frames"]
    }

@app.post("/api/session/{session_id}/segments")
async def get_segments(session_id: str, gap_threshold: float = 1.0):
    """Get detection segments"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    detections = session["detections"]
    
    if not detections:
        return {"segments": []}
    
    times = sorted([d["timestamp"] for d in detections])
    segments = []
    
    if times:
        start = times[0]
        prev = times[0]
        for t in times[1:]:
            if (t - prev) > gap_threshold:
                segments.append({
                    "start_time": str(timedelta(seconds=int(start))),
                    "end_time": str(timedelta(seconds=int(prev))),
                    "duration": str(timedelta(seconds=int(prev - start)))
                })
                start = t
            prev = t
        segments.append({
            "start_time": str(timedelta(seconds=int(start))),
            "end_time": str(timedelta(seconds=int(prev))),
            "duration": str(timedelta(seconds=int(prev - start)))
        })
    
    return {"segments": segments}

@app.post("/api/session/{session_id}/cleanup")
async def cleanup_session(session_id: str):
    """Clean up session"""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleaned"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
