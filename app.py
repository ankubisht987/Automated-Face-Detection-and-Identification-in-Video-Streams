import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import os
from datetime import timedelta
import time
from retinaface import RetinaFace
import insightface
from insightface.app import FaceAnalysis
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# Image enhancement utils
# ------------------------
# 180 CNN libraries retinaface,insightface
# These functions make dark or low-quality faces easier to detect.
# Adjusts brightness nonlinearly using gamma correction.
# Builds a lookup table and applies it with cv2.LUT.
def apply_gamma_correction(image_bgr, gamma=1.5):
    if gamma <= 0:
        return image_bgr
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image_bgr, table)

# Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the luminance (Y) channel to enhance local contrast
def apply_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    y_eq = clahe.apply(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

# Combines both methods for dark scenes.
def enhance_low_light(image_bgr):
    # Sequence: slight gamma correction then CLAHE
    img = apply_gamma_correction(image_bgr, gamma=1.5)
    img = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    return img

# Computes histogram → trims extremes (based on clip_hist_percent) → scales pixel values to 
# optimize brightness and contrast automatically
def auto_brightness_contrast(image_bgr, clip_hist_percent=1.0):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_size = len(hist)
    accumulator = np.cumsum(hist)
    clip_amount = (clip_hist_percent / 100.0) * (accumulator[-1])
    clip_amount /= 2.0

    # Locate left cut
    minimum_gray = np.searchsorted(accumulator, clip_amount)
    # Locate right cut
    maximum_gray = np.searchsorted(accumulator, accumulator[-1] - clip_amount)
    if maximum_gray == minimum_gray:
        return image_bgr

    # Calculate alpha and beta values
    alpha = 255.0 / max(1, (maximum_gray - minimum_gray))
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(image_bgr, alpha=alpha, beta=beta)
    return auto_result

# Creates multiple versions of an image with different adjustments 
# (gamma, CLAHE, contrast, brightness, etc.).
# Used to retry face detection if the image is poor quality.
def generate_enhancement_variants(image_bgr):
    variants = []
    variants.append(image_bgr)
    # Gamma corrections
    for g in (1.3, 1.6, 1.9):
        variants.append(apply_gamma_correction(image_bgr, gamma=g))
    # CLAHE
    variants.append(apply_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)))
    variants.append(apply_clahe(image_bgr, clip_limit=3.0, tile_grid_size=(8, 8)))
    # Auto brightness/contrast
    variants.append(auto_brightness_contrast(image_bgr, clip_hist_percent=1.0))
    variants.append(auto_brightness_contrast(image_bgr, clip_hist_percent=2.0))
    # Slight brightness lift/darken
    variants.append(cv2.convertScaleAbs(image_bgr, alpha=1.0, beta=20))
    variants.append(cv2.convertScaleAbs(image_bgr, alpha=1.0, beta=-20))
    # Contrast tweaks
    variants.append(cv2.convertScaleAbs(image_bgr, alpha=1.2, beta=0))
    variants.append(cv2.convertScaleAbs(image_bgr, alpha=0.8, beta=0))
    # Deduplicate by id
    unique = []
    seen = set()
    for v in variants:
        key = (v.shape, v.dtype.str, int(v.mean()))
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique

# Page configuration
st.set_page_config(
    page_title="Face Detection & Recognition App",
    page_icon="👤",
    layout="wide"
)

# Custom CSS for better styling in Stremlit
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .results-section {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .timestamp-table {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# FaceDetector Class (RetinaFace)
# Handles detecting faces in still images.
class FaceDetector:
    """Class for handling face detection using RetinaFace"""
    
    def __init__(self):
        """Initialize the face detector"""
        self.detector = None
        
    def load_detector(self):
        """Load RetinaFace detector"""
        try:
            self.detector = RetinaFace.build_model() #API 
            return True
        except Exception as e:
            st.error(f"Error loading RetinaFace detector: {str(e)}")
            return False

# Converts input to numpy.
# Runs RetinaFace.detect_faces(image).
# If none found, tries enhancement fallback.
# Parses output (RetinaFace returns different formats depending on version):
# dict: multiple entries, each with facial_area.
# tuple: bounding boxes list.
# Crops each detected face region and returns them as a list of face crops.    
    def detect_faces(self, image):
        """Detect faces in an image using RetinaFace"""
        if self.detector is None:
            if not self.load_detector():
                return []
        
        try:
            # Convert PIL image to numpy array
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Detect faces (different packages may return different formats)API
            faces = RetinaFace.detect_faces(image)

            if faces is None:
                # Try low-light enhancement fallback
                if isinstance(image, Image.Image):
                    img = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
                else:
                    img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                img_enh = enhance_low_light(img)
                faces = RetinaFace.detect_faces(cv2.cvtColor(img_enh, cv2.COLOR_BGR2RGB))
                if faces is None:
                    return []

            face_regions = []

            # Case 1: dict format: {key: { 'facial_area': [x1,y1,x2,y2], ... }, ...}
            if isinstance(faces, dict):
                for face_key in faces.keys():
                    face = faces[face_key]
                    facial_area = face.get("facial_area") or face.get("facialArea")
                    if facial_area is None:
                        continue
                    x1, y1, x2, y2 = map(int, facial_area)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
                    face_region = image[y1:y2, x1:x2]
                    face_regions.append(face_region)
                return face_regions

            # Case 2: tuple format: (bboxes, landmarks, scores)
            if isinstance(faces, tuple) and len(faces) >= 1:
                bboxes = faces[0]
                for box in bboxes:
                    x1, y1, x2, y2 = [int(v) for v in box[:4]]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
                    face_region = image[y1:y2, x1:x2]
                    face_regions.append(face_region)
                return face_regions

            # Unknown format
            return []
        except Exception as e:
            st.error(f"Error detecting faces: {str(e)}")
            return []

class FaceRecognizer:
    """Class for handling face recognition using ArcFace"""
    
    def __init__(self):
        """Initialize the face recognizer"""
        self.app = None
        self.reference_embedding = None
        
    def load_model(self):
        """Load InsightFace model"""
        try:
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])#API
            self.app.prepare(ctx_id=0, det_size=(640, 640)) #API
            return True
        except Exception as e:
            st.error(f"Error loading InsightFace model: {str(e)}")
            return False
    
    def extract_embedding(self, face_image_bgr):
        """Extract face embedding using ArcFace from a BGR image or face crop"""
        if self.app is None:
            if not self.load_model():
                return None

        try:
            image = np.ascontiguousarray(face_image_bgr)
            faces = self.app.get(image) #API
            if faces:
                return faces[0].embedding
            return None
        except Exception as e:
            st.error(f"Error extracting embedding: {str(e)}")
            return None
    
    def set_reference_face(self, reference_image_input):
        """Detect the main face in the input and store its embedding (BGR expected)"""
        try:
            # Convert UploadedFile / PIL to BGR numpy
            if isinstance(reference_image_input, Image.Image):
                rgb = reference_image_input.convert('RGB')
                bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
            else:
                bgr = reference_image_input

            if self.app is None:
                if not self.load_model():
                    return False

            # Try detection; if none, enhance for low light and try again
            image = np.ascontiguousarray(bgr)
            faces = self.app.get(image)
            if not faces:
                # Try multiple enhancement variants
                for var in generate_enhancement_variants(image):
                    faces = self.app.get(np.ascontiguousarray(var))
                    if faces:
                        break
            if not faces:
                st.error("No face detected in the reference image. Please try a clearer, frontal photo.")
                return False

            # Choose largest face by bbox area
            def bbox_area(face):
                x1, y1, x2, y2 = face.bbox.astype(int)
                return max(0, x2 - x1) * max(0, y2 - y1)

            faces.sort(key=bbox_area, reverse=True)
            self.reference_embedding = faces[0].embedding
            return True
        except Exception as e:
            st.error(f"Error setting reference face: {str(e)}")
            return False
    
    def compare_faces(self, face_image_bgr, threshold=0.6):
        """Compare a face (BGR crop) with the reference face"""
        if self.reference_embedding is None:
            return False
        
        try:
            current_embedding = self.extract_embedding(face_image_bgr)
            if current_embedding is None:
                return False
            
            # Calculate cosine similarity
            similarity = np.dot(self.reference_embedding, current_embedding) / (
                np.linalg.norm(self.reference_embedding) * np.linalg.norm(current_embedding)
            )
            
            return similarity > threshold
        except Exception as e:
            st.error(f"Error comparing faces: {str(e)}")
            return False

class VideoProcessor:
    """Class for processing video files and detecting faces"""
    
    def __init__(self, face_detector, face_recognizer):
        """Initialize video processor with face detector and recognizer"""
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.detection_timestamps = []
        self.preview_frames = []
        
    def process_video(self, video_path, sample_rate=1, similarity_threshold=0.6):
        """Process video and detect reference face appearances"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error opening video file")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            st.info(f"Video Info: {total_frames} frames, {fps:.2f} FPS, Duration: {duration:.2f}s")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            detection_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame based on sample rate
                if frame_count % sample_rate == 0:
                    # Use InsightFace to detect faces and get embeddings (expects BGR)
                    if self.face_recognizer.app is None:
                        if not self.face_recognizer.load_model():
                            return False
                    # Try normal frame first, then multiple enhanced variants
                    faces = self.face_recognizer.app.get(np.ascontiguousarray(frame))
                    if not faces:
                        for var in generate_enhancement_variants(frame):
                            faces = self.face_recognizer.app.get(np.ascontiguousarray(var))
                            if faces:
                                break

                    for face in faces:
                        emb = face.embedding
                        if emb is None:
                            continue
                        similarity = np.dot(self.face_recognizer.reference_embedding, emb) / (
                            np.linalg.norm(self.face_recognizer.reference_embedding) * np.linalg.norm(emb)
                        )
                        if similarity > similarity_threshold:
                            # Use precise stream position for reliable timing
                            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                            timestamp = float(pos_msec) / 1000.0 if pos_msec is not None else (frame_count / max(fps, 1e-6))
                            time_str = str(timedelta(seconds=timestamp))
                            self.detection_timestamps.append({
                                'timestamp': timestamp,
                                'time_string': time_str,
                                'frame_number': frame_count
                            })
                            if detection_count % 10 == 0:
                                # Convert to RGB only for display
                                preview_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                self.preview_frames.append({
                                    'frame': preview_rgb,
                                    'timestamp': time_str
                                })
                            detection_count += 1
                            break
                    
                    # Update progress
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")
                
                frame_count += 1
            
            cap.release()
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            return True
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return False
    
    def get_detection_dataframe(self):
        """Convert detection timestamps to pandas DataFrame"""
        if not self.detection_timestamps:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.detection_timestamps)
        df = df.sort_values('timestamp')
        return df

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">👤 Face Detection & Recognition App</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This app detects and recognizes faces in videos using RetinaFace for detection 
    and ArcFace for recognition. Upload a reference face image and a video file to get started.
    """)
    
    # Initialize components
    if 'face_detector' not in st.session_state:
        st.session_state.face_detector = FaceDetector()
    
    if 'face_recognizer' not in st.session_state:
        st.session_state.face_recognizer = FaceRecognizer()
    
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = VideoProcessor(
            st.session_state.face_detector, 
            st.session_state.face_recognizer
        )
    
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.header("📁 Upload Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reference Face Image")
        reference_image = st.file_uploader(  #API
            "Upload a reference face image (JPG/PNG)",
            type=['jpg', 'jpeg', 'png'],
            key="reference_upload"
        )
        
        if reference_image:
            st.image(reference_image, caption="Reference Face", use_column_width=True)
            
            # Set reference face
            if st.button("Set as Reference Face"):
                with st.spinner("Processing reference face..."):
                    # Convert UploadedFile to PIL.Image before processing
                    pil_ref_image = Image.open(reference_image)
                    if st.session_state.face_recognizer.set_reference_face(pil_ref_image):
                        st.success("Reference face set successfully!")
                    else:
                        st.error("Failed to set reference face")
    
    with col2:
        st.subheader("Video File")
        video_file = st.file_uploader(
            "Upload a video file (MP4)",
            type=['mp4'],
            key="video_upload"
        )
        
        if video_file:
            st.video(video_file)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing section
    if reference_image and video_file and st.session_state.face_recognizer.reference_embedding is not None:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.header("🔍 Process Video")
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            sample_rate = st.slider("Sample Rate (process every Nth frame)", 1, 30, 5, 
                                   help="Higher values = faster processing but may miss detections")
        
        with col2:
            similarity_threshold = st.slider("Similarity Threshold", 0.5, 0.9, 0.6, 0.05,
                                           help="Higher values = stricter matching")
        
        # Process button
        if st.button("🚀 Start Processing", type="primary"):
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_file.read())
                video_path = tmp_file.name
            
            try:
                with st.spinner("Processing video... This may take a while depending on video length."):
                    # Update similarity threshold
                    st.session_state.face_recognizer.similarity_threshold = similarity_threshold
                    
                    # Process video
                    success = st.session_state.video_processor.process_video(
                        video_path, sample_rate, similarity_threshold
                    )
                    
                    if success:
                        st.success("Video processing completed!")
                        
                        # Show results
                        show_results()
                    else:
                        st.error("Video processing failed!")
            
            finally:
                # Clean up temporary file
                if os.path.exists(video_path):
                    os.unlink(video_path)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif reference_image and video_file:
        st.warning("Please set the reference face before processing the video.")
    
    elif not (reference_image and video_file):
        st.info("Please upload both a reference face image and a video file to begin.")

def show_results():
    """Display processing results"""
    st.header("📊 Detection Results")
    
    # Get detection data
    df = st.session_state.video_processor.get_detection_dataframe()
    
    if df.empty:
        st.info("No face detections found in the video.")
        return
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Detections", len(df))
    with col2:
        st.metric("First Detection", df.iloc[0]['time_string'])
    with col3:
        st.metric("Last Detection", df.iloc[-1]['time_string'])
    
    # Display timestamps table
    st.subheader("⏰ Detection Timestamps")
    st.dataframe(df[['time_string', 'timestamp', 'frame_number']], 
                use_container_width=True)
    
    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="face_detections.csv",
        mime="text/csv"
    )

    # Summarize detections into contiguous segments
    if not df.empty:
        st.subheader("🕒 Detected Time Segments")
        gap_threshold = st.number_input(
            "Max gap between detections to merge into a segment (seconds)",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            help="Detections closer than this are merged into the same segment"
        )

        times = df['timestamp'].sort_values().tolist()
        segments = []
        if times:
            start = times[0]
            prev = times[0]
            for t in times[1:]:
                if (t - prev) > gap_threshold:
                    segments.append((start, prev))
                    start = t
                prev = t
            segments.append((start, prev))

        seg_rows = []
        for s, e in segments:
            seg_rows.append({
                'start_time': str(timedelta(seconds=s)),
                'end_time': str(timedelta(seconds=e)),
                'duration': str(timedelta(seconds=(e - s)))
            })
        seg_df = pd.DataFrame(seg_rows)
        st.dataframe(seg_df, use_container_width=True)
        seg_csv = seg_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Segments CSV",
            data=seg_csv,
            file_name="face_detection_segments.csv",
            mime="text/csv"
        )
    
    # Show preview frames
    if st.session_state.video_processor.preview_frames:
        st.subheader("🖼️ Preview Frames")
        
        # Display preview frames in a grid
        cols = st.columns(3)
        for i, preview in enumerate(st.session_state.video_processor.preview_frames):
            col_idx = i % 3
            with cols[col_idx]:
                st.image(preview['frame'], 
                        caption=f"Detection at {preview['timestamp']}", 
                        use_column_width=True)

if __name__ == "__main__":
    main()
