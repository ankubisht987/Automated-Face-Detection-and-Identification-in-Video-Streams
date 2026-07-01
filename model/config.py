"""
Configuration file for the Face Detection & Recognition App.
Modify these values to customize the app behavior.
"""

# Face Detection Settings
FACE_DETECTION = {
    'confidence_threshold': 0.9,  # RetinaFace confidence threshold
    'nms_threshold': 0.4,         # Non-maximum suppression threshold
}

# Face Recognition Settings
FACE_RECOGNITION = {
    'similarity_threshold': 0.6,   # Default similarity threshold (0.5-0.9)
    'embedding_size': 512,        # ArcFace embedding dimension
    'det_size': (640, 640),       # Detection size for InsightFace
}

# Video Processing Settings
VIDEO_PROCESSING = {
    'default_sample_rate': 5,     # Default frames to skip (1-30)
    'max_preview_frames': 30,     # Maximum preview frames to store
    'memory_limit_mb': 2048,      # Memory limit for processing (MB)
}

# UI Settings
UI = {
    'page_title': "Face Detection & Recognition App",
    'page_icon': "👤",
    'layout': "wide",
    'max_upload_size': 200,       # Maximum file size in MB
}

# Model Settings
MODELS = {
    'retinaface_model': None,     # Custom RetinaFace model path
    'insightface_model': None,    # Custom InsightFace model path
    'providers': ['CPUExecutionProvider'],  # Execution providers
}

# Performance Settings
PERFORMANCE = {
    'enable_gpu': False,          # Enable GPU acceleration if available
    'batch_size': 1,              # Batch size for processing
    'num_threads': 4,             # Number of threads for processing
}

# Export Settings
EXPORT = {
    'csv_include_timestamp': True,    # Include timestamp in CSV
    'csv_include_frame': True,        # Include frame number in CSV
    'csv_include_similarity': True,   # Include similarity score in CSV
    'preview_image_format': 'JPEG',   # Format for preview images
    'preview_image_quality': 85,      # Quality for preview images (1-100)
}
