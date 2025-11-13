# Face Detection & Recognition Web App

A Streamlit-based web application that detects and recognizes faces in videos using DeepFace with RetinaFace backend for face detection and ArcFace model for face recognition.

## Features

- **Face Detection**: Uses DeepFace with RetinaFace backend for accurate face detection in video frames
- **Face Recognition**: Employs DeepFace with ArcFace model for high-accuracy face recognition
- **Video Processing**: Processes MP4 video files with configurable frame sampling
- **Timestamp Extraction**: Extracts precise timestamps (HH:MM:SS) where reference faces appear
- **Results Export**: Provides downloadable CSV files with detection timestamps
- **Preview Frames**: Shows sample frames where detections occurred
- **Modern UI**: Clean, responsive interface built with Streamlit

## Requirements

- Python 3.8+
- Windows/Linux/macOS
- Sufficient RAM for video processing (recommended: 8GB+)

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## Usage

### Step 1: Upload Reference Face
- Upload a clear, front-facing image of the person you want to detect
- Supported formats: JPG, JPEG, PNG
- Click "Set as Reference Face" to process the image

### Step 2: Upload Video
- Upload an MP4 video file containing the person you want to track
- The app will display a preview of the video

### Step 3: Configure Processing
- **Sample Rate**: Choose how many frames to skip (higher = faster but may miss detections)
- **Similarity Threshold**: Adjust how strict the face matching should be (0.5-0.9)

### Step 4: Process Video
- Click "Start Processing" to begin face detection and recognition
- Monitor progress with the progress bar
- Wait for processing to complete

### Step 5: View Results
- **Summary Metrics**: Total detections, first/last detection times
- **Timestamps Table**: Detailed list of all detections with frame numbers
- **CSV Download**: Export results for further analysis
- **Preview Frames**: Visual confirmation of detections

## Technical Details

### Architecture
The app is built with a modular, object-oriented design:

- **`FaceDetector`**: Handles face detection using DeepFace with RetinaFace backend
- **`FaceRecognizer`**: Manages face recognition using DeepFace with ArcFace model
- **`VideoProcessor`**: Coordinates video processing and face detection workflow

### Face Recognition Pipeline
1. **Detection**: DeepFace with RetinaFace backend identifies face regions in each frame
2. **Extraction**: DeepFace with ArcFace model generates face embeddings
3. **Comparison**: Cosine similarity between reference and detected face embeddings
4. **Thresholding**: Configurable similarity threshold for match determination

### Performance Optimization
- **Frame Sampling**: Process every Nth frame to balance speed vs. accuracy
- **Memory Management**: Store preview frames selectively to avoid memory issues
- **Progress Tracking**: Real-time progress updates during processing

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure all dependencies are properly installed
   - Check internet connection for model downloads
   - Restart the application

2. **Memory Issues**:
   - Reduce sample rate for large videos
   - Close other applications to free up RAM
   - Process shorter video segments

3. **Slow Processing**:
   - Increase sample rate (process fewer frames)
   - Use smaller video files for testing
   - Ensure sufficient CPU resources

4. **No Detections**:
   - Verify reference face image quality
   - Adjust similarity threshold (try lower values)
   - Check if faces are clearly visible in video

### Performance Tips

- **Reference Image**: Use high-quality, well-lit, front-facing photos
- **Video Quality**: Higher resolution videos provide better detection accuracy
- **Sample Rate**: Start with 5-10 for good balance of speed/accuracy
- **Threshold**: Start with 0.6 and adjust based on results

## File Structure

```
facesss/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Dependencies

- **Streamlit**: Web application framework
- **DeepFace**: Face recognition library with built-in RetinaFace and ArcFace backends
- **OpenCV**: Video processing and computer vision
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and CSV export
- **Pillow**: Image processing
- **TensorFlow**: Deep learning backend for DeepFace

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## Support

If you encounter any issues or have questions, please check the troubleshooting section above or create an issue in the project repository.
