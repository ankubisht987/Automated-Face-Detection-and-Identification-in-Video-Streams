#!/usr/bin/env python3
"""
Test script to verify all dependencies are properly installed.
Run this before starting the main application.
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'streamlit',
        'cv2',
        'numpy',
        'pandas',
        'PIL',
        'deepface'
    ]
    
    failed_imports = []
    
    print("Testing package imports...")
    print("=" * 40)
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ {package} - OpenCV version: {cv2.__version__}")
            elif package == 'PIL':
                from PIL import Image
                print(f"✅ {package} - Pillow version: {Image.__version__}")
            elif package == 'numpy':
                import numpy as np
                print(f"✅ {package} - NumPy version: {np.__version__}")
            elif package == 'pandas':
                import pandas as pd
                print(f"✅ {package} - Pandas version: {pd.__version__}")
            elif package == 'streamlit':
                import streamlit as st
                print(f"✅ {package} - Streamlit version: {st.__version__}")
            elif package == 'deepface':
                import deepface
                print(f"✅ {package} - DeepFace version: {deepface.__version__}")
        except ImportError as e:
            print(f"❌ {package} - Import failed: {e}")
            failed_imports.append(package)
        except Exception as e:
            print(f"⚠️  {package} - Import succeeded but with warning: {e}")
    
    print("=" * 40)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        print("You can now run the main application with: streamlit run app.py")
        return True

def test_models():
    """Test if DeepFace models can be loaded"""
    print("\nTesting DeepFace model loading...")
    print("=" * 40)
    
    try:
        # Test DeepFace with RetinaFace backend
        print("Testing DeepFace with RetinaFace backend...")
        from deepface import DeepFace
        import tempfile
        import numpy as np
        from PIL import Image
        
        # Create a simple test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            pil_image = Image.fromarray(test_image)
            pil_image.save(tmp_file.name, 'JPEG')
            image_path = tmp_file.name
        
        try:
            # Test face detection
            faces = DeepFace.extract_faces(image_path, detector_backend="retinaface")
            print("✅ DeepFace RetinaFace backend working")
            
            # Test face recognition
            embeddings = DeepFace.represent(image_path, model_name="ArcFace", detector_backend="retinaface")
            print("✅ DeepFace ArcFace model working")
            
        finally:
            # Clean up
            import os
            if os.path.exists(image_path):
                os.unlink(image_path)
        
        print("=" * 40)
        print("✅ All DeepFace models working successfully!")
        return True
        
    except Exception as e:
        print(f"❌ DeepFace model loading failed: {e}")
        print("This might be due to missing model files or insufficient resources.")
        return False

if __name__ == "__main__":
    print("Face Detection & Recognition App - Installation Test")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    else:
        print("✅ Python version is compatible")
    
    print()
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        print()
        # Test models
        models_ok = test_models()
        
        if models_ok:
            print("\n🎉 All tests passed! Your installation is ready.")
        else:
            print("\n⚠️  Models failed to load. The app may not work properly.")
    else:
        print("\n❌ Installation test failed. Please fix the import issues.")
        sys.exit(1)
