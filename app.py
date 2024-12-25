from flask import Flask, jsonify, request
from deepface import DeepFace
import cv2
import os
import time
import base64
from flask_cors import CORS
import logging
import platform
import json
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def get_windows_cameras() -> List[Dict[str, Any]]:
    """Get available cameras on Windows using OpenCV."""
    cameras = []
    # Try the first 10 camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        if cap.isOpened():
            ret, frame = cap.read()
            cameras.append({
                "index": i,
                "working": ret and frame is not None,
                "resolution": str(frame.shape) if ret and frame is not None else None
            })
        cap.release()
    return cameras

def get_linux_cameras() -> List[Dict[str, Any]]:
    """Get available cameras on Linux systems."""
    cameras = []
    # Check /dev/video* devices
    for i in range(10):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            cap = cv2.VideoCapture(i)
            ret, frame = cap.read()
            cameras.append({
                "path": device_path,
                "index": i,
                "working": ret and frame is not None,
                "resolution": str(frame.shape) if ret and frame is not None else None,
                "permissions": oct(os.stat(device_path).st_mode)[-3:]
            })
            cap.release()
    return cameras

def get_system_info() -> Dict[str, Any]:
    """Gather system information for debugging."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "user": os.getenv('USER'),
        "pwd": os.getcwd(),
        "environment": os.environ.get('FLASK_ENV', 'development')
    }
    
    # Platform-specific camera detection
    if platform.system() == 'Windows':
        info['cameras'] = get_windows_cameras()
    else:  # Linux
        info['cameras'] = get_linux_cameras()
    
    # Test default camera (index 0)
    try:
        if platform.system() == 'Windows':
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
            
        info['opencv_camera_open'] = cap.isOpened()
        if cap.isOpened():
            ret, frame = cap.read()
            info['opencv_frame_read'] = ret
            info['frame_shape'] = str(frame.shape) if ret and frame is not None else None
        else:
            info['opencv_frame_read'] = False
        cap.release()
    except Exception as e:
        info['opencv_error'] = str(e)
    
    return info

@app.route('/camera-check', methods=['GET'])
def check_camera():
    """Enhanced endpoint to check camera availability and system configuration."""
    try:
        system_info = get_system_info()
        return jsonify(system_info)
    except Exception as e:
        logger.error(f"Error in camera check: {str(e)}")
        return jsonify({
            "error": str(e),
            "partial_info": get_system_info()
        })

@app.route('/test-capture', methods=['GET'])
def test_capture():
    """Test endpoint to try capturing a single frame."""
    try:
        if platform.system() == 'Windows':
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            return jsonify({"error": "Failed to open camera"}), 500
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return jsonify({"error": "Failed to capture frame"}), 500
            
        return jsonify({
            "success": True,
            "frame_shape": str(frame.shape),
            "frame_type": str(frame.dtype)
        })
        
    except Exception as e:
        logger.error(f"Error capturing frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/capture-emotion', methods=['GET', 'POST'])
def capture_emotion():
    """Main endpoint for emotion detection from camera or uploaded image."""
    try:
        # Handle image upload if provided
        if request.method == 'POST' and 'image' in request.files:
            image_file = request.files['image']
            image_path = f"uploaded_image_{int(time.time())}.jpg"
            image_file.save(image_path)
            frame = cv2.imread(image_path)
            
        else:
            # Platform-specific camera initialization
            if platform.system() == 'Windows':
                webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                webcam = cv2.VideoCapture(0)
                
            if not webcam.isOpened():
                return jsonify({"error": "Failed to open camera"}), 500

            # Multiple attempts to read frame
            max_attempts = 3
            frame = None
            for attempt in range(max_attempts):
                ret, frame = webcam.read()
                if ret and frame is not None:
                    break
                logger.warning(f"Attempt {attempt + 1} failed to capture frame")
                time.sleep(0.5)

            webcam.release()
            
            if frame is None:
                return jsonify({"error": "Failed to capture frame after multiple attempts"}), 500

            # Save frame temporarily
            image_path = f"captured_image_{int(time.time())}.jpg"
            cv2.imwrite(image_path, frame)

        try:
            # Analyze the image for emotions
            analysis = DeepFace.analyze(
                img_path=image_path, 
                actions=['emotion'], 
                detector_backend='mtcnn',
                enforce_detection=False
            )
            
            emotions = analysis[0]['emotion']
            dominant_emotion = analysis[0]['dominant_emotion']

            # Apply emotion thresholds
            if emotions['sad'] > 30:
                dominant_emotion = 'sad'
            elif emotions['angry'] > 30:
                dominant_emotion = 'angry'

            # Convert to base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                "dominant_emotion": dominant_emotion,
                "emotions": emotions,
                "captured_image": img_base64
            })

        except Exception as e:
            logger.error(f"Error during emotion analysis: {str(e)}")
            return jsonify({"error": f"Error during analysis: {str(e)}"}), 500

        finally:
            # Clean up
            if os.path.exists(image_path):
                os.remove(image_path)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"System info: {json.dumps(get_system_info(), indent=2)}")
    app.run(host='0.0.0.0', port=5000, debug=True)