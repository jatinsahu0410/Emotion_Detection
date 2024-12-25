from flask import Flask, jsonify, request
from deepface import DeepFace
import cv2
import os
import time
import base64
from flask_cors import CORS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def initialize_camera():
    """Try different camera indices and return working camera."""
    for index in range(10):  # Try first 10 camera indices
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap, index
    return None, None

def get_camera_list():
    """Get list of available video devices on Linux."""
    try:
        video_devices = []
        for i in range(10):
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                video_devices.append(device_path)
        return video_devices
    except Exception as e:
        logger.error(f"Error getting camera list: {str(e)}")
        return []

@app.route('/camera-check', methods=['GET'])
def check_camera():
    """Endpoint to check camera availability and configuration."""
    camera_info = {
        "available_devices": get_camera_list(),
        "opencv_version": cv2.__version__,
        "environment": os.environ.get('FLASK_ENV', 'development')
    }
    return jsonify(camera_info)

@app.route('/capture-emotion', methods=['GET', 'POST'])
def capture_emotion():
    try:
        # If image is provided in POST request, use it instead of capturing
        if request.method == 'POST' and 'image' in request.files:
            image_file = request.files['image']
            image_path = f"uploaded_image_{int(time.time())}.jpg"
            image_file.save(image_path)
            frame = cv2.imread(image_path)
            
        else:
            # Initialize the webcam with fallback options
            webcam, camera_index = initialize_camera()
            if webcam is None:
                return jsonify({"error": "No working camera found", 
                              "available_devices": get_camera_list()}), 500

            logger.info(f"Successfully opened camera at index {camera_index}")
            
            # Try to read frame with multiple attempts
            max_attempts = 3
            for attempt in range(max_attempts):
                ret, frame = webcam.read()
                if ret and frame is not None:
                    break
                time.sleep(0.5)
            
            if not ret or frame is None:
                webcam.release()
                return jsonify({"error": "Failed to capture image after multiple attempts"}), 500

            image_path = f"captured_image_{int(time.time())}.jpg"
            cv2.imwrite(image_path, frame)
            webcam.release()

        try:
            # Analyze the image for emotions
            analysis = DeepFace.analyze(img_path=image_path, 
                                      actions=['emotion'], 
                                      detector_backend='mtcnn',
                                      enforce_detection=False)  # Added to prevent face detection errors
            
            emotions = analysis[0]['emotion']
            dominant_emotion = analysis[0]['dominant_emotion']

            # Update the dominant emotion based on thresholds
            if emotions['sad'] > 5:
                dominant_emotion = 'sad'
            elif emotions['angry'] > 30:
                dominant_emotion = 'angry'

            # Convert the image to Base64
            with open(image_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        except Exception as e:
            logger.error(f"Error during emotion analysis: {str(e)}")
            return jsonify({"error": f"Error during analysis: {str(e)}"}), 500

        finally:
            # Clean up
            if os.path.exists(image_path):
                os.remove(image_path)

        return jsonify({
            "dominant_emotion": dominant_emotion,
            "emotions": emotions,
            "captured_image": img_base64
        })

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    # Check camera availability at startup
    camera, index = initialize_camera()
    if camera is not None:
        camera.release()
        logger.info(f"Camera check passed - working camera found at index {index}")
    else:
        logger.warning("No working camera found at startup")
    
    app.run(host='0.0.0.0', port=5000, debug=True)