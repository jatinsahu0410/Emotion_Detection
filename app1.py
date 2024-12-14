from flask import Flask, jsonify, Response
from deepface import DeepFace
import cv2
import os
import numpy as np
import base64
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

@app.route('/capture_emotion', methods=['GET'])
def capture_emotion():
    # Initialize webcam
    webcam = cv2.VideoCapture(0)
    
    # Capture image
    ret, frame = webcam.read()
    if not ret:
        webcam.release()
        return jsonify({"error": "Failed to capture image"}), 500

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply face detection using OpenCV (Haar cascades)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        webcam.release()
        return jsonify({"error": "No face detected"}), 400
    
    # Crop the face region
    x, y, w, h = faces[0]  # We take the first face detected
    face = frame[y:y+h, x:x+w]

    # Optional: Enhance the brightness and contrast of the image
    # Apply gamma correction (optional)
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = [((i / 255.0) ** invGamma) * 255 for i in range(256)]
    lookup_table = np.array(table, np.uint8)
    enhanced_face = cv2.LUT(face, lookup_table)

    # Save the captured and enhanced image temporarily
    image_path = "captured_image.jpg"
    cv2.imwrite(image_path, enhanced_face)

    # Analyze the image for emotions
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], detector_backend='mtcnn')
        dominant_emotion = analysis[0]['dominant_emotion']
        emotion_scores = analysis[0]['emotion']

        # Convert frame to base64 to send to the frontend
        _, buffer = cv2.imencode('.jpg', enhanced_face)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        webcam.release()
        return jsonify({"error": f"Error during analysis: {str(e)}"}), 500

    # Clean up the image file
    if os.path.exists(image_path):
        os.remove(image_path)
    webcam.release()

    # Return emotion data
    return jsonify({
        "dominant_emotion": dominant_emotion,
        "emotion_scores": emotion_scores,
        "captured_image": img_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
