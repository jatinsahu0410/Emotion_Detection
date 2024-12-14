from flask import Flask, jsonify, Response
from deepface import DeepFace
import cv2
import os
import base64
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

@app.route('/capture_emotion', methods=['GET'])
def capture_emotion():
    # Initialize webcam
    webcam = cv2.VideoCapture(0)
    ret, frame = webcam.read()
    if not ret:
        webcam.release()
        return jsonify({"error": "Failed to capture image"}), 500

    # Save the captured image temporarily
    image_path = "captured_image.jpg"
    cv2.imwrite(image_path, frame)

    # Analyze the image for emotions
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], detector_backend='mtcnn')
        dominant_emotion = analysis[0]['dominant_emotion']
        emotion_scores = analysis[0]['emotion']

        # Convert frame to base64 to send to the frontend
        _, buffer = cv2.imencode('.jpg', frame)
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
