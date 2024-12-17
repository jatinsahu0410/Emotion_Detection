from flask import Flask, jsonify
from deepface import DeepFace
import cv2
import os
import time
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for the app

@app.route('/capture-emotion', methods=['GET'])
def capture_emotion():
    # Initialize the webcam
    webcam = cv2.VideoCapture(0)
    ret, frame = webcam.read()
    if not ret:
        webcam.release()
        return jsonify({"error": "Failed to capture image"}), 500

    # Create a unique image filename
    timestamp = int(time.time())
    image_path = f"captured_image_{timestamp}.jpg"
    cv2.imwrite(image_path, frame)

    try:
        # Analyze the image for emotions
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], detector_backend='mtcnn')
        emotions = analysis[0]['emotion']
        dominant_emotion = analysis[0]['dominant_emotion']

        # Update the dominant emotion based on thresholds
        if emotions['sad'] > 5:
            dominant_emotion = 'sad'
        elif emotions['angry'] > 30:
            dominant_emotion = 'angry'

        # Convert the image to Base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        webcam.release()
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({"error": f"Error during analysis: {str(e)}"}), 500

    # Clean up
    if os.path.exists(image_path):
        os.remove(image_path)
    webcam.release()

    # Return the analysis results and the captured image
    return jsonify({
        "dominant_emotion": dominant_emotion,
        "emotions": emotions,
        "captured_image": img_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
