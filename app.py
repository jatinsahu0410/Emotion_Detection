from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import time
from flask_cors import CORS
import logging
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/analyze-emotion', methods=['POST'])
def analyze_emotion():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode the base64 image
        image_data = data['image'].split(',')[1]  # Split to remove the data:image/png;base64, prefix
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # Convert image to RGB mode if it has an alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Save the image temporarily
        timestamp = int(time.time())
        image_path = f"temp_image_{timestamp}.jpg"
        image.save(image_path, format='JPEG')

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
            if emotions['sad'] > 3:
                dominant_emotion = 'sad'
            elif emotions['angry'] > 30:
                dominant_emotion = 'angry'

            return jsonify({
                "dominant_emotion": dominant_emotion,
                "emotions": emotions
            })

        except Exception as e:
            logger.error(f"Error during emotion analysis: {str(e)}")
            return jsonify({"error": f"Error during analysis: {str(e)}"}), 500

        finally:
            # Clean up - remove temporary file
            if os.path.exists(image_path):
                os.remove(image_path)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)