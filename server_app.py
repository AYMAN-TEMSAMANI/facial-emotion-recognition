import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import tf_keras as keras

app = Flask(__name__)

# Model path resolution
MODEL_PATH = os.path.join('models', 'best_cnn_model.h5')
print(f'[SYSTEM] Loading model from {MODEL_PATH} using tf_keras...')
model = keras.models.load_model(MODEL_PATH, compile=False)
print('[SYSTEM] Model loaded successfully.')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'online',
        'service': 'Facial Emotion Recognition API',
        'endpoints': {
            '/predict': 'POST request with image file multipart/form-data (key: file)'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded under key "file"'}), 400

    file = request.files['file']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'message': 'No face detected in image', 'predictions': []})

    results = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        preds = model.predict(roi, verbose=0)[0]
        label = EMOTIONS[preds.argmax()]
        confidence = float(np.max(preds))

        results.append({
            'emotion': label,
            'confidence': round(confidence, 4),
            'box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })

    return jsonify({'faces_detected': len(faces), 'predictions': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
