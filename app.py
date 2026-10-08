from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load the lightweight model (<100MB)
MODEL_PATH = 'models/best_cnn_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

HTML_TEMPLATE = '''
<!doctype html>
<html>
<head><title>Emotion Recognition</title></head>
<body style="font-family: Arial; text-align: center; padding-top: 50px;">
    <h2>Facial Emotion Recognition API</h2>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Predict Emotion</button>
    </form>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=[0, -1])
    
    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))
    return jsonify({
        'emotion': EMOTIONS[idx],
        'confidence': float(preds[idx])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
