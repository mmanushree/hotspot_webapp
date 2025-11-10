from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# ✅ Create 'uploads' folder automatically (works on Render & locally)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Load CNN model safely
model = load_model(os.path.join(os.path.dirname(__file__), "hotspot_cnn.h5"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"
    file = request.files['file']
    if file.filename == '':
        return "No selected file!"

    # ✅ Save uploaded file inside uploads folder
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # ✅ Preprocess image for CNN
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # ✅ Prediction
        prediction = model.predict(img_array)[0][0]
        result = "Hotspot" if prediction > 0.5 else "Non-Hotspot"

        # ✅ Show result on webpage
        return render_template('index.html', prediction_text=f'Prediction: {result}')

    finally:
        # ✅ Auto-delete uploaded image after prediction
        if os.path.exists(filepath):
            os.remove(filepath)

# ✅ Proper settings for Render (no ngrok, no debug)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
