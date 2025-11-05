from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("hotspot_cnn.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"
    file = request.files['file']
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]
    result = "Hotspot" if prediction > 0.5 else "Non-Hotspot"

    return f"<h2>Prediction: {result}</h2>"

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)