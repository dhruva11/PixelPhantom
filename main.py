import os
import uuid
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

# Create a Flask web app
covid_ct = Flask(__name__, static_url_path='/static')


# Set up model and upload directories
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Models'))
model_path = os.path.join(models_dir, 'model.keras')

# Load the model
try:
    model = load_model(model_path)
except Exception as e:
    print("Error loading model:", e)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
covid_ct.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the home page route
@covid_ct.route('/')
def index():
    return render_template('index.html')

# Prediction route
@covid_ct.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = os.path.join(covid_ct.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Preprocess image
        img = image.load_img(filename, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)[0][0]
        result_label = "COVID-19 Positive" if prediction > 0.5 else "COVID-19 Negative"
        confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)

        # Save and display result
        result_image_filename = f'result_{uuid.uuid4()}.jpg'
        result_image_path = os.path.join(covid_ct.config['UPLOAD_FOLDER'], result_image_filename)
        result_img = Image.fromarray((img_array.squeeze() * 255).astype(np.uint8))
        result_img.save(result_image_path)

        return render_template('result.html', prediction=result_label, confidence=confidence, result_image=result_image_filename)

    flash("Invalid file type")
    return redirect(request.url)

# Route to serve uploaded images
@covid_ct.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(covid_ct.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    covid_ct.run(debug=True, use_reloader=False)
