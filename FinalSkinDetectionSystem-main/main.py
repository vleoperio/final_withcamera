from flask import Flask, render_template, send_from_directory, request
import os
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np


app = Flask(__name__)

from tensorflow.keras.models import load_model

# Define the absolute path to the model file
#PALITAN KUNG NASAAN UNG MODEL TEH
model_path = 'C:/Users/user/FinalSkinDetectionSystem-main/models/VGG16.h5'

# Load the model
model = load_model(model_path)

# Define class labels
class_labels = ['acne', 'eczema']  # Adjust the class labels as per your model


prescriptions = {
    'acne': [
        "Benzoyl peroxide",
        "Tretinoin",
        "Isotretinoin",
        "Adapalene",
        "Clindamycin",
        "Spironolactone",
        "Epiduo",
        "Minocycline",
        "Tazarotene",
        "Azelaic acid",
        "Doxycycline",
        "Erythromycin",
        "Salicylic acid",
        "Clascoterone",
        "Trimethoprim"
    ],
    'eczema': [
        "Calcineurin inhibitors",
        "Corticosteroids",
        "Cyclosporine",
        "Immunomodulators",
        "Methotrexate",
        "Antibiotics (e.g., Azathioprine)",
        "Janus kinase inhibitor",
        "Protopic"
    ]
}




@app.route('/')
def index():
    return render_template("index.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/assets/<path:folder>/<path:filename>')
def assets(folder, filename):
    return send_from_directory('assets', folder + '/' + filename)

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['imagefile']

        # Save the file to disk
        file_path = 'static/' + file.filename
        file.save(file_path)

        # Load the image
        img = image.load_img(file_path, target_size=(224, 224))

        # Preprocess the image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.  # Normalize pixel values

        # Make predictions
        predictions = model.predict(img_array)

       # Interpret predictions
        predicted_class = np.argmax(predictions[0])
        confidence_level = np.max(predictions[0])

        # Calculate confidence level as a percentage
        confidence_level = np.max(predictions[0]) * 100

        # Map predicted class to disease name
        if confidence_level >= 0.85:
            if predicted_class == 0:
                prediction = "Acne"
                prescription = prescriptions['acne']
            elif predicted_class == 1:
                prediction = "Eczema"
                prescription = prescriptions['eczema']
        else:
            prediction = "Undetermined"
            prescription = []

        return render_template('index.html', prediction=prediction, prescription=prescription, file_path=file_path, confidence=confidence_level)
       







if __name__ == '__main__':
    app.run(debug=True)