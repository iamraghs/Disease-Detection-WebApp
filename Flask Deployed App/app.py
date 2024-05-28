# '''

# import os
# from flask import Flask, redirect, render_template, request
# from PIL import Image
# from keras.models import load_model
# import numpy as np
# import pandas as pd

# # Load disease and supplement info
# disease_info = pd.read_csv('disease_info _disease_info.csv.csv', encoding='cp1252')
# supplement_info = pd.read_csv('supplement_info - supplement_info.csv.csv', encoding='cp1252')

# # Load your TensorFlow model
# model = load_model("model_best.hdf5")

# def predict_disease(image_path):
#     # Load and preprocess the image
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     input_data = np.expand_dims(np.array(image), axis=0)
#     input_data = input_data / 255.0  # Normalize input data
#     print(input_data)

#     # Perform prediction
#     output = model.predict(input_data)
#     index = np.argmax(output)

#     return index

# # Initialize Flask app
# app = Flask(__name__)

# # Define routes
# @app.route('/')
# def home_page():
#     return render_template('home.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact-us.html')

# @app.route('/index')
# def ai_engine_page():
#     return render_template('index.html')

# @app.route('/mobile-device')
# def mobile_device_detected_page():
#     return render_template('mobile-device.html')

# @app.route('/submit', methods=['POST'])
# def submit():
#     if request.method == 'POST':
#         # Get image file from form
#         image = request.files['image']
#         filename = image.filename
#         file_path = os.path.join('static/uploads', filename)
#         image.save(file_path)

#         # Perform prediction
#         pred_index = predict_disease(file_path)
        
        # # Retrieve disease and supplement information
        # disease_name = disease_info['disease_name'][pred_index]
        # description = disease_info['description'][pred_index]
        # possible_steps = disease_info['Possible Steps'][pred_index]
        # image_url = disease_info['image_url'][pred_index]
        # supplement_name = supplement_info['supplement name'][pred_index]
        # supplement_image_url = supplement_info['supplement image'][pred_index]
        # supplement_buy_link = supplement_info['buy link'][pred_index]

#         # Render template with prediction and information
#         return render_template('submit.html', title=disease_name, desc=description,
#                                prevent=possible_steps, image_url=image_url,
#                                sname=supplement_name, simage=supplement_image_url,
#                                buy_link=supplement_buy_link)

# @app.route('/market')
# def market():
#     return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
#                            supplement_name=list(supplement_info['supplement name']),
#                            disease=list(disease_info['disease_name']),
#                            buy=list(supplement_info['buy link']))

# if __name__ == '__main__':
#     app.run(debug=True)

#     '''


import os
from flask import Flask, render_template, request
from PIL import Image
from keras.models import load_model
import numpy as np
import pandas as pd
from flask import flash, redirect
from werkzeug.utils import secure_filename


# Load disease and supplement info
disease_info = pd.read_csv('disease_info-disease_info.csv - Worksheet.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_informations.csv', encoding='cp1252')

# Load your TensorFlow model
model = load_model("model_best.hdf5")

def predict_disease(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = np.expand_dims(np.array(image), axis=0)
    input_data = input_data / 255.0  # Normalize input data

    # Perform prediction
    output = model.predict(input_data)
    index = np.argmax(output)

    return index

# Initialize Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

# @app.route('/submit', methods=['POST'])
# def submit():
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             flash('No image part')
#             return redirect(request.url)
        
#         image = request.files['image']
#         if image.filename == '':
#             flash('No selected image')
#             return redirect(request.url)

#         filename = secure_filename(image.filename)
#         file_path = os.path.join('static/uploads', filename)
#         image.save(file_path)

#         pred_index = predict_disease(file_path)
        
#         disease_name = disease_info['disease_name'][pred_index]
#         description = disease_info['description'][pred_index]
        
#         # Get possible steps and remove duplicates
#         possible_steps = disease_info['Possible Steps'][pred_index]
#         unique_steps = set(possible_steps.split('\n'))
#         possible_steps = '\n'.join(unique_steps)
        
#         image_url = file_path  # Use uploaded image path
#         supplement_name = supplement_info['supplement name'][pred_index]
#         supplement_image_url = supplement_info['supplement image'][pred_index]
#         supplement_buy_link = supplement_info['buy link'][pred_index]


#         # Render template with prediction and information
#         return render_template('submit.html', title=disease_name, desc=description,
#                                prevent=possible_steps, image_url=image_url,
#                                sname=supplement_name, simage=supplement_image_url,
#                                buy_link=supplement_buy_link)


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image part')
            return redirect(request.url)
        
        image = request.files['image']
        if image.filename == '':
            flash('No selected image')
            return redirect(request.url)

        filename = secure_filename(image.filename)
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        pred_index = predict_disease(file_path)
        
        # disease_name = disease_info['disease_name'][pred_index]
        # description = disease_info['description'][pred_index]
        
        #   # Get possible steps and remove duplicates while preserving order
        # possible_steps = disease_info['Possible Steps'][pred_index]
        # unique_steps = []
        # seen_steps = set()
        # for step in possible_steps.split('\n'):
        #     if step not in seen_steps:
        #         unique_steps.append(step)
        #         seen_steps.add(step)
        # # Join the unique steps into a single string with numbered points
        # numbered_steps = '\n'.join([f"{i+1}. {step}" for i, step in enumerate(unique_steps)])
        
        # image_url = file_path  # Use uploaded image path
        # supplement_name = supplement_info['supplement name'][pred_index]
        # supplement_image_url = supplement_info['supplement image'][pred_index]
        # supplement_buy_link = supplement_info['buy link'][pred_index]


        # Retrieve disease and supplement information
        disease_name = disease_info['disease_name'][pred_index]
        description = disease_info['description'][pred_index]
        possible_steps = disease_info['Possible Steps'][pred_index]
        image_url = file_path  # Use uploaded image path
        supplement_name = supplement_info['supplement name'][pred_index]
        supplement_image_url = supplement_info['supplement image'][pred_index]
        supplement_buy_link = supplement_info['buy link'][pred_index]


        return render_template('submit.html', title=disease_name,desc=description, prevent=possible_steps,
                               image_url=image_url, sname=supplement_name,
                               simage=supplement_image_url, buy_link=supplement_buy_link)


@app.route('/market')
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)

