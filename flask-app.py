import os
import requests
import json
from flask import Flask, render_template, url_for, redirect, request
from flask_cors import CORS, cross_origin


# Imporiting Necessary Libraries
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results
from io import BytesIO

def load_model(path):
    
    # Xception Model
    xception_model = tf.keras.models.Sequential([
    tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4,activation='softmax')
    ])


    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet',input_shape=(512, 512, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4,activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))

    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)

    outputs = tf.keras.layers.average([densenet_output, xception_output])


    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Loading the Weights of Model
    model.load_weights(path)
    
    return model



def predict():
    print("Predict triggered")
    model = load_model('model_weights.h5')
    print("model loaded")
    response = requests.get("http://192.168.1.1/saved-photo")
    image = Image.open(BytesIO(response.content))
    # image = Image.open("C:\\Users\\91888\\OneDrive\\Desktop\\plant.jpg")
    print("image loaded")
    # st.image(np.array(Image.fromarray(np.array(image)).resize((700, 400), Image.ANTIALIAS)), width=None)
    image = clean_image(image)
    predictions, predictions_arr = get_prediction(model, image)
    result = make_results(predictions, predictions_arr) 
    print(result)  
    return result
        


app = Flask(__name__)


cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin()
def hello():
	return "Welcome"

@app.route('/predict')
@cross_origin()
def trigger():
      r = requests.get('http://192.168.1.1/capture')
      result = predict()
      return result


if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port,use_reloader = False)