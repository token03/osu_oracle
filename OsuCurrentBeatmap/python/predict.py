from flask import Flask, request, jsonify
import time
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import requests
import pickle
import tempfile
from test_model import parse_osu_file  # Replace with the actual module
from test_model import test_model_on_beatmap_id  # Replace with the actual module

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    beatmap_id = request.json['beatmap_id']
    folders = request.json['folders']
    models = []
    start = time.time()
    for folder in folders:
      model_folder = folder + "/"
      model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.h5')]
      model = [load_model(model_path, compile=False) for model_path in model_paths]
      models.append(model) 
    end = time.time()
    max_slider_length = 500.0
    max_time_diff = 1000
    label_encoder_path = model_folder + "label_encoder.pkl"
    print(f"Loading Time: {round(end - start, 2)}s")

    predictions = []
    for i, model in enumerate(models):
      print("----------------------------------------------------")
      print("Model: " + folders[i])
      config = model[0].get_config() # Returns pretty much every information about your model
      max_sen = config["layers"][0]["config"]["batch_input_shape"][1] # returns a tuple of width, height and channels
      prediction = test_model_on_beatmap_id(beatmap_id, model, max_sen, max_slider_length, max_time_diff, label_encoder_path)
      predictions.append(prediction)

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
