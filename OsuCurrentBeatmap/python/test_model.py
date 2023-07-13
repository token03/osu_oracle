import gdown
import keras
import numpy as np
import os
import pickle
import requests
import sys
import tempfile
import time
import zipfile

from IPython.display import clear_output
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

if not os.path.exists('models'):
  url = 'https://drive.google.com/uc?id=14zLtVPcBDyLP-Rlj-2b6-mPIqGq-JY9J'
  output = 'models.zip'
  gdown.download(url, output, quiet=False)

  with zipfile.ZipFile("models.zip", 'r') as zip_ref:
      zip_ref.extractall("models")

def test_model(folders):
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

    while True:
        print("----------------------------------------------------")
        print("Enter a beatmap ID to classify (or 'exit' to quit): ", end='')
        beatmap_id = input()
        print("----------------------------------------------------")
        clear_output(wait=True)  # Clear the output
        if beatmap_id == "exit":
            break
        for i, model in enumerate(models):
          print("----------------------------------------------------")
          print("Model: " + folders[i])
          config = model[0].get_config() # Returns pretty much every information about your model
          max_sen = config["layers"][0]["config"]["batch_input_shape"][1] # returns a tuple of width, height and channels
          test_model_on_beatmap_id(beatmap_id, model, max_sen, max_slider_length, max_time_diff, label_encoder_path)

def parse_osu_file(file_path, max_slider_length = 1, max_time_diff = 1, print_info = False):
    data = {
        'beatmap_id': None,
        'hp_drain': None,
        'circle_size': None,
        'od': None,
        'ar': None,
        'slider_multiplier': None,
        'slider_tick': None,
        'hit_objects': [],
        'label': None,
    }

    parent_folder = os.path.dirname(file_path)
    data['label'] = os.path.basename(parent_folder)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        section = None

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]
                continue

            if section == 'Metadata':
                key, value = line.split(':', maxsplit=1)
                if key == 'Title' and print_info:
                    print("Title: " + value, end = ' ')
                if key == 'Artist' and print_info:
                    print("by " + value)
                if key == 'Creator' and print_info:
                    print("Mapper: " + value)
                if key == 'Version' and print_info:
                    print("Diffuculty: " + value)
                if key == 'BeatmapID':
                    data['beatmap_id'] = int(value)
                    print("ID: " + value)
                    print("----------------------------------------------------")
                    if data['beatmap_id'] is None:
                        return None
            elif section == 'Difficulty':
                key, value = line.split(':', maxsplit=1)
                value = float(value)
                if key == 'HPDrainRate':
                    data['hp_drain'] = value
                elif key == 'CircleSize':
                    data['circle_size'] = value
                elif key == 'OverallDifficulty':
                    data['od'] = value
                elif key == 'ApproachRate':
                    data['ar'] = value
                elif key == 'SliderMultiplier':
                    data['slider_multiplier'] = value
                elif key == 'SliderTickRate':
                    data['slider_tick'] = value
            elif section == 'HitObjects':  # Move this line one level back
                    obj_data = line.split(',')
                    hit_object_type = int(obj_data[3])

                    hit_circle_flag = 0b1
                    slider_flag = 0b10

                    if hit_object_type & hit_circle_flag:
                        hit_object = {
                            'x': int(obj_data[0]),
                            'y': int(obj_data[1]),
                            'time': min(1000, int(obj_data[2])),
                            'length': float(0), # slider len
                        }
                        data['hit_objects'].append(hit_object)
                    elif hit_object_type & slider_flag:
                        hit_object = {
                            'x': int(obj_data[0]),
                            'y': int(obj_data[1]),
                            'time': min(1000, int(obj_data[2])),
                            'length': min(500, float(obj_data[7])), # slider len 
                        }
                        data['hit_objects'].append(hit_object)
                        
    # Normalize the coordinates
    max_x, max_y = 512, 384
    for obj in data['hit_objects']:
        obj['x_norm'] = obj['x'] / max_x
        obj['y_norm'] = obj['y'] / max_y

    # Compute the time differences
    if data['hit_objects']:  # Add this condition
        for i, obj in enumerate(data['hit_objects'][1:], start=1):
            obj['time_diff'] = obj['time'] - data['hit_objects'][i - 1]['time']
        data['hit_objects'][0]['time_diff'] = 0

    vectors = []
    for i, obj in enumerate(data['hit_objects'][1:], start=1):
        prev_obj = data['hit_objects'][i - 1]
        x_diff = round(obj['x_norm'] - prev_obj['x_norm'], 4)
        y_diff = round(obj['y_norm'] - prev_obj['y_norm'], 4)
        time_diff = round(obj['time_diff'], 4)
        length = round(obj['length'], 4)
        vectors.append((x_diff, y_diff, time_diff / max_time_diff, length / max_slider_length))

    data['vectors'] = vectors
    return data

def test_model_on_beatmap_file(map_file_path, bagged_models, max_sequence_length, max_slider_length, max_time_diff, label_encoder_path):
    # Load the label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Parse the .osu file and get the beatmap data
    beatmap_data = parse_osu_file(map_file_path, max_slider_length, print_info = True)
    if beatmap_data is None:
        print("Invalid .osu file.")
        return

    # Get the vectors and pad them
    beatmap_vectors = beatmap_data['vectors']
    #beatmap_vectors = np.delete(beatmap_vectors, 3, axis=1)
    beatmap_vectors_padded = pad_sequences([beatmap_vectors], dtype='float32', padding='post', maxlen=max_sequence_length)


    # Use the average prediction of the bagged models
    y_preds = []
    for model in bagged_models:
        y_pred = model.predict(beatmap_vectors_padded, verbose = 0)
        y_preds.append(y_pred)

    y_preds_mean = np.mean(y_preds, axis=0)
    y_preds_mean_numerical = np.argmax(y_preds_mean, axis=1)

    # Determine the predicted category and confidence for each category
    categories = label_encoder.inverse_transform(range(len(y_preds_mean[0])))
    confidences = y_preds_mean[0]

    # Create a list of (category, confidence) tuples
    predictions = list(zip(categories, confidences))

    # Sort the predictions by confidence in descending order
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Print the predicted categories and confidence for confidences greater than 5%
    for category, confidence in sorted_predictions:
        if confidence > 0.05:
            print(f"{category} - confidence: {confidence:.2f}")
            
import json

def get_predictions_as_json(beatmap_id, bagged_models, max_sequence_length, max_slider_length, max_time_diff, label_encoder_path):
    # Load the label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Fetch the .osu file content from the Kitsune API
    url = f"https://osu.direct/api/osu/{beatmap_id}"
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching .osu file for beatmap ID {beatmap_id}: {response.status_code}")
        return

    osu_file_content = response.content

    # Save the .osu file content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(osu_file_content)
        temp_file_path = temp_file.name

    # Parse the temporary .osu file and get the beatmap data
    beatmap_data = parse_osu_file(temp_file_path, max_slider_length, print_info = True)
    if beatmap_data is None:
        print("Invalid .osu file.")
        os.unlink(temp_file_path)  # Delete the temporary file
        return

    # Get the vectors and pad them
    beatmap_vectors = beatmap_data['vectors']
    beatmap_vectors_padded = pad_sequences([beatmap_vectors], dtype='float32', padding='post', maxlen=max_sequence_length)

    # Use the average prediction of the bagged models
    y_preds = []
    for model in bagged_models:
        y_pred = model.predict(beatmap_vectors_padded, verbose = 0)
        y_preds.append(y_pred)

    y_preds_mean = np.mean(y_preds, axis=0)

    # Determine the predicted category and confidence for each category
    categories = label_encoder.inverse_transform(range(len(y_preds_mean[0])))
    confidences = y_preds_mean[0]

    # Create a list of (category, confidence) tuples
    predictions = list(zip(categories, confidences))

    # Sort the predictions by confidence in descending order
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Prepare the JSON object for predictions with confidences greater than 5%
    json_predictions = {category: float(confidence) for category, confidence in sorted_predictions if confidence > 0.05}

    # Return the JSON object
    return json.dumps(json_predictions)


def get_json_predictions(folders, beatmap_id):
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

    # Go through each model and get the predictions as a JSON object
    json_predictions = []
    for i, model in enumerate(models):
        print("----------------------------------------------------")
        print("Model: " + folders[i])
        config = model[0].get_config()  # Returns pretty much every information about your model
        max_sen = config["layers"][0]["config"]["batch_input_shape"][1]  # Returns a tuple of width, height and channels
        json_prediction = get_predictions_as_json(beatmap_id, model, max_sen, max_slider_length, max_time_diff, label_encoder_path)
        json_predictions.append(json_prediction)

    return json_predictions
            
import sys

def main(beatmap_id):
    folders = ["models"] # set this to "models"
    get_json_predictions(folders, beatmap_id)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_models.py [beatmap_id]")
    else:
        beatmap_id = sys.argv[1]
        main(beatmap_id)