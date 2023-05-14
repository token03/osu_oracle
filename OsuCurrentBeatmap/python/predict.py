from flask import Flask, request, jsonify
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from test_model import test_model_on_beatmap_id  

app = Flask(__name__)

# Define the models and related parameters as global variables
models = []
max_slider_length = 500.0
max_time_diff = 1000
label_encoder_path = ""

# Load the models into memory when the Flask app starts
def load_models(folders):
    global models
    global label_encoder_path
    start = time.time()
    for folder in folders:
        model_folder = folder + "/"
        model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.h5')]
        model = [load_model(model_path, compile=False) for model_path in model_paths]
        models.append(model) 
    end = time.time()
    label_encoder_path = model_folder + "label_encoder.pkl"
    print(f"Loading Time: {round(end - start, 2)}s")

# Call the load_models function with your folders
load_models(["models"])  # Replace with your actual folders

@app.route('/predict', methods=['POST'])
def predict():
    beatmap_id = request.json['beatmap_id']

    predictions = []
    for i, model in enumerate(models):
        print("----------------------------------------------------")
        config = model[0].get_config() # Returns pretty much every information about your model
        max_sen = config["layers"][0]["config"]["batch_input_shape"][1] # returns a tuple of width, height and channels
        prediction = test_model_on_beatmap_id(beatmap_id, model, max_sen, max_slider_length, max_time_diff, label_encoder_path)
        predictions.append(prediction)
    print("----------------------------------------------------")


    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
