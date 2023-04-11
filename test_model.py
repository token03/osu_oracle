import os
import pickle
import sys
import tempfile

import numpy as np
import requests
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

from scripts import osu_parser


def test_model_on_beatmap_id(beatmap_id, bagged_models, max_sequence_length, label_encoder_path):
    # Load the label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Fetch the .osu file content from the Kitsune API
    url = f"https://kitsu.moe/api/osu/{beatmap_id}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching .osu file for beatmap ID {beatmap_id}: {response.status_code}")
        return

    osu_file_content = response.content

    # Save the .osu file content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(osu_file_content)
        temp_file_path = temp_file.name

    # Parse the temporary .osu file and get the beatmap data
    beatmap_data = parse_osu_file(temp_file_path, True)
    if beatmap_data is None:
        print("Invalid .osu file.")
        os.unlink(temp_file_path)  # Delete the temporary file
        return

    # Get the vectors and pad them
    beatmap_vectors = beatmap_data['vectors']
    #beatmap_vectors = np.delete(beatmap_vectors, 3, axis=1)
    beatmap_vectors_padded = pad_sequences([beatmap_vectors], dtype='float32', padding='post', maxlen=max_sequence_length)


    # Use the average prediction of the bagged models
    y_preds = []
    for model in bagged_models:
        y_pred = model.predict(beatmap_vectors_padded)
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
            print(f"Predicted category for beatmap {beatmap_id}: {category}, confidence: {confidence:.2f}")


from IPython.display import clear_output
import time

if __name__ == "__main__":
    model_paths = [f"/content/bagged_cnn_models/bagged_cnn_model_{i}.h5" for i in range(5)]  # Assuming you have saved 5 models
    max_sequence_length = 6948
    label_encoder_path = "/content/label_encoder.pkl"
    bagged_models = [load_model(model_path) for model_path in model_paths]
    while True:
        clear_output(wait=True)  # Clear the output
        print("----------------------------------------------------")
        print("Enter a beatmap ID to classify (or 'exit' to quit): ", end='')
        beatmap_id = input()
        if beatmap_id == "exit":
            break
        test_model_on_beatmap_id(beatmap_id, bagged_models, max_sequence_length, label_encoder_path)
        print("----------------------------------------------------\n")
        time.sleep(10)  # Add a short delay before clearing the output again
