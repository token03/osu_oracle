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


def test_model_on_beatmap_id(beatmap_id, model_path, max_sequence_length, label_encoder_path):
    # Load the model and the LabelEncoder
    model = load_model(model_path)
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
    beatmap_data = osu_parser.parse_osu_file(temp_file_path)
    if beatmap_data is None:
        print("Invalid .osu file.")
        os.unlink(temp_file_path)  # Delete the temporary file
        return

    # Get the vectors and pad them
    beatmap_vectors = beatmap_data['vectors']
    beatmap_vectors_padded = pad_sequences([beatmap_vectors], dtype='float32', padding='post', maxlen=max_sequence_length)

    # Make a prediction using the model
    prediction = model.predict(beatmap_vectors_padded)

    # Convert the prediction to the original category label
    predicted_category_numerical = np.argmax(prediction, axis=-1)
    predicted_category = label_encoder.inverse_transform(predicted_category_numerical)

    print(f"Predicted category for beatmap ID {beatmap_id}: {predicted_category[0]}")

    os.unlink(temp_file_path)  # Delete the temporary file


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py beatmap_id")
        exit(1)

    beatmap_id = sys.argv[1]
    model_path = "oracle/cnn_model.h5"
    max_sequence_length = 2444  # Set this to the same value you used in the training script
    label_encoder_path = "oracle/label_encoder.pkl"  # Replace with the path to your saved LabelEncoder

    test_model_on_beatmap_id(beatmap_id, model_path, max_sequence_length, label_encoder_path)
