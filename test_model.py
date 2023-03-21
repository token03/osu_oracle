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
    beatmap_data = osu_parser.parse_osu_file(temp_file_path, True)
    if beatmap_data is None:
        print("Invalid .osu file.")
        os.unlink(temp_file_path)  # Delete the temporary file
        return

    # Get the vectors and pad them
    beatmap_vectors = beatmap_data['vectors']
    beatmap_vectors_trimmed = np.delete(beatmap_vectors, 3, axis=1)
    beatmap_vectors_padded = pad_sequences([beatmap_vectors_trimmed], dtype='float32', padding='post', maxlen=max_sequence_length)


    # Make a prediction using the model
    prediction = model.predict(beatmap_vectors_padded)

    # Get the predicted category indices and their corresponding confidence scores
    predicted_category_indices = np.argsort(-prediction, axis=-1)
    predicted_category_confidences = np.sort(prediction, axis=-1)[:, ::-1]

    # Get the labels for the first and second most confident predictions
    predicted_category = label_encoder.inverse_transform(predicted_category_indices[:, 0])
    second_predicted_category = label_encoder.inverse_transform(predicted_category_indices[:, 1])

    # Get the confidence scores for the first and second most confident predictions
    predicted_confidence = predicted_category_confidences[:, 0]
    second_predicted_confidence = predicted_category_confidences[:, 1]

    print(f"Predicted category for beatmap ID {beatmap_id}: {predicted_category[0]}")
    print(f"Confidence score for the predicted category: {predicted_confidence[0]}")
    print(f"Second most confident prediction: {second_predicted_category[0]}")
    print(f"Confidence score for the second most confident prediction: {second_predicted_confidence[0]}")

    os.unlink(temp_file_path)  # Delete the temporary file


if __name__ == "__main__":
    
    model_path = "oracle/cnn_model_final.h5"
    max_sequence_length = 6948  # Set this to the same value you used in the training script
    label_encoder_path = "oracle/label_encoder.pkl"  # Replace with the path to your saved LabelEncoder
    if len(sys.argv) != 2:
            print("Insert beatmap_id:")
            beatmap_id = input()
            test_model_on_beatmap_id(beatmap_id, model_path, max_sequence_length, label_encoder_path)
            exit(1)
    
    beatmap_id = sys.argv[1]

    test_model_on_beatmap_id(beatmap_id, model_path, max_sequence_length, label_encoder_path)
    exit(1)
