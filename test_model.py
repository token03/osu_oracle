import pickle
import sys

import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

import osu_parser  # Import the parser script you provided


def test_model_on_osu_file(osu_file_path, model_path, max_sequence_length, label_encoder_path):
    # Load the model and the LabelEncoder
    model = load_model(model_path)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Parse the osu file and get the beatmap data
    beatmap_data = osu_parser.parse_osu_file(osu_file_path)
    if beatmap_data is None:
        print("Invalid .osu file.")
        return

    # Get the vectors and pad them
    beatmap_vectors = beatmap_data['vectors']
    beatmap_vectors_padded = pad_sequences([beatmap_vectors], dtype='float32', padding='post', maxlen=max_sequence_length)

    # Make a prediction using the model
    prediction = model.predict(beatmap_vectors_padded)

    # Convert the prediction to the original category label
    predicted_category_numerical = np.argmax(prediction, axis=-1)
    predicted_category = label_encoder.inverse_transform(predicted_category_numerical)

    print(f"Predicted category for {osu_file_path}: {predicted_category[0]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_osu_file.py path/to/osu/file")
        exit(1)

    osu_file_path = sys.argv[1]
    model_path = "cnn_model.h5"
    max_sequence_length = 2444  # Set this to the same value you used in the training script
    label_encoder_path = "label_encoder.pkl"  # Replace with the path to your saved LabelEncoder

    test_model_on_osu_file(osu_file_path, model_path, max_sequence_length, label_encoder_path)