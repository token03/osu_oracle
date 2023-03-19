import pickle
import sqlite3
from collections import defaultdict

import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (LSTM, BatchNormalization, Conv1D, Dense, Dropout,
                          Flatten, MaxPooling1D)
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''SELECT b.id, b.category, v.x_diff, v.y_diff, v.time_diff
                      FROM beatmaps b
                      JOIN beatmap_vectors v ON b.id = v.beatmap_id''')
    rows = cursor.fetchall()

    X = defaultdict(list)
    y = {}

    for beatmap_id, category, x_diff, y_diff, time_diff in rows:
        X[beatmap_id].append((x_diff, y_diff, time_diff))
        y[beatmap_id] = category

    conn.close()

    return list(X.values()), np.array(list(y.values()))

def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def main():
    # Check if TensorFlow is built with GPU support
    if tf.test.is_built_with_cuda():
        print("TensorFlow is built with GPU support.")

    # Check if any GPUs are available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Number of available GPUs: {len(gpus)}")
        for gpu in gpus:
            print(f"GPU: {gpu}")
    else:
        print("No GPUs available.")
        
    db_path = './oracle/beatmaps.db'
    #X, y = get_data(db_path)
    #np.save('X.npy', X)
    #np.save('y.npy', y)

    X = np.load('X.npy', allow_pickle=True)
    y = np.load('y.npy')
    
    max_length = max(len(seq) for seq in X)
    X_array = np.zeros((len(X), max_length, 3))

    for i, seq in enumerate(X):
        for j, vector in enumerate(seq):
            X_array[i, j] = vector

    X_train, X_test, y_train, y_test = train_test_split(X_array, y, test_size=0.2, random_state=42)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    y_train_numerical = label_encoder.transform(y_train)
    y_test_numerical = label_encoder.transform(y_test)
    y_train_categorical = to_categorical(y_train_numerical)
    y_test_categorical = to_categorical(y_test_numerical)
    num_classes = len(label_encoder.classes_)
    input_shape = X_train.shape[1:]

    # Use the RNN model with LSTM layers
    model = build_cnn_model(input_shape, num_classes)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('cnn_model_best.h5', monitor='val_loss', save_best_only=True)

    model.fit(X_train, y_train_categorical, epochs=15, batch_size=16, validation_data=(X_test, y_test_categorical),
              callbacks=[early_stopping, model_checkpoint])

    model.save('cnn_model_final.h5')


if __name__ == '__main__':
    main()