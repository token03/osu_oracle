import pickle
import sqlite3

import keras
import numpy as np
from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT id, category FROM beatmaps')
    beatmaps = cursor.fetchall()

    X = []
    y = []

    for beatmap_id, category in beatmaps:
        cursor.execute('SELECT x_diff, y_diff, time_diff, obj_type FROM beatmap_vectors WHERE beatmap_id = ?', (beatmap_id,))
        vectors = cursor.fetchall()
        if len(vectors) > 0:
            X.append(vectors)
            y.append(category)

    conn.close()

    max_length = max(len(seq) for seq in X)
    X_padded = pad_sequences(X, maxlen=max_length, dtype=np.float32, padding='post')
    y = np.array(y)

    return X_padded, y


def build_model(input_shape, num_classes):
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
    db_path = '/data/beatmaps.db'
    X, y = get_data(db_path)

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Pad the sequences with zeros to have the same length
    X_padded = pad_sequences(X, dtype='float32', padding='post')

    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

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
    model = build_model(input_shape, num_classes)

    model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, validation_data=(X_test, y_test_categorical))

    model.save('cnn_model.h5')


if __name__ == '__main__':
    main()