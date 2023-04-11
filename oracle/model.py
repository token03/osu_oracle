import pickle
import joblib
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
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


def build_cnn_model(input_shape, num_classes, learning_rate=0.001, dropout_rate=0.5, l2_reg=0.001):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape, kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_and_evaluate(X_train, y_train, X_test, y_test, input_shape, num_classes):
    model = build_cnn_model(input_shape, num_classes)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('cnn_model_best.h5', monitor='val_loss', save_best_only=True)

    model.fit(X_train, y_train, epochs=25, batch_size=16, validation_data=(X_test, y_test),
              callbacks=[early_stopping, model_checkpoint])

    return model

def main():
    # try:
    #     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    #     print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    # except ValueError:
    #     raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    # tf.config.experimental_connect_to_cluster(tpu)
    # tf.tpu.experimental.initialize_tpu_system(tpu)
    # tpu_strategy = tf.distribute.TPUStrategy(tpu)

    X = np.load('X.npy', allow_pickle=True)
    y = np.load('y.npy')

    max_length = max(len(seq) for seq in X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Pad sequences
    X_train = pad_sequences(X_train, maxlen=max_length, padding='post', dtype='float32')
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post', dtype='float32')

    input_shape = X_train.shape[1:]

    n_estimators = 5
    bagging_models = []
    for i in range(n_estimators):
        X_bagging_train, _, y_bagging_train, _ = train_test_split(
            X_train, y_train_categorical, test_size=0.5, stratify=y_train_numerical, random_state=42+i
        )
        model = train_and_evaluate(X_bagging_train, y_bagging_train, X_test, y_test_categorical, input_shape, num_classes)
        bagging_models.append(model)

    # Use the average prediction of the bagging models
    y_preds = []
    for model in bagging_models:
        y_pred = model.predict(X_test)
        y_preds.append(y_pred)

    y_preds_mean = np.mean(y_preds, axis=0)
    y_preds_mean_numerical = np.argmax(y_preds_mean, axis=1)

    score = accuracy_score(y_test_numerical, y_preds_mean_numerical)
    print(f"Accuracy: {score}")

    # Save the bagging models using Keras's save function
    for i, model in enumerate(bagging_models):
        model.save(f'/content/bagged_cnn_model_{i}.h5')

if __name__ == '__main__':
    main()