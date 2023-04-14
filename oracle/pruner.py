import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model

# Quantizing models
def prune_models(models):
    prune_models = []
    
    for model in models:
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        # Compute end step to finish pruning after 2 epochs.
        batch_size = 32
        epochs = 2
        validation_split = 0.1 # 10% of training set will be used for validation set. 

        num_train = X_train.shape[0] * (1 - validation_split)

        end_step = np.ceil(num_train / batch_size).astype(np.int32) * epochs

        # Define model for pruning.
        pruning_params = {
              'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                      final_sparsity=0.80,
                                                                      begin_step=0,
                                                                      end_step=end_step)
        }
        model_for_pruning = prune_low_magnitude(model, **pruning_params)

        # `prune_low_magnitude` requires a recompile.
        model_for_pruning.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])

        callbacks = [
          tfmot.sparsity.keras.UpdatePruningStep(),
        ]

        model_for_pruning.fit(X_train, y_train_categorical,
                          batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                          callbacks=callbacks)
        
        prune_models.append(model_for_pruning)
    
    return prune_models

pruned_models = prune_models(bagged_models)

# Save the bagging models using Keras's save function
for i, model in enumerate(pruned_models):
  model_for_export = tfmot.sparsity.keras.strip_pruning(model)
  tf.keras.models.save_model(model_for_export, f"/content/pruned_models/pruned_model_{i}.h5", include_optimizer=False)

start = time.time()
model_folder = "/content/bagged_cnn_models/"
model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.h5')]
bagged_models = [load_model(model_path) for model_path in model_paths]
end = time.time()
print(end - start)

start = time.time()
model_folder = "/content/pruned_models/"
model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.h5')]
stripped_pruned = [load_model(model_path) for model_path in model_paths]
end = time.time()
print(end - start)

import time

max_length = max(len(seq) for seq in X)
print(max_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', dtype='float32')
y_test_numerical = label_encoder.transform(y_test)

# Use the average prediction of the bagging models
y_preds = []
start = time.time()
for model in bagged_models:
    y_pred = model.predict(X_test)
    y_preds.append(y_pred)
end = time.time()
print(end - start)

y_preds_mean = np.mean(y_preds, axis=0)
y_preds_mean_numerical = np.argmax(y_preds_mean, axis=1)

score = accuracy_score(y_test_numerical, y_preds_mean_numerical)
cm = confusion_matrix(y_test_numerical, y_preds_mean_numerical)
report = classification_report(y_test_numerical, y_preds_mean_numerical)

print("Classification Report:\n", report)    
print("Confusion Matrix:\n", cm)
print(f"Accuracy: {score}")

# Use the average prediction of the pruned models
y_preds = []
start = time.time()
for model in stripped_pruned:
    y_pred = model.predict(X_test)
    y_preds.append(y_pred)
end = time.time()
print(end - start)

y_preds_mean = np.mean(y_preds, axis=0)
y_preds_mean_numerical = np.argmax(y_preds_mean, axis=1)

score = accuracy_score(y_test_numerical, y_preds_mean_numerical)
cm = confusion_matrix(y_test_numerical, y_preds_mean_numerical)
report = classification_report(y_test_numerical, y_preds_mean_numerical)

print("Classification Report:\n", report)    
print("Confusion Matrix:\n", cm)
print(f"Accuracy: {score}")