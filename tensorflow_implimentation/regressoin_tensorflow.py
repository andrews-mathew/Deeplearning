import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Device setup (TensorFlow automatically handles GPU if available)
print(f"Using TensorFlow backend. GPU available: {tf.config.list_physical_devices('GPU')}")

# === Step 1: Load and Save Dataset ===
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
csv_path = "diabetes.csv"
df.to_csv(csv_path, index=False)
print(f"Dataset saved as {csv_path}")

# === Step 2: Preprocessing ===
data = pd.read_csv(csv_path)
X = data.drop(columns='target').values
y = data['target'].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)) # Keep y as 2D for TensorFlow's fit

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# === Step 3: Dataset Preparation for TensorFlow (no custom Dataset class needed) ===
# TensorFlow uses numpy arrays directly or tf.data.Dataset for more complex pipelines.
# For this case, numpy arrays are sufficient and simpler.

# === Step 4: Define Model ===
def create_regression_model():
    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),
        layers.Dense(32),
        layers.ReLU(),
        layers.Dense(1)
    ])
    return model

model = create_regression_model()
model.summary()

# === Step 5: Train Model ===
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

epochs = 1500
best_model_path = "best_diabetes_model_tf.h5" # Keras model save format

history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=0 # Suppress default Keras verbose output to use tqdm
)

# Manually print progress using tqdm (Keras fit has its own progress bar, but for consistency with PyTorch)
print("\nTraining completed.")
# Keras history object already contains epoch-wise loss
train_losses = history.history['loss']
val_losses = history.history['val_loss']
epochs_ran = len(train_losses)

for i in range(epochs_ran):
    print(f"Epoch {i+1}: Train Loss = {train_losses[i]:.4f}, Val Loss = {val_losses[i]:.4f}")

# Save the best model (EarlyStopping already restored best weights)
model.save(best_model_path)
print(f"Best model saved as {best_model_path}")

# === Step 6: Load Best Model for Evaluation and Inference ===
# When EarlyStopping is used with restore_best_weights=True, the model already has the best weights.
# We explicitly load it here to simulate the PyTorch behavior of loading a saved model.
loaded_model = models.load_model(best_model_path)
print(f"Best model loaded from {best_model_path}")

# Evaluate on test set
eval_results = loaded_model.evaluate(X_test, y_test, verbose=0)
avg_test_loss = eval_results

all_preds_scaled = loaded_model.predict(X_test)
all_preds = scaler_y.inverse_transform(all_preds_scaled).flatten()
all_actuals = scaler_y.inverse_transform(y_test).flatten() # y_test is already 2D

r2 = r2_score(all_actuals, all_preds)
print(f"Test Loss: {avg_test_loss:.4f}, R² Score: {r2:.3f}")

# === Step 7: Inference on 1 Sample ===
sample = X[0]  # original (not scaled)
sample_scaled = scaler_X.transform([sample])

# TensorFlow models expect batch dimension, so add an extra dimension
sample_scaled_tensor = tf.constant(sample_scaled, dtype=tf.float32)

pred_scaled = loaded_model.predict(sample_scaled_tensor)[0][0]
prediction = scaler_y.inverse_transform([[pred_scaled]])[0][0]
print(f"\n🎯 Predicted progression for sample 0: {prediction:.2f}")

# === Step 8: Inference on 10 Samples ===
samples_original = X[:10]
samples_scaled = scaler_X.transform(samples_original)
samples_scaled_tensor = tf.constant(samples_scaled, dtype=tf.float32)

preds_scaled = loaded_model.predict(samples_scaled_tensor)
preds = scaler_y.inverse_transform(preds_scaled)
print("\n📊 Predictions for first 10 samples:")
print(np.round(preds.flatten(), 2))

# === Step 9: Improved Visualization ===
plt.figure(figsize=(8, 8))
plt.scatter(all_actuals, all_preds, alpha=0.6, color='blue', edgecolors='w', s=100)
plt.plot([min(all_actuals), max(all_actuals)], [min(all_actuals), max(all_actuals)], 'r--', lw=2)
plt.xlabel("Actual Disease Progression", fontsize=12)
plt.ylabel("Predicted Disease Progression", fontsize=12)
plt.title(f"Predicted vs. Actual Disease Progression (Test Set, R² = {r2:.3f})", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Clean up the saved CSV and model file
# os.remove(csv_path)
# os.remove(best_model_path)