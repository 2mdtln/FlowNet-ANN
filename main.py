"""
Neural Network Regression Model for Predicting U and V

- Loads data from 'Data.xlsx' and creates polynomial features.
- Splits data into train, validation, and test sets.
- Scales data using MinMaxScaler.
- Defines a Sequential Keras model with dropout and LeakyReLU layers.
- Trains the model and evaluates on validation and test data.
- Saves performance plots and predictions to files.
- Saves the trained model for future use.

Outputs:
- loss_plot.png: Training and validation loss across epochs.
- train_scatter.png, val_scatter.png, test_scatter.png: Scatter plots of true vs predicted values.
- prediction.xlsx: Excel file with true and predicted U, V values.
- model.keras: Saved Keras model.

Author: Muhammed T. (@2mdtln)
Date: 2025-05-18
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = pd.read_excel('Data.xlsx')

data['x2'] = data['x']**2
data['y2'] = data['y']**2
data['xy'] = data['x'] * data['y']

X = data[['x', 'y', 'x2', 'y2', 'xy']]
y = data[['U', 'V']]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)
print("Test set size:", X_test.shape)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

model = Sequential([
    Dense(256, activation='relu', input_shape=(5,)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    LeakyReLU(alpha=0.1),
    Dense(32, activation='relu'),
    LeakyReLU(alpha=0.1),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
 
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=230,
    batch_size=32,
)

y_train_pred_scaled = model.predict(X_train_scaled)
y_val_pred_scaled = model.predict(X_val_scaled)
y_test_pred_scaled = model.predict(X_test_scaled)

y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

y_train_orig = y_train
y_val_orig = y_val
y_test_orig = y_test

def performance_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mse, rmse, mae

train_metrics = performance_metrics(y_train_orig, y_train_pred)
val_metrics = performance_metrics(y_val_orig, y_val_pred)
test_metrics = performance_metrics(y_test_orig, y_test_pred)

plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (Log Scale)')
plt.title('Training and Validation Loss Change')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')

def plot_scatter(y_true, y_pred, title, metrics, filename):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min().min(), y_true.max().max()],
             [y_true.min().min(), y_true.max().max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    r2, mse, rmse, mae = metrics
    plt.text(0.05, 0.95, f"R²: {r2:.4f}\nMSE: {mse:.6f}\nRMSE: {rmse:.6f}\nMAE: {mae:.6f}",
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

plot_scatter(y_train_orig, y_train_pred, "Training Set Prediction Results", train_metrics, "train_scatter.png")
plot_scatter(y_val_orig, y_val_pred, "Validation Set Prediction Results", val_metrics, "val_scatter.png")
plot_scatter(y_test_orig, y_test_pred, "Test Set Prediction Results", test_metrics, "test_scatter.png")

X_all = pd.concat([X_train, X_val, X_test])[['x', 'y']].reset_index(drop=True)
y_all_orig = pd.concat([y_train, y_val, y_test]).reset_index(drop=True)
y_all_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
y_all_pred_df = pd.DataFrame(y_all_pred, columns=['U_pred', 'V_pred'])

result_df = pd.concat([X_all, y_all_orig, y_all_pred_df], axis=1)
result_df.to_excel('prediction.xlsx', index=False)

model.save('model.keras')
print("Model saved: model.keras")

print(f"Training set R²: {train_metrics[0]:.4f}, MSE: {train_metrics[1]:.6f}, RMSE: {train_metrics[2]:.6f}, MAE: {train_metrics[3]:.6f}")
print(f"Validation set R²: {val_metrics[0]:.4f}, MSE: {val_metrics[1]:.6f}, RMSE: {val_metrics[2]:.6f}, MAE: {val_metrics[3]:.6f}")
print(f"Test set R²: {test_metrics[0]:.4f}, MSE: {test_metrics[1]:.6f}, RMSE: {test_metrics[2]:.6f}, MAE: {test_metrics[3]:.6f}")