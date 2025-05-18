import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = pd.read_excel('Data.xlsx')

X = data[['x', 'y']]
y = data[['U', 'V']]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#print("Eğitim seti boyutu:", X_train.shape)
#print("Doğrulama seti boyutu:", X_val.shape)
#print("Test seti boyutu:", X_test.shape)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(2,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=200,
    batch_size=32
)


plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Eğitim Kayıp (MSE)')
plt.plot(history.history['val_loss'], label='Doğrulama Kayıp (MSE)')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Log Scale)')
plt.title('Eğitim ve Doğrulama Kayıp Değişimi')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('loss_plot.png')

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

model.save('model.keras')
print("Model kaydedildi: model.keras")

print(f"Eğitim seti R²: {train_metrics[0]:.4f}, MSE: {train_metrics[1]:.6f}, RMSE: {train_metrics[2]:.6f}, MAE: {train_metrics[3]:.6f}")
print(f"Doğrulama seti R²: {val_metrics[0]:.4f}, MSE: {val_metrics[1]:.6f}, RMSE: {val_metrics[2]:.6f}, MAE: {val_metrics[3]:.6f}")
print(f"Test seti R²: {test_metrics[0]:.4f}, MSE: {test_metrics[1]:.6f}, RMSE: {test_metrics[2]:.6f}, MAE: {test_metrics[3]:.6f}")